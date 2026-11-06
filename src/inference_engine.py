#!/usr/bin/env python3
"""
Industrial Biofouling Inference & Confidence-Based Human-in-the-Loop (HITL) Engine
---------------------------------------------------------------------------------
Features:
1. Multi-backend inference: Supports PyTorch models/checkpoints and ONNX Runtime sessions.
2. High-resolution preprocessing (380x380) with strict color preservation.
3. Full probability vector, Shannon entropy, and prediction confidence calculations.
4. Operational HITL Safety Gating (e.g. 75% threshold) routing uncertain samples to
   a 'Human Expert Review Required' escalation queue.
5. Batch auditing utility for operational SLA validation on test/production manifests.

Author: Principal AI Research Engineer
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.dataset import (
    IDX_TO_LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_TO_IDX,
    get_val_transforms,
)
from src.model import BiofoulingClassifier, create_biofouling_model
from src.tta import TestTimeAugmentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingInferenceEngine")


class BiofoulingInferenceResult:
    """Structured container for single-sample inference & gating results."""

    def __init__(
        self,
        predicted_class: str,
        predicted_idx: int,
        confidence: float,
        probabilities: Dict[str, float],
        entropy: float,
        gating_decision: str,
        human_review_required: bool,
        review_reason: str,
        latency_ms: float,
        image_path: Optional[str] = None,
        true_class: Optional[str] = None,
        true_idx: Optional[int] = None,
    ):
        self.predicted_class = predicted_class
        self.predicted_idx = predicted_idx
        self.confidence = confidence
        self.probabilities = probabilities
        self.entropy = entropy
        self.gating_decision = gating_decision
        self.human_review_required = human_review_required
        self.review_reason = review_reason
        self.latency_ms = latency_ms
        self.image_path = image_path
        self.true_class = true_class
        self.true_idx = true_idx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "predicted_idx": self.predicted_idx,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "gating_decision": self.gating_decision,
            "human_review_required": self.human_review_required,
            "review_reason": self.review_reason,
            "latency_ms": self.latency_ms,
            "true_class": self.true_class,
            "true_idx": self.true_idx,
            "prob_Low": self.probabilities.get("Low", 0.0),
            "prob_Moderate": self.probabilities.get("Moderate", 0.0),
            "prob_High": self.probabilities.get("High", 0.0),
            "prob_Critical": self.probabilities.get("Critical", 0.0),
        }

    def __repr__(self) -> str:
        status_tag = "[REVIEW REQUIRED]" if self.human_review_required else "[AUTOMATED OK]"
        return (
            f"BiofoulingInferenceResult({status_tag} Pred: {self.predicted_class} "
            f"(Conf: {self.confidence*100:.1f}%), Decision: {self.gating_decision})"
        )


def calculate_entropy(probs: np.ndarray) -> float:
    """Compute normalized Shannon entropy in [0, 1] as an uncertainty indicator."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))  # log2(4) = 2.0
    return float(entropy / max_entropy)


# Convenient alias
InferenceResult = BiofoulingInferenceResult


class BiofoulingInferenceEngine:
    """
    Industrial inference engine with confidence-based Human-in-the-Loop gating.
    Supports PyTorch and ONNX Runtime inference backends.
    """

    def __init__(
        self,
        model_or_path: Optional[Union[str, Path, torch.nn.Module]] = None,
        device: Optional[torch.device] = None,
        image_size: int = 380,
        confidence_threshold: float = 0.75,
        use_tta: bool = False,
        backend: str = "auto",  # 'auto', 'pytorch', 'onnx'
        model_path: Optional[Union[str, Path, torch.nn.Module]] = None,
        model: Optional[Union[str, Path, torch.nn.Module]] = None,
    ):
        """
        Args:
            model_or_path: PyTorch nn.Module, path to .pth checkpoint, or path to .onnx file.
            device: Target torch.device (defaults to CUDA if available, else CPU).
            image_size: Input resolution (default: 380).
            confidence_threshold: Operational confidence threshold (default: 0.75).
            use_tta: If True, uses 8-view D4 spatial Test-Time Augmentation.
            backend: Execution engine backend ('auto', 'pytorch', 'onnx').
        """
        target_model = model_or_path or model_path or model
        if target_model is None:
            raise ValueError("Must provide model_or_path, model_path, or model.")
        model_or_path = target_model
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.use_tta = use_tta
        self.val_transforms = get_val_transforms(image_size=image_size)

        # Detect backend
        if backend == "auto":
            if isinstance(model_or_path, (str, Path)) and str(model_or_path).endswith(".onnx"):
                self.backend = "onnx"
            else:
                self.backend = "pytorch"
        else:
            self.backend = backend.lower()

        if self.backend == "onnx":
            if ort is None:
                raise ImportError("onnxruntime is required for ONNX backend. Run 'pip install onnxruntime'.")
            onnx_path = Path(model_or_path)
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
            # Use CPU or CUDA execution provider
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            self.pytorch_model = None
            self.device = torch.device("cpu")
            logger.info(f"Initialized ONNX Runtime session: {onnx_path} (Providers: {self.ort_session.get_providers()})")
        else:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if isinstance(model_or_path, torch.nn.Module):
                self.pytorch_model = model_or_path.to(self.device)
            else:
                ckpt_path = Path(model_or_path)
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model_name = checkpoint.get("model_name", "efficientnet_b3")
                drop_rate = checkpoint.get("drop_rate", 0.4)
                self.pytorch_model = create_biofouling_model(
                    model_name=model_name,
                    num_classes=4,
                    pretrained=False,
                    drop_rate=drop_rate,
                ).to(self.device)
                self.pytorch_model.load_state_dict(checkpoint["model_state_dict"])
            self.pytorch_model.eval()
            self.ort_session = None
            self.tta_augmentor = TestTimeAugmentor(self.pytorch_model, device=self.device) if use_tta else None
            logger.info(f"Initialized PyTorch BiofoulingInferenceEngine on {self.device.type.upper()} (TTA={use_tta})")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess an image input into normalized float32 tensor/array (3, H, W)."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input if image_input.ndim == 3 else image_input.squeeze()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        augmented = self.val_transforms(image=img_np)
        return augmented["image"]  # Albumentations ToTensorV2 returns torch.Tensor (3, H, W)

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Compute normalized Shannon entropy in [0, 1] as an uncertainty indicator."""
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))  # log2(4) = 2.0
        return float(entropy / max_entropy)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        true_label: Optional[Union[str, int]] = None,
        custom_threshold: Optional[float] = None,
    ) -> BiofoulingInferenceResult:
        """
        Execute single-image inference with confidence assessment and HITL gating.
        """
        threshold = custom_threshold if custom_threshold is not None else self.confidence_threshold
        t0 = time.perf_counter()

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None
        img_tensor = self.preprocess_image(image_input)  # shape (3, H, W)

        if self.backend == "onnx":
            input_batch = img_tensor.unsqueeze(0).numpy()  # (1, 3, H, W)
            ort_inputs = {self.input_name: input_batch}
            logits = self.ort_session.run([self.output_name], ort_inputs)[0]  # (1, 4)
            logits_tensor = torch.from_numpy(logits)
            probs_np = F.softmax(logits_tensor, dim=1).squeeze(0).numpy()
        else:
            img_batch = img_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.use_tta and self.tta_augmentor:
                    tta_probs, _ = self.tta_augmentor.predict_batch(img_batch, num_views=8)
                    probs_np = tta_probs.squeeze(0).cpu().numpy()
                else:
                    logits = self.pytorch_model(img_batch)
                    probs_np = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract predictions and confidence metrics
        pred_idx = int(np.argmax(probs_np))
        pred_class = IDX_TO_LABEL[pred_idx]
        confidence = float(probs_np[pred_idx])
        entropy = self._calculate_entropy(probs_np)

        prob_dict = {IDX_TO_LABEL[i]: float(probs_np[i]) for i in range(len(probs_np))}

        # Sorted classes by probability
        sorted_indices = np.argsort(probs_np)[::-1]
        top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
        top1_class, top2_class = IDX_TO_LABEL[top1_idx], IDX_TO_LABEL[top2_idx]
        top1_p, top2_p = probs_np[top1_idx], probs_np[top2_idx]

        # Human-in-the-Loop Operational Gating Logic
        if confidence >= threshold:
            gating_decision = "AUTOMATED_DECISION"
            human_review_required = False
            review_reason = (
                f"Confidence ({confidence * 100:.1f}%) meets or exceeds operational "
                f"safety threshold ({threshold * 100:.1f}%)."
            )
        else:
            gating_decision = "HUMAN_EXPERT_REVIEW_REQUIRED"
            human_review_required = True
            review_reason = (
                f"Confidence ({confidence * 100:.1f}%) is below operational safety "
                f"threshold ({threshold * 100:.1f}%). Borderline decision between "
                f"'{top1_class}' ({top1_p * 100:.1f}%) and '{top2_class}' ({top2_p * 100:.1f}%)."
            )

        # Ground truth mapping if provided
        true_class_str = None
        true_idx_val = None
        if true_label is not None:
            if isinstance(true_label, int):
                true_idx_val = true_label
                true_class_str = IDX_TO_LABEL.get(true_label, "Unknown")
            elif isinstance(true_label, str):
                true_class_str = true_label
                true_idx_val = LABEL_TO_IDX.get(true_label, -1)

        return BiofoulingInferenceResult(
            predicted_class=pred_class,
            predicted_idx=pred_idx,
            confidence=confidence,
            probabilities=prob_dict,
            entropy=entropy,
            gating_decision=gating_decision,
            human_review_required=human_review_required,
            review_reason=review_reason,
            latency_ms=latency_ms,
            image_path=image_path_str,
            true_class=true_class_str,
            true_idx=true_idx_val,
        )

    predict_image = predict

    def audit_dataset_split(
        self,
        csv_file_or_df: Union[str, Path, pd.DataFrame],
        root_dir: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive Human-in-the-Loop confidence audit over an entire dataset split.
        """
        if isinstance(csv_file_or_df, (str, Path)):
            df = pd.read_csv(csv_file_or_df)
            csv_name = Path(csv_file_or_df).name
        else:
            df = csv_file_or_df.copy()
            csv_name = "DataFrame"

        root_path = Path(root_dir) if root_dir else PROJECT_ROOT
        applied_threshold = threshold if threshold is not None else self.confidence_threshold

        logger.info(
            f"Starting Confidence-Gating Audit on {len(df)} samples ({csv_name}) "
            f"| Threshold: {applied_threshold * 100:.1f}% | Backend: {self.backend.upper()}"
        )

        results: List[BiofoulingInferenceResult] = []
        for idx, row in df.iterrows():
            img_rel_path = row["filename"]
            img_full_path = root_path / img_rel_path
            if not img_full_path.exists():
                img_full_path = Path(img_rel_path)

            true_lbl = row.get("class_label", None) or row.get("label_idx", None)
            res = self.predict(
                image_input=img_full_path,
                true_label=true_lbl,
                custom_threshold=applied_threshold,
            )
            results.append(res)

        res_dicts = [r.to_dict() for r in results]
        audit_df = pd.DataFrame(res_dicts)

        # Audit Statistics
        total_samples = len(audit_df)
        automated_df = audit_df[~audit_df["human_review_required"]]
        review_df = audit_df[audit_df["human_review_required"]]

        automated_count = len(automated_df)
        review_count = len(review_df)

        automated_coverage = (automated_count / total_samples) * 100 if total_samples > 0 else 0.0
        review_rate = (review_count / total_samples) * 100 if total_samples > 0 else 0.0

        # Automated accuracy
        has_ground_truth = "true_class" in audit_df.columns and audit_df["true_class"].notna().any()
        auto_acc = 0.0
        total_acc = 0.0
        review_acc = 0.0

        if has_ground_truth:
            audit_df["is_correct"] = audit_df["predicted_class"] == audit_df["true_class"]
            total_acc = (audit_df["is_correct"].sum() / total_samples) * 100
            if automated_count > 0:
                auto_acc = (automated_df["predicted_class"] == automated_df["true_class"]).sum() / automated_count * 100
            if review_count > 0:
                review_acc = (review_df["predicted_class"] == review_df["true_class"]).sum() / review_count * 100

        avg_latency = float(audit_df["latency_ms"].mean())
        avg_conf = float(audit_df["confidence"].mean())

        audit_summary = {
            "total_samples": total_samples,
            "threshold": applied_threshold,
            "automated_count": automated_count,
            "automated_coverage_pct": automated_coverage,
            "review_count": review_count,
            "review_rate_pct": review_rate,
            "total_accuracy_pct": total_acc,
            "automated_accuracy_pct": auto_acc,
            "review_cohort_acc_pct": review_acc,
            "avg_confidence": avg_conf,
            "avg_latency_ms": avg_latency,
            "results_df": audit_df,
        }

        return audit_summary


def format_audit_report(summary: Dict[str, Any]) -> str:
    """Generate a clean ASCII / markdown table representation of the audit summary."""
    df = summary["results_df"]
    lines = [
        "=" * 85,
        "          CONFIDENCE-BASED HUMAN-IN-THE-LOOP (HITL) OPERATIONAL AUDIT REPORT          ",
        "=" * 85,
        f"Operational Safety Threshold: {summary['threshold'] * 100:.1f}%",
        f"Total Evaluated Test Samples: {summary['total_samples']}",
        f"Average Inference Latency:   {summary['avg_latency_ms']:.2f} ms / sample",
        f"Mean Prediction Confidence:  {summary['avg_confidence'] * 100:.2f}%",
        "-" * 85,
        f"Automated Decisions Passed:  {summary['automated_count']} ({summary['automated_coverage_pct']:.1f}% Coverage)",
        f"Automated Cohort Accuracy:   {summary['automated_accuracy_pct']:.2f}%",
        f"Human Review Escalations:    {summary['review_count']} ({summary['review_rate_pct']:.1f}% Escalation Rate)",
        f"Total Model Raw Accuracy:    {summary['total_accuracy_pct']:.2f}%",
        "=" * 85,
    ]

    # Show review queue items if any
    review_items = df[df["human_review_required"]]
    if len(review_items) > 0:
        lines.append("\n[ESCALATED SAMPLES QUEUE - HUMAN EXPERT REVIEW REQUIRED]:")
        review_table = []
        for _, row in review_items.iterrows():
            fname = Path(row["image_path"]).name if row["image_path"] else "N/A"
            probs = f"L:{row['prob_Low']:.2f}|M:{row['prob_Moderate']:.2f}|H:{row['prob_High']:.2f}|C:{row['prob_Critical']:.2f}"
            review_table.append([
                fname,
                row["true_class"] or "N/A",
                row["predicted_class"],
                f"{row['confidence'] * 100:.1f}%",
                f"{row['entropy']:.2f}",
                probs,
            ])
        headers = ["Filename", "True Label", "Model Pred", "Confidence", "Entropy", "Probabilities [L|M|H|C]"]
        if tabulate:
            lines.append(tabulate(review_table, headers=headers, tablefmt="grid"))
        else:
            fmt = "{:<22} | {:<10} | {:<10} | {:<10} | {:<8} | {:<30}"
            lines.append(fmt.format(*headers))
            lines.append("-" * 95)
            for r in review_table:
                lines.append(fmt.format(*r))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Biofouling HITL Inference Engine")
    parser.add_argument("--model", type=str, default="checkpoints/best_model_v3.pth", help="Checkpoint (.pth) or ONNX (.onnx) path")
    parser.add_argument("--image", type=str, default=None, help="Single image path for inference")
    parser.add_argument("--csv", type=str, default="data/splits_cleaned/test.csv", help="CSV split for batch audit")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence gating threshold (0.0 to 1.0)")
    parser.add_argument("--use-tta", action="store_true", help="Enable 4-view Test-Time Augmentation")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx"], help="Backend")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output path to save audit CSV")

    args = parser.parse_args()

    engine = BiofoulingInferenceEngine(
        model_or_path=args.model,
        image_size=380,
        confidence_threshold=args.threshold,
        use_tta=args.use_tta,
        backend=args.backend,
    )

    if args.image:
        result = engine.predict(args.image)
        print("\n" + "=" * 60)
        print("                 SINGLE IMAGE INFERENCE RESULT                 ")
        print("=" * 60)
        print(f"Image:            {args.image}")
        print(f"Prediction:       {result.predicted_class} (Index: {result.predicted_idx})")
        print(f"Confidence:       {result.confidence * 100:.2f}%")
        print(f"Shannon Entropy:  {result.entropy:.4f}")
        print(f"Gating Status:    {result.gating_decision}")
        print(f"Review Required:  {result.human_review_required}")
        print(f"Review Rationale: {result.review_reason}")
        print(f"Latency:          {result.latency_ms:.2f} ms")
        print("Class Probabilities:")
        for cls_name, p in result.probabilities.items():
            print(f"  - {cls_name:<10}: {p * 100:.2f}%")
        print("=" * 60 + "\n")
    else:
        summary = engine.audit_dataset_split(csv_file_or_df=args.csv, threshold=args.threshold)
        report = format_audit_report(summary)
        print("\n" + report + "\n")

        if args.output_csv:
            summary["results_df"].to_csv(args.output_csv, index=False)
            logger.info(f"Saved audit results to {args.output_csv}")


if __name__ == "__main__":
    main()

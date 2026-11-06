#!/usr/bin/env python3
"""
Incremental Fine-Tuning & Active Learning Retraining Pipeline (v4.1)
---------------------------------------------------------------------
1. Automatically ingests newly resolved human review queue items into data/expanded/.
2. Loads current baseline checkpoint (EfficientNet-B3 380x380).
3. Freezes backbone layers to preserve foundational visual representations while
   optimizing the classification head with AdamW, class weighting, and label smoothing.
4. Validates on independent validation split, recording Macro-F1 and per-class metrics.
5. Automatically exports fine-tuned weights to production ONNX format with parity verification.

Author: Principal ML Operations & Deep Learning Engineer
"""

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from active_learning import ingest_resolved_samples
from src.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    BiofoulingDataset,
    get_hardened_train_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.export_onnx import export_to_onnx
from src.model import BiofoulingClassifier, create_biofouling_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingRetrainer")

# Default Paths
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_v4.pth"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "biofouling_model.onnx"
DEFAULT_INDUSTRIAL_VAL_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "val_industrial.csv"
DEFAULT_LEGACY_VAL_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "val.csv"
DEFAULT_VAL_CSV = DEFAULT_INDUSTRIAL_VAL_CSV if DEFAULT_INDUSTRIAL_VAL_CSV.exists() else DEFAULT_LEGACY_VAL_CSV
DEFAULT_EXPANDED_TRAIN_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "train_expanded.csv"


def calculate_class_weights(train_df: pd.DataFrame, num_classes: int = 4) -> torch.Tensor:
    """Compute balanced class weights from label distribution."""
    labels = train_df["label_idx"].values
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Computed Class Weights: {dict(zip([IDX_TO_LABEL[i] for i in range(num_classes)], weights.round(4)))}")
    return class_weights


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate model on validation split.

    Returns:
        (val_loss, val_acc, val_macro_f1, confusion_matrix)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = float(np.mean(np.array(all_preds) == np.array(all_targets))) if total > 0 else 0.0
    val_macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(4)))

    return val_loss, val_acc, val_macro_f1, cm


def fine_tune_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_onnx_path: Path = DEFAULT_ONNX_PATH,
    val_csv_path: Path = DEFAULT_VAL_CSV,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.04,
    label_smoothing: float = 0.1,
    image_size: int = 380,
    device_name: Optional[str] = None,
    db_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute active learning dataset ingestion, fine-tuning, validation, and ONNX export.

    Args:
        checkpoint_path: Path to baseline PyTorch model weights.
        output_onnx_path: Destination for exported ONNX engine.
        val_csv_path: Path to validation set manifest.
        epochs: Number of incremental fine-tuning epochs.
        batch_size: Batch size for fine-tuning.
        learning_rate: Learning rate for classification head optimization.
        weight_decay: AdamW regularization weight decay.
        label_smoothing: Cross-entropy label smoothing epsilon.
        image_size: Input spatial resolution (380x380).
        device_name: 'cuda', 'cpu', or None (auto-detect).
        db_session: Optional active SQLAlchemy database session.

    Returns:
        Structured dictionary of training telemetry, validation metrics, and export paths.
    """
    t_start = time.time()
    checkpoint_path = Path(checkpoint_path)
    val_csv_path = Path(val_csv_path)
    output_onnx_path = Path(output_onnx_path)

    # 1. Device Selection
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing active learning fine-tuning on device: {device}")

    # 2. Ingest Active Learning Data
    ingestion_stats = ingest_resolved_samples(db=db_session)
    expanded_csv = Path(ingestion_stats["expanded_manifest_path"])

    if not expanded_csv.exists():
        raise FileNotFoundError(f"Expanded training manifest missing: {expanded_csv}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation manifest missing: {val_csv_path}")

    train_df = pd.read_csv(expanded_csv)
    logger.info(f"Loaded {len(train_df)} training samples from {expanded_csv}")

    # 3. Create Datasets and DataLoaders
    train_transforms = get_hardened_train_transforms(image_size=image_size)
    val_transforms = get_val_transforms(image_size=image_size)

    train_dataset = BiofoulingDataset(
        csv_file_or_df=expanded_csv,
        root_dir=PROJECT_ROOT,
        transform=train_transforms,
    )
    val_dataset = BiofoulingDataset(
        csv_file_or_df=val_csv_path,
        root_dir=PROJECT_ROOT,
        transform=val_transforms,
    )

    num_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Load Baseline Model & Freeze Backbone
    model_name = "efficientnet_b3"
    drop_rate = 0.4

    if checkpoint_path.exists():
        logger.info(f"Loading baseline checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = ckpt.get("model_name", "efficientnet_b3")
        drop_rate = ckpt.get("drop_rate", 0.4)
        model = create_biofouling_model(
            model_name=model_name,
            num_classes=4,
            pretrained=False,
            drop_rate=drop_rate,
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning(f"Baseline checkpoint {checkpoint_path} not found. Initializing pretrained {model_name}...")
        model = create_biofouling_model(
            model_name=model_name,
            num_classes=4,
            pretrained=True,
            drop_rate=drop_rate,
        )

    model = model.to(device)

    # Freeze backbone feature extractor
    model.freeze_backbone()
    params_info = model.get_parameter_summary()
    logger.info(f"Parameter Allocation: Total={params_info['total_params']:,} | Trainable={params_info['trainable_params']:,} | Frozen={params_info['frozen_params']:,}")

    # 5. Loss & Optimizer Setup
    class_weights = calculate_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    # Baseline Evaluation
    init_loss, init_acc, init_f1, _ = evaluate_model(model, val_loader, criterion, device)
    logger.info(f"Initial Baseline on Validation Split: Loss={init_loss:.4f} | Acc={init_acc*100:.2f}% | Macro-F1={init_f1:.4f}")

    # 6. Fine-Tuning Loop
    best_val_f1 = init_f1
    best_state_dict = model.state_dict()
    epoch_histories: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc*100:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        epoch_histories.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
        })

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 7. Save Best Checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a timestamped backup of the prior checkpoint if it exists
    if checkpoint_path.exists():
        backup_path = checkpoint_path.parent / f"{checkpoint_path.stem}_backup_{int(time.time())}.pth"
        try:
            shutil.copy2(checkpoint_path, backup_path)
            logger.info(f"Archived previous checkpoint to: {backup_path}")
        except Exception as e:
            logger.warning(f"Could not backup previous checkpoint: {e}")

    save_payload = {
        "model_name": model_name,
        "num_classes": 4,
        "drop_rate": drop_rate,
        "model_state_dict": best_state_dict,
        "val_macro_f1": best_val_f1,
        "fine_tuned_at": datetime.now(timezone.utc).isoformat(),
        "epochs_trained": epochs,
        "active_learning_samples_count": ingestion_stats.get("total_active_learning_samples", 0),
    }
    torch.save(save_payload, checkpoint_path)
    logger.info(f"Saved updated model checkpoint to: {checkpoint_path} (Val Macro-F1: {best_val_f1:.4f})")

    # 8. Export to ONNX
    logger.info("Exporting fine-tuned model to production ONNX format...")
    exported_onnx_path, onnx_metrics = export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_onnx_path=output_onnx_path,
        image_size=image_size,
    )

    elapsed_sec = round(time.time() - t_start, 2)

    result = {
        "status": "COMPLETED",
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(exported_onnx_path),
        "initial_val_f1": round(init_f1, 4),
        "final_val_f1": round(best_val_f1, 4),
        "f1_delta": round(best_val_f1 - init_f1, 4),
        "epochs_trained": epochs,
        "training_samples_count": len(train_df),
        "active_learning_samples_count": ingestion_stats.get("total_active_learning_samples", 0),
        "ingestion_stats": ingestion_stats,
        "onnx_metrics": onnx_metrics,
        "epoch_histories": epoch_histories,
        "elapsed_seconds": elapsed_sec,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Active Learning Incremental Fine-Tuning Pipeline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_v4.pth", help="Model checkpoint path")
    parser.add_argument("--onnx", type=str, default="biofouling_model.onnx", help="Exported ONNX destination")
    parser.add_argument("--val-csv", type=str, default="data/splits_cleaned/val.csv", help="Validation split CSV")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")

    args = parser.parse_args()

    results = fine_tune_model(
        checkpoint_path=Path(args.checkpoint),
        output_onnx_path=Path(args.onnx),
        val_csv_path=Path(args.val_csv),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
    )

    print("\n" + "=" * 65)
    print(" Active Learning Fine-Tuning Complete")
    print("=" * 65)
    print(f" Status               : {results['status']}")
    print(f" Checkpoint           : {results['checkpoint_path']}")
    print(f" ONNX Model           : {results['onnx_path']}")
    print(f" Initial Val Macro-F1 : {results['initial_val_f1']:.4f}")
    print(f" Final Val Macro-F1   : {results['final_val_f1']:.4f} (Delta: {results['f1_delta']:+.4f})")
    print(f" Total Samples        : {results['training_samples_count']} (HITL: {results['active_learning_samples_count']})")
    print(f" Elapsed Time         : {results['elapsed_seconds']}s")
    print("=" * 65)


if __name__ == "__main__":
    main()

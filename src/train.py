#!/usr/bin/env python3
"""
Two-Stage Transfer Learning Training Pipeline for 4-Class Industrial Biofouling
-----------------------------------------------------------------------------
Implements two-stage training (Stage 1: Head Warmup -> Stage 2: Fine-Tuning)
with automatic mixed precision, inverse-frequency class weighting, CosineAnnealingLR
scheduling, Macro-F1 validation tracking, and checkpoint persistence.

Author: Senior Computer Vision & PyTorch Engineer
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IDX_TO_LABEL, LABEL_TO_IDX, get_dataloaders
from src.model import BiofoulingClassifier, create_biofouling_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingTrainer")


def calculate_class_weights(train_csv_path: Path, num_classes: int = 4) -> torch.Tensor:
    """
    Compute balanced class weights from training label distribution.
    Formula: N_samples / (N_classes * count(class_i))
    """
    df = pd.read_csv(train_csv_path)
    labels = df["label_idx"].values
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Computed Class Weights: {dict(zip([IDX_TO_LABEL[i] for i in range(num_classes)], weights.round(4)))}")
    return class_weights


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """
    Execute one training epoch with optional mixed precision.
    Returns: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset split.
    Returns: (val_loss, val_accuracy, val_macro_f1, all_targets, all_preds)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = np.mean(np.array(all_preds) == np.array(all_targets)) if total > 0 else 0.0
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_macro_f1, np.array(all_targets), np.array(all_preds)


def run_two_stage_training(
    splits_dir: str = "data/splits",
    checkpoints_dir: str = "checkpoints",
    model_name: str = "efficientnet_b0",
    num_classes: int = 4,
    stage1_epochs: int = 5,
    stage2_epochs: int = 20,
    batch_size: int = 16,
    num_workers: int = 2,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    weight_decay: float = 1e-2,
    drop_rate: float = 0.3,
    seed: int = 42,
) -> Tuple[BiofoulingClassifier, Dict[str, any]]:
    """
    Execute full two-stage training pipeline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits_path = Path(splits_dir)
    checkpoints_path = Path(checkpoints_dir)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_path / "best_model.pth"

    # Device & Mixed Precision configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("=" * 80)
    logger.info(f"Hardware Acceleration: {device.type.upper()} | Mixed Precision (AMP): {use_amp}")
    if device.type == "cuda":
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 80)

    # 1. Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_dir=splits_path,
        batch_size=batch_size,
        num_workers=num_workers if sys.platform != "win32" else 0,  # Safe on Windows
        pin_memory=(device.type == "cuda"),
    )

    # 2. Compute Class Weights & Criterion
    class_weights = calculate_class_weights(splits_path / "train.csv", num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Instantiate Model
    model = create_biofouling_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_rate=drop_rate,
    ).to(device)

    best_val_f1 = 0.0
    history: List[Dict[str, any]] = []

    # ==========================================================================
    # STAGE 1: Classification Head Warm-up (Frozen Backbone)
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STAGE 1: WARM-UP CLASSIFIER HEAD ({stage1_epochs} Epochs, lr={stage1_lr})")
    logger.info("=" * 80)

    model.freeze_backbone()
    trainable_params_s1 = [p for p in model.parameters() if p.requires_grad]
    optimizer_s1 = AdamW(trainable_params_s1, lr=stage1_lr, weight_decay=weight_decay)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, device, scaler, use_amp
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "stage": "Stage 1 (Warmup)",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s1.state_dict(),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
                "class_to_idx": LABEL_TO_IDX,
                "model_name": model_name,
            }, best_model_path)

        record = {
            "Epoch": f"S1-{epoch:02d}",
            "Stage": "Warmup",
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc * 100:.2f}%",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val Macro-F1": f"{val_f1:.4f}",
            "LR": f"{stage1_lr:.1e}",
            "Best": "*" if is_best else "",
            "Time": f"{elapsed:.1f}s",
        }
        history.append(record)
        logger.info(
            f"[Stage 1] Epoch {epoch:02d}/{stage1_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}% | Val F1: {val_f1:.4f} "
            f"{'(* Best Saved)' if is_best else ''} ({elapsed:.1f}s)"
        )

    # ==========================================================================
    # STAGE 2: End-to-End Fine-Tuning (Unfrozen Backbone + Cosine Annealing)
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STAGE 2: END-TO-END FINE-TUNING ({stage2_epochs} Epochs, lr={stage2_lr})")
    logger.info("=" * 80)

    model.unfreeze_backbone()
    optimizer_s2 = AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        current_lr = optimizer_s2.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, device, scaler, use_amp
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler_s2.step()
        elapsed = time.time() - t0

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": stage1_epochs + epoch,
                "stage": "Stage 2 (Fine-tuning)",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s2.state_dict(),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
                "class_to_idx": LABEL_TO_IDX,
                "model_name": model_name,
            }, best_model_path)

        record = {
            "Epoch": f"S2-{epoch:02d}",
            "Stage": "Fine-tune",
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc * 100:.2f}%",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val Macro-F1": f"{val_f1:.4f}",
            "LR": f"{current_lr:.1e}",
            "Best": "*" if is_best else "",
            "Time": f"{elapsed:.1f}s",
        }
        history.append(record)
        logger.info(
            f"[Stage 2] Epoch {epoch:02d}/{stage2_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}% | Val F1: {val_f1:.4f} "
            f"{'(* Best Saved)' if is_best else ''} ({elapsed:.1f}s)"
        )

    # ==========================================================================
    # Print Training Progress Table
    # ==========================================================================
    print("\n" + "=" * 95)
    print("                     TWO-STAGE MODEL TRAINING METRICS HISTORY                     ")
    print("=" * 95)
    headers = ["Epoch", "Stage", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val Macro-F1", "LR", "Best", "Time"]
    rows = [[r[h] for h in headers] for r in history]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        row_fmt = "{:<8} | {:<10} | {:<10} | {:<10} | {:<9} | {:<9} | {:<12} | {:<8} | {:<4} | {:<6}"
        print(row_fmt.format(*headers))
        print("-" * 95)
        for r in rows:
            print(row_fmt.format(*[str(x) for x in r]))
    print("=" * 95)

    # ==========================================================================
    # Final Test Set Evaluation using Best Saved Checkpoint
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"LOADING BEST MODEL CHECKPOINT -> {best_model_path}")
    logger.info("=" * 80)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from Epoch {checkpoint['epoch']} ({checkpoint['stage']}) with Val F1: {checkpoint['val_macro_f1']:.4f}")

    test_loss, test_acc, test_macro_f1, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device
    )

    target_names = [IDX_TO_LABEL[i] for i in range(num_classes)]
    clf_report = classification_report(
        test_targets,
        test_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    conf_mat = confusion_matrix(test_targets, test_preds)

    print("\n" + "=" * 80)
    print("                     FINAL INDEPENDENT TEST SET EVALUATION                     ")
    print("=" * 80)
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"Test Macro-F1:    {test_macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(clf_report)
    print("Confusion Matrix:")
    print("Rows: Actual | Columns: Predicted [Low, Moderate, High, Critical]")
    print(conf_mat)
    print("=" * 80 + "\n")

    eval_results = {
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "history": history,
    }

    return model, eval_results


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Biofouling Model Training Pipeline")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory containing split CSVs")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0", help="timm backbone architecture")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Warmup epochs for classifier head (default: 5)")
    parser.add_argument("--stage2-epochs", type=int, default=20, help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 learning rate (default: 1e-3)")
    parser.add_argument("--stage2-lr", type=float, default=1e-4, help="Stage 2 learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW (default: 1e-2)")
    parser.add_argument("--drop-rate", type=float, default=0.3, help="Dropout probability (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_two_stage_training(
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        num_classes=4,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        drop_rate=args.drop_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

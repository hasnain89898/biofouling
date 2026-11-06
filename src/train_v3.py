#!/usr/bin/env python3
"""
Advanced Industrial Biofouling Model Training Pipeline (v3)
----------------------------------------------------------
Features:
1. Scaled Model Capacity: timm 'efficientnet_b2' backbone (in_features=1408).
2. Cleaned & Balanced Dataset: Trains on data/splits_cleaned/ (400 balanced train images).
3. Hardened Regularization: Head Dropout (p=0.5), Label Smoothing (0.1), AdamW (wd=0.05).
4. Long Training & EarlyStopping: Patience of 10 consecutive epochs.
5. Test-Time Augmentation (TTA): 4-view spatial ensembling on independent test evaluation.

Author: Principal AI Research Engineer
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

from src.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    get_dataloaders,
)
from src.model import BiofoulingClassifier, create_biofouling_model
from src.tta import TestTimeAugmentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingTrainerV3")


def calculate_class_weights(train_csv_path: Path, num_classes: int = 4) -> torch.Tensor:
    """Compute balanced class weights from training label distribution."""
    df = pd.read_csv(train_csv_path)
    labels = df["label_idx"].values
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class Weights: {dict(zip([IDX_TO_LABEL[i] for i in range(num_classes)], weights.round(4)))}")
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
    """Execute one training epoch."""
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
def evaluate_standard(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard single-pass deterministic evaluation.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = np.mean(np.array(all_preds) == np.array(all_targets)) if total > 0 else 0.0
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return (
        val_loss,
        val_acc,
        val_macro_f1,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


def run_v3_pipeline(
    splits_dir: str = "data/splits_cleaned",
    checkpoints_dir: str = "checkpoints",
    model_name: str = "efficientnet_b2",
    num_classes: int = 4,
    stage1_epochs: int = 5,
    stage2_epochs: int = 35,
    early_stopping_patience: int = 10,
    label_smoothing: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 2,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    weight_decay: float = 0.05,
    drop_rate: float = 0.5,
    seed: int = 42,
) -> Tuple[BiofoulingClassifier, Dict[str, any]]:
    """
    Train upgraded biofouling model on cleaned dataset and evaluate with TTA.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits_path = Path(splits_dir)
    checkpoints_path = Path(checkpoints_dir)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_path / "best_model_v3.pth"
    compat_model_path = checkpoints_path / "best_model.pth"

    # Hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("=" * 85)
    logger.info("            UPGRADED INDUSTRIAL BIOFOULING PIPELINE (v3) - EFFICIENTNET-B2            ")
    logger.info("=" * 85)
    logger.info(f"Hardware Acceleration: {device.type.upper()} | Mixed Precision (AMP): {use_amp}")
    logger.info(f"Model Backbone: {model_name} | Head Dropout: {drop_rate}")
    logger.info(f"Dataset: {splits_path} (Scrubbed & Balanced)")
    logger.info(f"Regularization: Weight Decay={weight_decay} | Label Smoothing={label_smoothing}")
    logger.info(f"EarlyStopping Patience: {early_stopping_patience} epochs (monitoring Val Loss)")
    logger.info("=" * 85)

    # 1. Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_dir=splits_path,
        batch_size=batch_size,
        num_workers=num_workers if sys.platform != "win32" else 0,
        pin_memory=(device.type == "cuda"),
        hardened=True,
    )

    # 2. Compute Class Weights & Criterion
    class_weights = calculate_class_weights(splits_path / "train.csv", num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    # 3. Instantiate Upgraded Model (efficientnet_b2)
    model = create_biofouling_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_rate=drop_rate,
    ).to(device)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    early_stop_counter = 0
    history: List[Dict[str, any]] = []

    # ==========================================================================
    # STAGE 1: Head Warm-up (Frozen Backbone, 5 Epochs)
    # ==========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(f"STAGE 1: WARM-UP CLASSIFIER HEAD ({stage1_epochs} Epochs, lr={stage1_lr}, wd={weight_decay})")
    logger.info("=" * 85)

    model.freeze_backbone()
    trainable_params_s1 = [p for p in model.parameters() if p.requires_grad]
    optimizer_s1 = AdamW(trainable_params_s1, lr=stage1_lr, weight_decay=weight_decay)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, device, scaler, use_amp
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate_standard(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            ckpt = {
                "epoch": epoch,
                "stage": "Stage 1 (Warmup)",
                "model_state_dict": model.state_dict(),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
                "class_to_idx": LABEL_TO_IDX,
                "model_name": model_name,
                "drop_rate": drop_rate,
                "label_smoothing": label_smoothing,
            }
            torch.save(ckpt, best_model_path)
            torch.save(ckpt, compat_model_path)

        record = {
            "Epoch": f"S1-{epoch:02d}",
            "Stage": "Warmup",
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc * 100:.2f}%",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val Macro-F1": f"{val_f1:.4f}",
            "LR": f"{stage1_lr:.1e}",
            "ES Counter": f"0/{early_stopping_patience}",
            "Best": "*" if is_best else "",
            "Time": f"{elapsed:.1f}s",
        }
        history.append(record)
        logger.info(
            f"[Stage 1] Epoch {epoch:02d}/{stage1_epochs:02d} | Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%) | Val F1: {val_f1:.4f} "
            f"{'(* Best Saved)' if is_best else ''} ({elapsed:.1f}s)"
        )

    # ==========================================================================
    # STAGE 2: End-to-End Fine-Tuning with EarlyStopping (35 Epochs)
    # ==========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(f"STAGE 2: END-TO-END FINE-TUNING ({stage2_epochs} Epochs, lr={stage2_lr}, wd={weight_decay})")
    logger.info("=" * 85)

    model.unfreeze_backbone()
    optimizer_s2 = AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        current_lr = optimizer_s2.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, device, scaler, use_amp
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate_standard(model, val_loader, criterion, device)
        scheduler_s2.step()
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            early_stop_counter = 0
            ckpt = {
                "epoch": stage1_epochs + epoch,
                "stage": "Stage 2 (Fine-tuning)",
                "model_state_dict": model.state_dict(),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1),
                "val_acc": float(val_acc),
                "class_to_idx": LABEL_TO_IDX,
                "model_name": model_name,
                "drop_rate": drop_rate,
                "label_smoothing": label_smoothing,
            }
            torch.save(ckpt, best_model_path)
            torch.save(ckpt, compat_model_path)
        else:
            early_stop_counter += 1

        record = {
            "Epoch": f"S2-{epoch:02d}",
            "Stage": "Fine-tune",
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc * 100:.2f}%",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val Macro-F1": f"{val_f1:.4f}",
            "LR": f"{current_lr:.1e}",
            "ES Counter": f"{early_stop_counter}/{early_stopping_patience}",
            "Best": "*" if is_best else "",
            "Time": f"{elapsed:.1f}s",
        }
        history.append(record)
        logger.info(
            f"[Stage 2] Epoch {epoch:02d}/{stage2_epochs:02d} | Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%) | Val F1: {val_f1:.4f} | "
            f"ES: {early_stop_counter}/{early_stopping_patience} "
            f"{'(* Best Saved)' if is_best else ''} ({elapsed:.1f}s)"
        )

        if early_stop_counter >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered at Stage 2 Epoch {epoch:02d} (no Val Loss improvement for {early_stopping_patience} epochs)."
            )
            break

    # Print Training History
    print("\n" + "=" * 105)
    print("                    ADVANCED MODEL TRAINING METRICS HISTORY (v3)                    ")
    print("=" * 105)
    headers = ["Epoch", "Stage", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val Macro-F1", "LR", "ES Counter", "Best", "Time"]
    rows = [[r[h] for h in headers] for r in history]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        row_fmt = "{:<8} | {:<10} | {:<10} | {:<10} | {:<9} | {:<9} | {:<12} | {:<8} | {:<10} | {:<4} | {:<6}"
        print(row_fmt.format(*headers))
        print("-" * 105)
        for r in rows:
            print(row_fmt.format(*[str(x) for x in r]))
    print("=" * 105)

    # ==========================================================================
    # Evaluation: Standard vs Test-Time Augmentation (TTA)
    # ==========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(f"LOADING BEST MODEL CHECKPOINT -> {best_model_path}")
    logger.info("=" * 85)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoint from Epoch {checkpoint['epoch']} ({checkpoint['stage']}) | "
        f"Best Val Loss: {checkpoint['val_loss']:.4f} | Best Val F1: {checkpoint['val_macro_f1']:.4f}"
    )

    # 1. Standard Single-Pass Evaluation
    std_loss, std_acc, std_f1, test_targets, std_preds, std_probs = evaluate_standard(
        model, test_loader, criterion, device
    )

    # 2. Test-Time Augmentation (4-View Ensembling)
    tta_engine = TestTimeAugmentor(model=model, device=device)
    tta_loss, tta_acc, tta_f1, tta_targets, tta_preds, tta_probs = tta_engine.evaluate_loader(
        test_loader, criterion=criterion
    )

    target_names = [IDX_TO_LABEL[i] for i in range(num_classes)]
    tta_report = classification_report(
        tta_targets,
        tta_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    tta_conf_mat = confusion_matrix(tta_targets, tta_preds)

    baseline_v1_f1 = 0.5233
    baseline_v2_f1 = 0.5541

    print("\n" + "=" * 90)
    print("                     INDEPENDENT TEST SET BENCHMARK REPORT (v3)                     ")
    print("=" * 90)
    print(f"Standard Single-Pass Test Accuracy:  {std_acc * 100:.2f}% | Test Macro-F1: {std_f1:.4f}")
    print(f"Test-Time Augmentation (TTA) Acc:    {tta_acc * 100:.2f}% | TTA Macro-F1:  {tta_f1:.4f}")
    print(f"Improvement over v1 Baseline (0.5233): +{tta_f1 - baseline_v1_f1:.4f}")
    print(f"Improvement over v2 Hardened (0.5541): +{tta_f1 - baseline_v2_f1:.4f}")
    print("\nEnhanced Classification Report (with TTA):\n")
    print(tta_report)
    print("Enhanced Confusion Matrix (with TTA):")
    print("Rows: Ground Truth | Columns: Predicted [Low, Moderate, High, Critical]")
    print(tta_conf_mat)
    print("=" * 90 + "\n")

    return model, {
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "std_acc": std_acc,
        "std_f1": std_f1,
        "tta_acc": tta_acc,
        "tta_f1": tta_f1,
        "tta_report": tta_report,
        "tta_conf_mat": tta_conf_mat,
    }


def main():
    parser = argparse.ArgumentParser(description="Upgraded Biofouling Model Training (v3)")
    parser.add_argument("--splits-dir", type=str, default="data/splits_cleaned", help="Cleaned splits directory")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--model-name", type=str, default="efficientnet_b2", help="timm backbone")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--stage2-epochs", type=int, default=35, help="Fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--stage2-lr", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--drop-rate", type=float, default=0.5, help="Dropout in classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_v3_pipeline(
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        num_classes=4,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        drop_rate=args.drop_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

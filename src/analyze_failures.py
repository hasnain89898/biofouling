#!/usr/bin/env python3
"""
Diagnostic Failure Analysis Module for 4-Class Industrial Biofouling
-------------------------------------------------------------------
Loads the trained model checkpoint, performs deterministic inference on the
independent test split (data/splits/test.csv), identifies all misclassified images,
and ranks the Top 10 most confident WRONG predictions.

Author: Senior AI Research Engineer
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

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

from src.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    BiofoulingDataset,
    get_val_transforms,
)
from src.model import BiofoulingClassifier, create_biofouling_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FailureAnalysis")


def run_failure_analysis(
    checkpoint_path: Path,
    test_csv_path: Path,
    metadata_csv_path: Path,
    output_csv_path: Path,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Perform deep failure analysis on the test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running failure analysis on device: {device.type.upper()}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test split CSV not found at: {test_csv_path}")

    # Load test split metadata
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df)} test samples from {test_csv_path}")

    # Load provenance metadata if available
    metadata_lookup: Dict[str, str] = {}
    if metadata_csv_path.exists():
        meta_df = pd.read_csv(metadata_csv_path)
        for _, row in meta_df.iterrows():
            metadata_lookup[row["filename"]] = row.get("source_query", "N/A")

    # Load Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    drop_rate = checkpoint.get("drop_rate", 0.5)

    model = create_biofouling_model(
        model_name=model_name,
        num_classes=4,
        pretrained=False,
        drop_rate=drop_rate,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Successfully loaded checkpoint: {checkpoint_path} (Model: {model_name})")

    # Prepare Dataset with deterministic validation transforms
    val_transforms = get_val_transforms(image_size=224)
    test_dataset = BiofoulingDataset(test_df, transform=val_transforms)

    results: List[Dict[str, any]] = []

    # Inference loop
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image_tensor, label_tensor = test_dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device)

            logits = model(image_batch)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            true_idx = int(label_tensor.item())
            conf = float(probs[pred_idx])

            filename = test_df.iloc[idx]["filename"]
            is_correct = (pred_idx == true_idx)

            results.append({
                "test_idx": idx,
                "filename": filename,
                "true_idx": true_idx,
                "true_label": IDX_TO_LABEL[true_idx],
                "pred_idx": pred_idx,
                "pred_label": IDX_TO_LABEL[pred_idx],
                "confidence": conf,
                "is_correct": is_correct,
                "prob_Low": float(probs[0]),
                "prob_Moderate": float(probs[1]),
                "prob_High": float(probs[2]),
                "prob_Critical": float(probs[3]),
                "source_query": metadata_lookup.get(filename, "N/A"),
            })

    all_results_df = pd.DataFrame(results)

    # Filter to misclassified images
    failures_df = all_results_df[~all_results_df["is_correct"]].copy()
    failures_df.sort_values(by="confidence", ascending=False, inplace=True)
    failures_df.reset_index(drop=True, inplace=True)

    # Save full failures catalog
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved full failure catalog ({len(failures_df)} errors) -> {output_csv_path}")

    # Top K most confident errors
    top_k_failures = failures_df.head(top_k).copy()

    # Format summary table
    table_data = []
    for rank, (_, row) in enumerate(top_k_failures.iterrows(), 1):
        probs_str = (
            f"L:{row['prob_Low']:.2f} | M:{row['prob_Moderate']:.2f} | "
            f"H:{row['prob_High']:.2f} | C:{row['prob_Critical']:.2f}"
        )
        table_data.append([
            f"#{rank:02d}",
            Path(row["filename"]).name,
            row["true_label"],
            row["pred_label"],
            f"{row['confidence'] * 100:.1f}%",
            probs_str,
            row["source_query"],
        ])

    headers = [
        "Rank",
        "Image Filename",
        "True Label",
        "Model Prediction",
        "Confidence",
        "Class Probabilities [L, M, H, C]",
        "Source Search Query",
    ]

    total_test = len(all_results_df)
    total_errors = len(failures_df)
    accuracy = (total_test - total_errors) / total_test * 100

    print("\n" + "=" * 115)
    print("                     TOP 10 MOST CONFIDENT MISCLASSIFICATIONS (FAILURE ANALYSIS)                     ")
    print("=" * 115)
    print(f"Total Test Samples: {total_test} | Total Errors: {total_errors} | Test Accuracy: {accuracy:.2f}%\n")

    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        row_fmt = "{:<5} | {:<22} | {:<10} | {:<16} | {:<10} | {:<32} | {:<25}"
        print(row_fmt.format(*headers))
        print("-" * 115)
        for r in table_data:
            print(row_fmt.format(*[str(x) for x in r]))

    print("=" * 115)
    print(f"Full Failure Catalog: {output_csv_path.resolve()}\n")

    return failures_df


def main():
    parser = argparse.ArgumentParser(description="Failure Analysis on Test Split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_v2.pth", help="Model checkpoint path")
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv", help="Test split CSV path")
    parser.add_argument("--metadata-csv", type=str, default="metadata.csv", help="Metadata CSV path")
    parser.add_argument("--output-csv", type=str, default="data/failure_analysis.csv", help="Output failure report CSV")
    parser.add_argument("--top-k", type=int, default=10, help="Top K confident failures to display")

    args = parser.parse_args()

    checkpoint_p = Path(args.checkpoint)
    if not checkpoint_p.exists():
        # Fallback to best_model.pth if best_model_v2.pth not yet created
        checkpoint_p = Path("checkpoints/best_model.pth")

    run_failure_analysis(
        checkpoint_path=checkpoint_p,
        test_csv_path=Path(args.test_csv),
        metadata_csv_path=Path(args.metadata_csv),
        output_csv_path=Path(args.output_csv),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

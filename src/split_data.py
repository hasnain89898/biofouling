#!/usr/bin/env python3
"""
Dataset Splitting Pipeline for 4-Class Industrial Biofouling
-----------------------------------------------------------
Reads metadata.csv, applies integer label encoding ('Low': 0, 'Moderate': 1,
'High': 2, 'Critical': 3), and performs a rigorous 70% Train / 15% Validation /
15% Test stratified split using scikit-learn StratifiedShuffleSplit.

Saves output CSVs to data/splits/train.csv, val.csv, test.csv.

Author: Senior Computer Vision & PyTorch Engineer
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure safe console printing on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataSplitter")

# Class label to index mapping
LABEL_TO_IDX: Dict[str, int] = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3,
}

IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}


def perform_stratified_split(
    metadata_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a 70/15/15 stratified split on dataset metadata.

    Args:
        metadata_path: Path to metadata.csv.
        output_dir: Directory to save train.csv, val.csv, test.csv.
        train_ratio: Proportion for train split (default: 0.70).
        val_ratio: Proportion for validation split (default: 0.15).
        test_ratio: Proportion for test split (default: 0.15).
        random_state: Seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} records from {metadata_path}")

    # Map string class labels to integer indices
    if "class_label" not in df.columns:
        raise ValueError("Missing 'class_label' column in metadata.csv")

    df["label_idx"] = df["class_label"].map(LABEL_TO_IDX)
    if df["label_idx"].isnull().any():
        unknown_labels = df[df["label_idx"].isnull()]["class_label"].unique()
        raise ValueError(f"Unknown class labels encountered: {unknown_labels}")

    # Step 1: Split full dataset into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_indices, temp_indices = next(sss1.split(df, df["label_idx"]))

    train_df = df.iloc[train_indices].copy()
    temp_df = df.iloc[temp_indices].copy()

    # Step 2: Split Temp (30%) into Val (15%) and Test (15%) -> 50% / 50% of Temp
    val_rel_ratio = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=random_state)
    val_indices_rel, test_indices_rel = next(sss2.split(temp_df, temp_df["label_idx"]))

    val_df = temp_df.iloc[val_indices_rel].copy()
    test_df = temp_df.iloc[test_indices_rel].copy()

    # Assign split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Select standard output columns: filename, class_label, label_idx, split
    out_cols = ["filename", "class_label", "label_idx", "split"]
    train_out = train_df[out_cols].reset_index(drop=True)
    val_out = val_df[out_cols].reset_index(drop=True)
    test_out = test_df[out_cols].reset_index(drop=True)

    # Save CSVs into output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    logger.info(f"Saved train split ({len(train_out)} rows) -> {train_path}")
    logger.info(f"Saved val split   ({len(val_out)} rows) -> {val_path}")
    logger.info(f"Saved test split  ({len(test_out)} rows) -> {test_path}")

    return train_out, val_out, test_out


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print an organized distribution table showing sample counts per class across splits."""
    classes = list(LABEL_TO_IDX.keys())
    table_data = []

    total_train = 0
    total_val = 0
    total_test = 0
    total_all = 0

    for cls in classes:
        idx = LABEL_TO_IDX[cls]
        n_train = (train_df["class_label"] == cls).sum()
        n_val = (val_df["class_label"] == cls).sum()
        n_test = (test_df["class_label"] == cls).sum()
        n_total = n_train + n_val + n_test

        train_pct = (n_train / n_total * 100) if n_total > 0 else 0
        val_pct = (n_val / n_total * 100) if n_total > 0 else 0
        test_pct = (n_test / n_total * 100) if n_total > 0 else 0

        table_data.append([
            cls,
            idx,
            f"{n_train} ({train_pct:.1f}%)",
            f"{n_val} ({val_pct:.1f}%)",
            f"{n_test} ({test_pct:.1f}%)",
            n_total,
        ])

        total_train += n_train
        total_val += n_val
        total_test += n_test
        total_all += n_total

    tot_train_pct = (total_train / total_all * 100) if total_all > 0 else 0
    tot_val_pct = (total_val / total_all * 100) if total_all > 0 else 0
    tot_test_pct = (total_test / total_all * 100) if total_all > 0 else 0

    table_data.append([
        "TOTAL",
        "-",
        f"{total_train} ({tot_train_pct:.1f}%)",
        f"{total_val} ({tot_val_pct:.1f}%)",
        f"{total_test} ({tot_test_pct:.1f}%)",
        total_all,
    ])

    headers = ["Class Name", "Label Index", "Train (70%)", "Val (15%)", "Test (15%)", "Total Samples"]

    print("\n" + "=" * 80)
    print("                BIOFOULING DATASET STRATIFIED SPLIT REPORT                ")
    print("=" * 80)
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        row_fmt = "{:<12} | {:<11} | {:<14} | {:<13} | {:<13} | {:<12}"
        print(row_fmt.format(*headers))
        print("-" * 80)
        for row in table_data:
            print(row_fmt.format(*[str(x) for x in row]))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stratified Train/Val/Test Dataset Splitter")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Directory to store split CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    train_df, val_df, test_df = perform_stratified_split(
        metadata_path=Path(args.metadata_file),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    print_split_summary(train_df, val_df, test_df)


if __name__ == "__main__":
    main()

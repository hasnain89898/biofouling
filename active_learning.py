#!/usr/bin/env python3
"""
Active Learning Dataset Integration & Expansion Pipeline
---------------------------------------------------------
1. Queries the PostgreSQL Review Queue for verified human-resolved inspection frames.
2. Ingests expert-reviewed images into the active training directory structure (data/expanded/{class_label}/).
3. Generates/updates an expanded training manifest (data/splits_cleaned/train_expanded.csv)
   merging base training samples with human-in-the-loop expert corrections.

Author: Principal ML Operations & Deep Learning Engineer
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

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

from database import ReviewQueueRecord, SessionLocal, init_db
from src.dataset import IDX_TO_LABEL, LABEL_TO_IDX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ActiveLearningPipeline")

# Default Paths
DEFAULT_EXPANDED_DIR = PROJECT_ROOT / "data" / "expanded"
DEFAULT_INDUSTRIAL_TRAIN_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "train_industrial.csv"
DEFAULT_LEGACY_TRAIN_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "train.csv"
DEFAULT_BASE_TRAIN_CSV = DEFAULT_INDUSTRIAL_TRAIN_CSV if DEFAULT_INDUSTRIAL_TRAIN_CSV.exists() else DEFAULT_LEGACY_TRAIN_CSV
DEFAULT_EXPANDED_TRAIN_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "train_expanded.csv"
REVIEW_QUEUE_IMG_DIR = PROJECT_ROOT / "data" / "review_queue_images"


def get_resolved_review_records(db: Optional[Session] = None) -> List[ReviewQueueRecord]:
    """
    Query the database for all resolved inspection frames with verified ground-truth labels.
    """
    should_close = False
    if db is None:
        init_db()
        db = SessionLocal()
        should_close = True

    try:
        records = (
            db.query(ReviewQueueRecord)
            .filter(
                ReviewQueueRecord.review_status == "RESOLVED",
                ReviewQueueRecord.resolved_label.in_(list(LABEL_TO_IDX.keys())),
            )
            .order_by(ReviewQueueRecord.resolved_at.asc())
            .all()
        )
        return records
    except Exception as e:
        logger.error(f"Error querying resolved review records from database: {e}")
        return []
    finally:
        if should_close:
            db.close()


def ingest_resolved_samples(
    db: Optional[Session] = None,
    target_dir: Path = DEFAULT_EXPANDED_DIR,
    base_train_csv: Path = DEFAULT_BASE_TRAIN_CSV,
    output_train_csv: Path = DEFAULT_EXPANDED_TRAIN_CSV,
) -> Dict[str, Any]:
    """
    Ingest verified human-reviewed frames into the expanded dataset structure
    and generate an expanded training manifest.

    Returns:
        Dict containing ingestion statistics, per-class counts, and manifest path.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for class_name in LABEL_TO_IDX.keys():
        (target_dir / class_name).mkdir(parents=True, exist_ok=True)

    records = get_resolved_review_records(db=db)
    logger.info(f"Found {len(records)} resolved review queue records in database.")

    newly_ingested = 0
    already_ingested = 0
    missing_source_count = 0
    ingested_samples: List[Dict[str, Any]] = []

    for rec in records:
        resolved_label = rec.resolved_label
        if resolved_label not in LABEL_TO_IDX:
            continue

        label_idx = LABEL_TO_IDX[resolved_label]

        # Locate source image file
        source_path = None
        if rec.saved_image_path and rec.saved_image_path != "N/A":
            candidate = Path(rec.saved_image_path)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / rec.saved_image_path
            if candidate.exists():
                source_path = candidate

        if source_path is None:
            # Fallback search by filename in review_queue_images
            for candidate in REVIEW_QUEUE_IMG_DIR.glob(f"*{Path(rec.filename).name}"):
                if candidate.exists():
                    source_path = candidate
                    break

        if source_path is None or not source_path.exists():
            logger.warning(f"Source image missing for queue item {rec.queue_id} ({rec.filename})")
            missing_source_count += 1
            continue

        # Destination filename: standardized and unique
        dest_filename = f"hitl_{rec.queue_id}_{Path(rec.filename).name}"
        dest_path = target_dir / resolved_label / dest_filename
        try:
            rel_dest_path = str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_dest_path = str(dest_path).replace("\\", "/")

        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            newly_ingested += 1
            logger.info(f"Ingested new sample: {source_path.name} -> {rel_dest_path} (Class: {resolved_label})")
        else:
            already_ingested += 1

        ingested_samples.append({
            "filename": rel_dest_path,
            "class_label": resolved_label,
            "label_idx": label_idx,
            "split": "train",
            "is_synthetic": False,
            "is_hitl_active_learning": True,
            "queue_id": rec.queue_id,
            "resolved_by": rec.resolved_by or "Expert Inspector",
        })

    # Build Expanded Training Manifest
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    if base_train_csv.exists():
        base_df = pd.read_csv(base_train_csv)
        if "is_hitl_active_learning" not in base_df.columns:
            base_df["is_hitl_active_learning"] = False
        if "queue_id" not in base_df.columns:
            base_df["queue_id"] = None
        if "resolved_by" not in base_df.columns:
            base_df["resolved_by"] = None
    else:
        logger.warning(f"Base training CSV not found at {base_train_csv}. Initializing empty baseline.")
        base_df = pd.DataFrame(columns=[
            "filename", "class_label", "label_idx", "split", "is_synthetic",
            "is_hitl_active_learning", "queue_id", "resolved_by"
        ])

    if ingested_samples:
        new_df = pd.DataFrame(ingested_samples)
        # Deduplicate to prevent duplicate entries for the same filename
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["filename"], keep="last")
    else:
        combined_df = base_df.copy()

    combined_df.to_csv(output_train_csv, index=False)
    logger.info(f"Saved expanded training manifest to: {output_train_csv} ({len(combined_df)} total samples)")

    # Per-class distribution summary
    class_distribution = combined_df["class_label"].value_counts().to_dict()
    hitl_count = int(combined_df["is_hitl_active_learning"].sum()) if "is_hitl_active_learning" in combined_df.columns else 0

    stats = {
        "status": "success",
        "total_resolved_records": len(records),
        "newly_ingested_images": newly_ingested,
        "already_ingested_images": already_ingested,
        "missing_source_images": missing_source_count,
        "total_active_learning_samples": hitl_count,
        "total_training_samples": len(combined_df),
        "class_distribution": class_distribution,
        "expanded_manifest_path": str(output_train_csv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Active Learning Dataset Expansion")
    parser.add_argument("--target-dir", type=str, default="data/expanded", help="Destination folder for expanded dataset")
    parser.add_argument("--base-csv", type=str, default="data/splits_cleaned/train.csv", help="Base training CSV")
    parser.add_argument("--output-csv", type=str, default="data/splits_cleaned/train_expanded.csv", help="Output expanded CSV")
    args = parser.parse_args()

    stats = ingest_resolved_samples(
        target_dir=Path(args.target_dir),
        base_train_csv=Path(args.base_csv),
        output_train_csv=Path(args.output_csv),
    )

    print("\n" + "=" * 60)
    print(" Active Learning Dataset Ingestion Summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f" {k:32s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()

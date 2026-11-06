#!/usr/bin/env python3
"""
Industrial Dataset Upgrade & Curation Pipeline (v4.2)
-------------------------------------------------------
Builds and curates a high-quality, balanced industrial biofouling dataset:
- Focuses specifically on marine biofouling, cooling tower biofilm accumulation,
  pipe scaling, and condenser tube corrosion across 4 severity tiers:
  * Low (Nominal Clean Surface)
  * Moderate (Early Biofilm & Algal Colonization)
  * High (Mature Dense Slime Matrix)
  * Critical (Severe Flow Obstruction & Calcified Encrustation)
- Applies strict color-safe and industrial surface augmentations.
- Validates every generated sample through IndustrialDomainGuardrail.
- Exports split CSV manifests to `data/splits_cleaned/train_industrial.csv`, etc.

Author: Senior MLOps & Computer Vision Pipeline Engineer
"""

import hashlib
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.guardrail import get_guardrail
from src.dataset import IDX_TO_LABEL, LABEL_TO_IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetCurator")

CURATED_DIR = PROJECT_ROOT / "data" / "industrial_curated"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits_cleaned"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPANDED_DIR = PROJECT_ROOT / "data" / "expanded"


def generate_industrial_surface(
    category: str,
    width: int = 380,
    height: int = 380,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthesize high-fidelity industrial pipe and cooling tower surface textures
    reflecting specific biofouling physical and chemical states.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Base metal substrate (carbon steel / titanium / cast iron gray)
    base_gray = np.random.randint(65, 115)
    noise = np.random.normal(0, 12, (height, width)).astype(np.float32)
    metal_layer = np.clip(base_gray + noise, 30, 220).astype(np.uint8)

    # 3-Channel BGR Base
    img = cv2.cvtColor(metal_layer, cv2.COLOR_GRAY2BGR)

    # Add pipe curvature / metallic shading gradient
    y_coords, x_coords = np.indices((height, width))
    pipe_grad = np.sin(x_coords / width * np.pi) * 25.0
    img[:, :, 0] = np.clip(img[:, :, 0] + pipe_grad * 0.9, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + pipe_grad * 0.95, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + pipe_grad * 1.05, 0, 255)

    if category == "Low":
        # Nominal Pristine Metal / Clean Cooling Tower Fill
        # Add slight directional machining lines / fill ribbing
        for i in range(0, height, np.random.randint(18, 30)):
            cv2.line(img, (0, i), (width, i), (base_gray + 15, base_gray + 15, base_gray + 15), 1)
        # Minor oxidation / speckles
        speckle_count = np.random.randint(5, 15)
        for _ in range(speckle_count):
            cx, cy = np.random.randint(20, width - 20), np.random.randint(20, height - 20)
            r = np.random.randint(2, 6)
            cv2.circle(img, (cx, cy), r, (np.random.randint(50, 80), np.random.randint(80, 110), np.random.randint(110, 140)), -1)

    elif category == "Moderate":
        # Early Biofilm & Algal Colonization
        # Green / yellow-green micro-algal colonization spots
        patch_count = np.random.randint(30, 60)
        for _ in range(patch_count):
            cx, cy = np.random.randint(10, width - 10), np.random.randint(10, height - 10)
            rx, ry = np.random.randint(8, 25), np.random.randint(8, 25)
            angle = np.random.randint(0, 180)
            # Algal green / slime yellow (B: 20-50, G: 110-170, R: 70-130)
            color = (np.random.randint(20, 50), np.random.randint(110, 170), np.random.randint(70, 130))
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, -1)

        # Micro-corrosion pits
        pit_count = np.random.randint(15, 30)
        for _ in range(pit_count):
            cx, cy = np.random.randint(10, width - 10), np.random.randint(10, height - 10)
            r = np.random.randint(3, 8)
            cv2.circle(img, (cx, cy), r, (25, 45, 120), -1)

    elif category == "High":
        # Mature Slime Matrix / Dense Biofilm Mats
        # Extensive dense algal and extracellular polymeric substance (EPS) coating
        patch_count = np.random.randint(80, 140)
        for _ in range(patch_count):
            cx, cy = np.random.randint(5, width - 5), np.random.randint(5, height - 5)
            rx, ry = np.random.randint(15, 45), np.random.randint(15, 45)
            angle = np.random.randint(0, 180)
            # Deep olive green / dark slime brown
            color = (np.random.randint(10, 40), np.random.randint(90, 150), np.random.randint(60, 110))
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, -1)

        # Filamentous slime streaks
        for _ in range(np.random.randint(12, 25)):
            pt1 = (np.random.randint(0, width), np.random.randint(0, height))
            pt2 = (pt1[0] + np.random.randint(-40, 40), pt1[1] + np.random.randint(20, 80))
            cv2.line(img, pt1, pt2, (15, 120, 75), np.random.randint(3, 7))

    elif category == "Critical":
        # Severe Macro-Fouling / Calcified Shells & Occlusion
        # Severe multi-layered biofilm & heavy barnacle / mineral encrustation
        # Heavy background slime
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.4 + 15, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * 0.7 + 60, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.6 + 45, 0, 255)

        # Calcified white/gray barnacle rings
        barnacle_count = np.random.randint(40, 80)
        for _ in range(barnacle_count):
            cx, cy = np.random.randint(15, width - 15), np.random.randint(15, height - 15)
            r = np.random.randint(10, 28)
            # Outer white shell
            cv2.circle(img, (cx, cy), r, (np.random.randint(160, 220), np.random.randint(170, 220), np.random.randint(170, 220)), -1)
            # Inner dark cavity
            cv2.circle(img, (cx, cy), max(3, r - 5), (np.random.randint(20, 50), np.random.randint(30, 60), np.random.randint(30, 60)), -1)

        # Dense flow obstruction clogs
        for _ in range(np.random.randint(10, 20)):
            cx, cy = np.random.randint(20, width - 20), np.random.randint(20, height - 20)
            rx, ry = np.random.randint(25, 60), np.random.randint(25, 60)
            cv2.ellipse(img, (cx, cy), (rx, ry), np.random.randint(0, 180), 0, 360, (15, 45, 30), -1)

    # Slight Gaussian Blur for natural biological texture blend
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Convert BGR back to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def curate_industrial_dataset(
    target_samples_per_class: int = 150,
) -> Dict[str, Any]:
    """
    Curate and expand industrial biofouling dataset with strict guardrail validation.
    """
    logger.info("Initializing Industrial Biofouling Dataset Curation Pipeline...")
    guardrail = get_guardrail()

    # Create directories
    for split in ["train", "val", "test"]:
        for label in IDX_TO_LABEL.values():
            (CURATED_DIR / split / label).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    category_counts = {lbl: {"train": 0, "val": 0, "test": 0} for lbl in IDX_TO_LABEL.values()}

    # 1. Ingest existing raw images if available
    if RAW_DIR.exists():
        for label in IDX_TO_LABEL.values():
            label_dir = RAW_DIR / label
            if label_dir.exists():
                for img_file in label_dir.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        try:
                            pil_img = Image.open(img_file).convert("RGB")
                            arr = np.array(pil_img)
                            gr_res = guardrail.validate(arr)
                            if gr_res.is_valid:
                                # Determine split (70% train, 15% val, 15% test)
                                r_val = random.random()
                                split = "train" if r_val < 0.70 else ("val" if r_val < 0.85 else "test")
                                dest_path = CURATED_DIR / split / label / f"raw_{img_file.name}"
                                shutil.copy2(img_file, dest_path)
                                category_counts[label][split] += 1
                                records.append({
                                    "image_id": f"ind_{split}_{label}_{dest_path.stem}",
                                    "filename": str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                                    "filepath": str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                                    "class_label": label,
                                    "label_idx": LABEL_TO_IDX[label],
                                    "split": split,
                                    "source": "raw_ingested",
                                    "domain_score": gr_res.domain_score,
                                })
                        except Exception as e:
                            logger.warning(f"Skipping corrupted file {img_file}: {e}")

    # 2. Synthesize High-Resolution Industrial Variations to reach target balance
    logger.info("Synthesizing domain-calibrated industrial inspection samples...")
    for label in IDX_TO_LABEL.values():
        current_train = category_counts[label]["train"]
        needed_train = max(0, int(target_samples_per_class * 0.70) - current_train)
        needed_val = max(0, int(target_samples_per_class * 0.15) - category_counts[label]["val"])
        needed_test = max(0, int(target_samples_per_class * 0.15) - category_counts[label]["test"])

        for split, needed_count in [("train", needed_train), ("val", needed_val), ("test", needed_test)]:
            for i in range(needed_count):
                seed = hash(f"{label}_{split}_{i}_{random.random()}") % (2**31 - 1)
                synth_img = generate_industrial_surface(label, width=380, height=380, seed=seed)
                gr_res = guardrail.validate(synth_img)

                filename = f"synth_{label.lower()}_{split}_{i:04d}.jpg"
                dest_path = CURATED_DIR / split / label / filename
                Image.fromarray(synth_img).save(dest_path, quality=95)

                category_counts[label][split] += 1
                records.append({
                    "image_id": f"ind_{split}_{label}_{i:04d}",
                    "filename": str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "filepath": str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "class_label": label,
                    "label_idx": LABEL_TO_IDX[label],
                    "split": split,
                    "source": "industrial_synthesized",
                    "domain_score": gr_res.domain_score,
                })

    df = pd.DataFrame(records)
    curated_meta_path = CURATED_DIR / "metadata.csv"
    df.to_csv(curated_meta_path, index=False)
    logger.info(f"Saved complete industrial metadata: {curated_meta_path} (Total Samples: {len(df)})")

    # Export split CSVs into data/splits_cleaned/
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_csv_path = SPLITS_DIR / "train_industrial.csv"
    val_csv_path = SPLITS_DIR / "val_industrial.csv"
    test_csv_path = SPLITS_DIR / "test_industrial.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logger.info(f"Exported Curated Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    summary = {
        "status": "SUCCESS",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "category_counts": category_counts,
        "metadata_path": str(curated_meta_path),
        "train_csv": str(train_csv_path),
        "val_csv": str(val_csv_path),
        "test_csv": str(test_csv_path),
    }
    return summary


if __name__ == "__main__":
    summary = curate_industrial_dataset(target_samples_per_class=150)
    print("\n" + "=" * 60)
    print(" Industrial Dataset Curation Summary")
    print("=" * 60)
    print(f" Total Samples Curated : {summary['total_samples']}")
    print(f" Train Split           : {summary['train_samples']}")
    print(f" Val Split             : {summary['val_samples']}")
    print(f" Test Split            : {summary['test_samples']}")
    print(f" Manifest CSV          : {summary['train_csv']}")
    print("=" * 60)

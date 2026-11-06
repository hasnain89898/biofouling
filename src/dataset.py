#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader Pipeline for 4-Class Industrial Biofouling
-----------------------------------------------------------------------
Implements BiofoulingDataset and get_dataloaders() with Albumentations
transformations and strict diagnostic color guardrails.

Author: Senior Computer Vision & PyTorch Engineer
"""

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingDataset")

# Label mappings
LABEL_TO_IDX: Dict[str, int] = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3,
}

IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}

# ImageNet standard normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==============================================================================
# Albumentations Transformations with Strict Color Guardrails (380x380)
# ==============================================================================

def get_train_transforms(image_size: int = 380) -> A.Compose:
    """
    Build data augmentation pipeline for training with strict diagnostic color guardrails.

    CRITICAL COLOR GUARDRAIL:
    Do NOT apply hue, saturation, color-jitter, or ChannelShuffle augmentations.
    Color contrast (e.g. green algae vs. brown/amber biofilm matrix vs. silver metallic pipe)
    is the primary discriminative visual signature for industrial biofouling assessment.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.12,
            rotate_limit=15,
            p=0.5,
            border_mode=0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.3,
        ),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_hardened_train_transforms(image_size: int = 380) -> A.Compose:
    """
    Build aggressive regularization augmentation pipeline for model hardening (v4).
    Includes CoarseDropout masking and GaussNoise scaled for 380x380 resolution to prevent
    memorization and force micro-structural feature learning, while strictly preserving
    the biofouling color guardrail.
    """
    # Safe instantiation of CoarseDropout and GaussNoise across albumentations versions
    hole_size = int(image_size * 0.08)  # ~30px for 380x380
    try:
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(12, max(16, hole_size)),
            hole_width_range=(12, max(16, hole_size)),
            p=0.5,
        )
    except TypeError:
        coarse_dropout = A.CoarseDropout(
            max_holes=8,
            max_height=max(16, hole_size),
            max_width=max(16, hole_size),
            p=0.5,
        )

    try:
        gauss_noise = A.GaussNoise(std_range=(0.08, 0.20), p=0.3)
    except (TypeError, ValueError):
        gauss_noise = A.GaussNoise(var_limit=(10.0, 40.0), p=0.3)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.12,
            rotate_limit=15,
            p=0.5,
            border_mode=0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.3,
        ),
        gauss_noise,
        coarse_dropout,
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 380) -> A.Compose:
    """
    Build validation / testing transformation pipeline (deterministic 380x380 resizing + normalization).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# ==============================================================================
# PyTorch Dataset
# ==============================================================================

class BiofoulingDataset(Dataset):
    """
    PyTorch Dataset for 4-class industrial biofouling images.
    """

    def __init__(
        self,
        csv_file_or_df: Union[str, Path, pd.DataFrame],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_file_or_df: Path to split CSV file (or loaded DataFrame).
            root_dir: Base directory for image paths (defaults to current project root).
            transform: Albumentations transformation pipeline.
        """
        if isinstance(csv_file_or_df, (str, Path)):
            csv_path = Path(csv_file_or_df)
            if not csv_path.exists():
                raise FileNotFoundError(f"Split CSV file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
        elif isinstance(csv_file_or_df, pd.DataFrame):
            self.df = csv_file_or_df.copy()
        else:
            raise TypeError("csv_file_or_df must be a filepath string, Path, or pd.DataFrame")

        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.transform = transform

        # Validate and normalize required columns
        if "filename" not in self.df.columns and "filepath" in self.df.columns:
            self.df["filename"] = self.df["filepath"]

        required_cols = {"filename", "class_label", "label_idx"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain columns {required_cols}, found: {set(self.df.columns)}")

        self.filenames = self.df["filename"].tolist()
        self.labels = self.df["label_idx"].astype(int).tolist()
        self.class_labels = self.df["class_label"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load, transform, and return image tensor and integer class label.

        Returns:
            (image_tensor, label_tensor):
                - image_tensor: torch.FloatTensor of shape (3, H, W)
                - label_tensor: torch.LongTensor scalar with value in {0, 1, 2, 3}
        """
        rel_path = self.filenames[idx]
        img_path = self.root_dir / rel_path

        if not img_path.exists():
            # Try direct path
            img_path = Path(rel_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path} (idx: {idx})")

        # Load RGB image via PIL
        try:
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"Error decoding image {img_path}: {e}")

        # Apply Albumentations transformation
        if self.transform is not None:
            augmented = self.transform(image=img_np)
            image_tensor = augmented["image"]
        else:
            # Fallback to ToTensor
            image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_tensor, label_tensor


# ==============================================================================
# DataLoader Constructor Helper
# ==============================================================================

def get_dataloaders(
    splits_dir: Union[str, Path] = "data/splits_cleaned",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 380,
    pin_memory: bool = True,
    root_dir: Optional[Union[str, Path]] = None,
    hardened: bool = False,
    train_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct PyTorch DataLoaders for train, validation, and test splits (380x380).

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv.
        batch_size: Batch size for loaders (default: 16).
        num_workers: Number of subprocesses for data loading (default: 2).
        image_size: Target square image resolution (default: 380).
        pin_memory: If True, copies Tensors into CUDA pinned memory.
        root_dir: Root directory for resolving relative image paths.
        hardened: If True, uses aggressive regularization transforms (CoarseDropout + GaussNoise).
        train_transform: Optional explicit custom training transform.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    splits_path = Path(splits_dir)
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"Split file {p} does not exist. Run data prep / split scripts first."
            )

    if train_transform is not None:
        selected_train_transforms = train_transform
    elif hardened:
        selected_train_transforms = get_hardened_train_transforms(image_size=image_size)
    else:
        selected_train_transforms = get_train_transforms(image_size=image_size)

    val_transforms = get_val_transforms(image_size=image_size)

    train_dataset = BiofoulingDataset(train_csv, root_dir=root_dir, transform=selected_train_transforms)
    val_dataset = BiofoulingDataset(val_csv, root_dir=root_dir, transform=val_transforms)
    test_dataset = BiofoulingDataset(test_csv, root_dir=root_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders initialized (380x380) | Train: {len(train_dataset)} samples ({len(train_loader)} batches) | "
        f"Val: {len(val_dataset)} samples ({len(val_loader)} batches) | "
        f"Test: {len(test_dataset)} samples ({len(test_loader)} batches) | Batch Size: {batch_size}"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing Biofouling Dataset & DataLoader Pipeline (380x380)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_dir="data/splits_cleaned",
        batch_size=8,
        num_workers=0,  # Single-process for standalone test
        image_size=380,
    )

    for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        images, labels = next(iter(loader))
        print(f"\n[{split_name} Loader Batch Inspection]")
        print(f"  Image batch shape: {images.shape} (dtype: {images.dtype})")
        print(f"  Label batch shape: {labels.shape} (dtype: {labels.dtype})")
        print(f"  Labels in batch:   {labels.tolist()} -> {[IDX_TO_LABEL[i] for i in labels.tolist()]}")
        print(f"  Tensor Min/Max:    [{images.min():.3f}, {images.max():.3f}] (normalized)")
        assert images.shape == (8, 3, 380, 380), f"Expected shape (8, 3, 380, 380), got {images.shape}"

    print("\nDataset & DataLoader (380x380) self-test successfully completed!")

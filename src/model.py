#!/usr/bin/env python3
"""
Model Architecture for 4-Class Industrial Biofouling Assessment
--------------------------------------------------------------
Implements BiofoulingClassifier with timm pretrained backbones (EfficientNet-B0 /
ResNet-18), custom dropout-regularized classification head, and explicit parameter
freezing / unfreezing methods for two-stage transfer learning.

Author: Senior Computer Vision & PyTorch Engineer
"""

import logging
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("BiofoulingModel")


class BiofoulingClassifier(nn.Module):
    """
    Industrial Biofouling Assessment Classifier (v4).

    Wraps a pretrained computer vision backbone from timm (default: efficientnet_b3),
    replaces the final classification layer with a custom dropout-regularized linear head,
    and provides modular methods to freeze/unfreeze backbone parameters for two-stage transfer learning.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = 4,
        pretrained: bool = True,
        drop_rate: float = 0.4,
    ):
        """
        Args:
            model_name: Name of the timm backbone ('efficientnet_b3', 'efficientnet_b2', etc.)
            num_classes: Number of target biofouling severity classes (default: 4).
            pretrained: Whether to load ImageNet-pretrained weights (default: True).
            drop_rate: Dropout probability before classification layer (default: 0.4).
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # Load backbone without classification head (num_classes=0 returns pooled feature tensor)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Retrieve feature dimension
        self.in_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.in_features, num_classes),
        )

        logger.info(
            f"Initialized BiofoulingClassifier: backbone={model_name} (in_features={self.in_features}), "
            f"num_classes={num_classes}, dropout={drop_rate}, pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input batch of shape (B, 3, H, W).
        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters for Stage 1 head warm-up training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure classifier parameters remain trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters FROZEN. Only classification head is trainable.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters for Stage 2 end-to-end fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters UNFROZEN. Entire network is trainable.")

    def get_parameter_summary(self) -> Dict[str, int]:
        """
        Return total, trainable, and frozen parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
        }


def create_biofouling_model(
    model_name: str = "efficientnet_b3",
    num_classes: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.4,
) -> BiofoulingClassifier:
    """
    Factory helper function to instantiate a BiofoulingClassifier.
    """
    return BiofoulingClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing BiofoulingClassifier instantiation and freeze/unfreeze mechanisms (380x380)...")

    model = create_biofouling_model("efficientnet_b3", num_classes=4, pretrained=True, drop_rate=0.4)

    # Initial state (all trainable)
    s0 = model.get_parameter_summary()
    print(f"Initial State: Total={s0['total_params']:,} | Trainable={s0['trainable_params']:,} | Frozen={s0['frozen_params']:,}")

    # Stage 1: Freeze backbone
    model.freeze_backbone()
    s1 = model.get_parameter_summary()
    print(f"Stage 1 (Frozen): Total={s1['total_params']:,} | Trainable={s1['trainable_params']:,} | Frozen={s1['frozen_params']:,}")
    assert s1["frozen_params"] > 0, "Backbone should have frozen parameters"
    assert s1["trainable_params"] == (model.in_features * 4 + 4), "Only classifier layer should be trainable"

    # Stage 2: Unfreeze backbone
    model.unfreeze_backbone()
    s2 = model.get_parameter_summary()
    print(f"Stage 2 (Unfrozen): Total={s2['total_params']:,} | Trainable={s2['trainable_params']:,} | Frozen={s2['frozen_params']:,}")
    assert s2["frozen_params"] == 0, "All parameters should be trainable"

    # Test forward pass with 380x380 resolution
    dummy_input = torch.randn(4, 3, 380, 380)
    logits = model(dummy_input)
    print(f"Forward pass output logits shape: {logits.shape} (Expected: [4, 4])")
    assert logits.shape == (4, 4), f"Expected shape (4, 4), got {logits.shape}"

    print("BiofoulingClassifier self-test passed successfully!")

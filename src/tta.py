#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Engine for Industrial Biofouling Assessment
-----------------------------------------------------------------------
Implements TestTimeAugmentor for PyTorch models:
Evaluates each test image across 4 distinct spatial augmentations:
1. Original (identity)
2. Horizontal Flip
3. Vertical Flip
4. 90-degree Rotation
Averages softmax probability vectors across all 4 spatial views before
determining final argmax prediction.

Author: Principal AI Research Engineer
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestTimeAugmentor:
    """
    Test-Time Augmentation (TTA) wrapper.
    Averages predictions across 4 spatial symmetry transformations.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @staticmethod
    def get_views(x: torch.Tensor, num_views: int = 8) -> List[torch.Tensor]:
        """
        Generate spatial augmented views of input tensor batch (B, C, H, W).
        Uses D4 dihedral group symmetries (4 rotations x 2 reflections = 8 views):
        1. Original (identity)
        2. Horizontal Flip
        3. Vertical Flip
        4. 90-degree Rotation
        5. 180-degree Rotation
        6. 270-degree Rotation
        7. Horizontal Flip + 90-degree Rotation
        8. Vertical Flip + 90-degree Rotation
        """
        v_orig = x
        v_hflip = torch.flip(x, dims=[-1])
        v_vflip = torch.flip(x, dims=[-2])
        v_rot90 = torch.rot90(x, k=1, dims=[-2, -1])
        
        if num_views <= 4:
            return [v_orig, v_hflip, v_vflip, v_rot90]

        v_rot180 = torch.rot90(x, k=2, dims=[-2, -1])
        v_rot270 = torch.rot90(x, k=3, dims=[-2, -1])
        v_h_rot90 = torch.rot90(v_hflip, k=1, dims=[-2, -1])
        v_v_rot90 = torch.rot90(v_vflip, k=1, dims=[-2, -1])

        return [v_orig, v_hflip, v_vflip, v_rot90, v_rot180, v_rot270, v_h_rot90, v_v_rot90]

    @staticmethod
    def get_4_views(x: torch.Tensor) -> List[torch.Tensor]:
        return TestTimeAugmentor.get_views(x, num_views=4)

    @staticmethod
    def get_8_views(x: torch.Tensor) -> List[torch.Tensor]:
        return TestTimeAugmentor.get_views(x, num_views=8)

    @torch.no_grad()
    def predict_batch(
        self, x: torch.Tensor, num_views: int = 8, return_all_probs: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Run multi-view TTA inference on input batch tensor (B, C, H, W).

        Returns:
            (tta_probs, tta_preds)
            - tta_probs: Tensor of shape (B, num_classes) with ensembled softmax probabilities
            - tta_preds: Tensor of shape (B,) with class indices in {0, 1, 2, 3}
        """
        self.model.eval()
        x = x.to(self.device)
        views = self.get_views(x, num_views=num_views)
        view_probs = []

        for v in views:
            logits = self.model(v)
            probs = F.softmax(logits, dim=-1)
            view_probs.append(probs)

        # Average softmax probabilities across the spatial transformations
        tta_probs = torch.stack(view_probs, dim=0).mean(dim=0)
        tta_preds = torch.argmax(tta_probs, dim=-1)

        if return_all_probs:
            return tta_probs, tta_preds, view_probs
        return tta_probs, tta_preds

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None,
        num_views: int = 8,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate full DataLoader using D4 multi-view spatial TTA.

        Returns:
            (avg_loss, accuracy, macro_f1, targets, preds, probs)
        """
        from sklearn.metrics import f1_score

        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            tta_probs, tta_preds = self.predict_batch(images, num_views=num_views)

            if criterion is not None:
                # Loss on raw model logits or cross-entropy on log(prob)
                raw_logits = self.model(images)
                loss = criterion(raw_logits, labels)
                running_loss += loss.item() * images.size(0)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(tta_preds.cpu().numpy())
            all_probs.extend(tta_probs.cpu().numpy())

        total = len(all_targets)
        avg_loss = running_loss / total if total > 0 else 0.0
        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds)
        probs_np = np.array(all_probs)

        acc = float(np.mean(targets_np == preds_np)) if total > 0 else 0.0
        macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))

        return avg_loss, acc, macro_f1, targets_np, preds_np, probs_np

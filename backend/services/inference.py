import numpy as np
import albumentations as A
import onnxruntime as ort
import torch
from typing import Tuple, List

class ONNXInferenceTTAWrapper:
    """
    Lightweight Test-Time Augmentation (TTA) wrapper for ONNX inference.
    Generates variants of an input image, runs ONNX inference on all,
    and returns the consensus probability vector and entropy score.
    """
    def __init__(self, session: ort.InferenceSession, val_transforms: A.Compose, input_name: str, output_name: str):
        self.session = session
        self.val_transforms = val_transforms
        self.input_name = input_name
        self.output_name = output_name
        
        # Define TTA specific transforms (applied BEFORE val_transforms)
        self.tta_transforms = [
            A.Compose([]),  # Variant 0: Original
            A.Compose([A.HorizontalFlip(p=1.0)]),  # Variant 1: Horizontal Flip
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=0.0, p=1.0)]),  # Variant 2: Slight Brightness increase
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(0.1, 0.2), p=1.0)]),  # Variant 3: Slight Contrast increase
        ]

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Compute normalized Shannon entropy in [0.0, 1.0]."""
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))  # log2(4 classes) = 2.0
        return float(entropy / max_entropy)

    def predict_tta(self, image_np: np.ndarray) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Runs TTA prediction.
        
        Args:
            image_np: Raw RGB numpy image array (uint8)
            
        Returns:
            consensus_probs: The averaged probability vector (shape: (4,))
            entropy: The Shannon entropy of the consensus vector.
            individual_probs: List of probability vectors for each TTA variant.
        """
        individual_probs = []
        for tta_t in self.tta_transforms:
            # Apply TTA variant transformation
            aug_img = tta_t(image=image_np)["image"]
            
            # Apply standard validation transforms (Resize, Normalize, ToTensorV2)
            tensor_img = self.val_transforms(image=aug_img)["image"]
            input_batch = tensor_img.unsqueeze(0).numpy().astype(np.float32)
            
            # Run ONNX inference
            ort_inputs = {self.input_name: input_batch}
            logits_np = self.session.run([self.output_name], ort_inputs)[0]
            
            # Softmax
            logits_tensor = torch.from_numpy(logits_np)
            probs = torch.nn.functional.softmax(logits_tensor, dim=1).squeeze(0).numpy()
            individual_probs.append(probs)
            
        # Compute consensus (average) across all 4 variants
        consensus_probs = np.mean(np.array(individual_probs), axis=0)
        entropy = self._calculate_entropy(consensus_probs)
        
        return consensus_probs, entropy, individual_probs

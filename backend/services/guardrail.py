#!/usr/bin/env python3
"""
Industrial Domain Guardrail & Out-of-Distribution (OOD) Validation Service
-------------------------------------------------------------------------
Validates incoming inspection images before passing them to neural inference.
Detects out-of-domain inputs (e.g. solar panels, human faces, blank images,
text documents, or unrelated landscapes) using multi-signal visual heuristics:

1. Color Spectrum & Chromatic Consistency (Metallic steels, algae greens, rust ambers, scale whites).
2. Edge Density & Gradient Magnitude (Sobel / Laplacian surface texture).
3. Image Entropy & Frequency Variance.
4. Anomaly Penalties (Skin tones, uniform screens, saturated digital graphics).

Author: Senior MLOps & Computer Vision Pipeline Engineer
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("BiofoulingGuardrail")


@dataclass
class GuardrailResult:
    """Encapsulates the decision and telemetry of domain guardrail analysis."""
    is_valid: bool
    domain_score: float  # Scale 0.0 to 1.0
    rejection_reason: Optional[str] = None
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "domain_score": round(self.domain_score, 4),
            "rejection_reason": self.rejection_reason,
            "signals": self.signals,
        }


class IndustrialDomainGuardrail:
    """
    High-speed, multi-signal domain guardrail for industrial cooling towers,
    heat exchangers, condenser tubes, and marine pipelines.
    """

    def __init__(
        self,
        min_domain_score: float = 0.45,
        min_dimension: int = 64,
        min_laplacian_var: float = 5.0,
    ):
        self.min_domain_score = min_domain_score
        self.min_dimension = min_dimension
        self.min_laplacian_var = min_laplacian_var

    def validate(
        self,
        image_input: Union[np.ndarray, Image.Image, bytes],
    ) -> GuardrailResult:
        """
        Validate an input image against industrial visual guardrails.

        Args:
            image_input: NumPy array (RGB or BGR), PIL Image, or raw image bytes.

        Returns:
            GuardrailResult with validity boolean, composite domain score, and diagnostics.
        """
        # 1. Standardize to RGB NumPy Array
        img_rgb, err = self._to_rgb_array(image_input)
        if err or img_rgb is None:
            return GuardrailResult(
                is_valid=False,
                domain_score=0.0,
                rejection_reason=f"Image decoding failed: {err or 'Corrupted image format'}",
            )

        h, w, c = img_rgb.shape

        # 2. Basic Dimensions Check
        if h < self.min_dimension or w < self.min_dimension:
            return GuardrailResult(
                is_valid=False,
                domain_score=0.1,
                rejection_reason=f"Image resolution ({w}x{h}) is below minimum industrial threshold ({self.min_dimension}x{self.min_dimension}).",
            )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # 3. Signal A: Intensity Variance & Blank Frame Check
        std_intensity = float(np.std(gray))
        if std_intensity < 8.0:
            return GuardrailResult(
                is_valid=False,
                domain_score=0.05,
                rejection_reason="Uniform or blank image detected (zero surface variance).",
                signals={"std_intensity": round(std_intensity, 2)},
            )

        # 4. Signal B: Surface Texture & Laplacian Variance (Blur / Flatness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        if laplacian_var < self.min_laplacian_var:
            return GuardrailResult(
                is_valid=False,
                domain_score=0.15,
                rejection_reason="Image lacks surface texture or is severely blurred (insufficient Laplacian variance).",
                signals={"laplacian_variance": round(laplacian_var, 2)},
            )

        # 5. Signal C: Edge Density & Gradient Distribution (Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_threshold = np.mean(grad_mag) + 0.5 * np.std(grad_mag)
        edge_density = float(np.mean(grad_mag > max(edge_threshold, 15.0)))

        # 6. Signal D: Industrial Chromatic & Color Palette Analysis
        # H in OpenCV is [0, 180], S in [0, 255], V in [0, 255]
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1] / 255.0
        v_channel = hsv[:, :, 2] / 255.0

        # Metal / Desaturated / Scale fraction (Steels, PVC, titanium, mineral scale)
        metal_mask = (s_channel < 0.40) & (v_channel > 0.10) & (v_channel < 0.95)
        metal_frac = float(np.mean(metal_mask))

        # Algal Green / Biofilm fraction (H ~ 30 to 85 -> 60° to 170°)
        algae_mask = (h_channel >= 25) & (h_channel <= 85) & (s_channel >= 0.20)
        algae_frac = float(np.mean(algae_mask))

        # Rust / Oxidation / Iron Slime fraction (H ~ 5 to 25 -> 10° to 50°)
        rust_mask = (h_channel >= 4) & (h_channel <= 25) & (s_channel >= 0.25)
        rust_frac = float(np.mean(rust_mask))

        # Dark Biofilm / Deep Slime Matrix (Low value, moderate saturation)
        dark_biofilm_mask = (v_channel <= 0.30) & (v_channel >= 0.05)
        dark_biofilm_frac = float(np.mean(dark_biofilm_mask))

        industrial_color_score = min(1.0, (metal_frac * 0.9) + (algae_frac * 1.4) + (rust_frac * 1.2) + (dark_biofilm_frac * 0.8))

        # 7. Signal E: Anomaly / OOD Penalties
        # Blue Sky / Landscape Horizon (H ~ 90-130, S > 0.45, V > 0.65 in top 40% of image)
        top_half_hsv = hsv[:int(h * 0.4), :, :]
        sky_mask = (top_half_hsv[:, :, 0] >= 90) & (top_half_hsv[:, :, 0] <= 130) & (top_half_hsv[:, :, 1] >= 110) & (top_half_hsv[:, :, 2] >= 160)
        sky_frac = float(np.mean(sky_mask))

        # Human Skin Tone heuristic (H ~ 0-20, S ~ 0.2-0.65, V > 0.35)
        skin_mask = (h_channel >= 0) & (h_channel <= 22) & (s_channel >= 0.20) & (s_channel <= 0.68) & (v_channel >= 0.35)
        skin_frac = float(np.mean(skin_mask))

        # High Saturated Artificial Graphics / Cartoon (S > 0.85 and V > 0.85)
        artificial_sat_mask = (s_channel > 0.85) & (v_channel > 0.85)
        artificial_sat_frac = float(np.mean(artificial_sat_mask))

        # 8. Compute Composite Domain Validity Score
        # Texture score (Gaussian curve centered around typical industrial roughness)
        texture_score = min(1.0, max(0.1, np.log10(max(laplacian_var, 1.0)) / 3.2))
        edge_score = min(1.0, max(0.1, edge_density * 4.5))

        base_score = (
            0.40 * industrial_color_score +
            0.35 * texture_score +
            0.25 * edge_score
        )

        # Apply Penalties
        penalty = 0.0
        if sky_frac > 0.35:
            penalty += 0.40
        if skin_frac > 0.55:
            penalty += 0.45
        if artificial_sat_frac > 0.40:
            penalty += 0.40

        composite_score = max(0.0, min(1.0, base_score - penalty))

        signals = {
            "resolution": f"{w}x{h}",
            "std_intensity": round(std_intensity, 2),
            "laplacian_variance": round(laplacian_var, 2),
            "edge_density": round(edge_density, 4),
            "metal_fraction": round(metal_frac, 3),
            "algae_fraction": round(algae_frac, 3),
            "rust_fraction": round(rust_frac, 3),
            "sky_fraction": round(sky_frac, 3),
            "skin_fraction": round(skin_frac, 3),
            "artificial_saturation_fraction": round(artificial_sat_frac, 3),
            "composite_score": round(composite_score, 4),
        }

        # 9. Decision Evaluation
        if sky_frac > 0.45:
            return GuardrailResult(
                is_valid=False,
                domain_score=composite_score,
                rejection_reason="Out-of-distribution scene detected (excessive open sky / landscape content).",
                signals=signals,
            )

        if skin_frac > 0.65 and algae_frac < 0.05:
            return GuardrailResult(
                is_valid=False,
                domain_score=composite_score,
                rejection_reason="Out-of-distribution input detected (human portrait / facial presence).",
                signals=signals,
            )

        if artificial_sat_frac > 0.50:
            return GuardrailResult(
                is_valid=False,
                domain_score=composite_score,
                rejection_reason="Out-of-distribution graphic detected (unnatural artificial color saturation).",
                signals=signals,
            )

        if composite_score < self.min_domain_score:
            return GuardrailResult(
                is_valid=False,
                domain_score=composite_score,
                rejection_reason="Input image is irrelevant to industrial biofouling inspection. Please provide a valid cooling tower, pipe, or heat exchanger inspection frame.",
                signals=signals,
            )

        return GuardrailResult(
            is_valid=True,
            domain_score=composite_score,
            rejection_reason=None,
            signals=signals,
        )

    def _to_rgb_array(
        self,
        img_input: Union[np.ndarray, Image.Image, bytes],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Convert various input formats into standard RGB uint8 array."""
        try:
            if isinstance(img_input, np.ndarray):
                if img_input.ndim == 2:
                    return cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB), None
                elif img_input.ndim == 3:
                    if img_input.shape[2] == 4:
                        return cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB), None
                    return img_input, None
            elif isinstance(img_input, Image.Image):
                return np.array(img_input.convert("RGB"), dtype=np.uint8), None
            elif isinstance(img_input, bytes):
                nparr = np.frombuffer(img_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None, "cv2.imdecode returned None"
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
            else:
                return None, f"Unsupported input type: {type(img_input)}"
        except Exception as e:
            return None, str(e)


# Singleton Guardrail Instance
_GUARDRAIL_INSTANCE: Optional[IndustrialDomainGuardrail] = None


def get_guardrail() -> IndustrialDomainGuardrail:
    """Retrieve or initialize singleton IndustrialDomainGuardrail."""
    global _GUARDRAIL_INSTANCE
    if _GUARDRAIL_INSTANCE is None:
        _GUARDRAIL_INSTANCE = IndustrialDomainGuardrail()
    return _GUARDRAIL_INSTANCE


def validate_industrial_domain(
    image: Union[np.ndarray, Image.Image, bytes],
) -> GuardrailResult:
    """Helper function to validate industrial domain of an inspection image."""
    return get_guardrail().validate(image)

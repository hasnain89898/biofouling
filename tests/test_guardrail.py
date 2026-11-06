#!/usr/bin/env python3
"""
Automated Test Suite for Industrial Domain Guardrail & OOD Filtering
--------------------------------------------------------------------
Tests:
1. Guardrail validation on valid industrial surfaces (metallic, biofilm, pipe).
2. Guardrail rejection on OOD images (blank white/black, sky-only, saturated digital).
3. FastAPI POST /predict integration with HTTP 400 INVALID_DOMAIN_INPUT.
4. FastAPI POST /predict with valid industrial image returning HTTP 200.

Author: Senior MLOps & Computer Vision Pipeline Engineer
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from backend.services.guardrail import (
    IndustrialDomainGuardrail,
    get_guardrail,
    validate_industrial_domain,
)
from src.curate_dataset import generate_industrial_surface

client = TestClient(app)


def create_blank_image(color=(255, 255, 255), size=(380, 380)) -> Image.Image:
    """Create uniform single-color image."""
    img_arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(img_arr)


def create_sky_image(size=(380, 380)) -> Image.Image:
    """Create synthetic out-of-domain blue sky landscape."""
    img_arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # Bright blue sky in top 60%
    img_arr[: int(size[1] * 0.6), :] = [100, 180, 255]  # RGB Sky Blue
    img_arr[int(size[1] * 0.6) :, :] = [60, 160, 60]   # Grass Green
    return Image.fromarray(img_arr)


def create_face_like_image(size=(380, 380)) -> Image.Image:
    """Create synthetic face/skin tone portrait image."""
    img_arr = np.full((size[1], size[0], 3), [240, 200, 175], dtype=np.uint8)  # Skin tone
    cv2.circle(img_arr, (190, 190), 120, (230, 180, 150), -1)
    return Image.fromarray(img_arr)


def test_guardrail_initialization():
    """Verify guardrail singleton and parameter defaults."""
    gr = get_guardrail()
    assert gr is not None
    assert isinstance(gr, IndustrialDomainGuardrail)
    assert gr.min_domain_score == 0.45


def test_valid_industrial_surfaces_pass():
    """Verify all 4 classes of synthesized industrial surfaces pass guardrail."""
    for category in ["Low", "Moderate", "High", "Critical"]:
        surface_rgb = generate_industrial_surface(category, width=380, height=380, seed=101)
        res = validate_industrial_domain(surface_rgb)
        assert res.is_valid is True, f"Valid {category} industrial surface rejected: {res.rejection_reason}"
        assert res.domain_score >= 0.45, f"Domain score too low for {category}: {res.domain_score}"


def test_ood_blank_images_rejected():
    """Verify uniform blank images are rejected by guardrail."""
    blank_white = np.full((380, 380, 3), 255, dtype=np.uint8)
    blank_black = np.full((380, 380, 3), 0, dtype=np.uint8)

    res_w = validate_industrial_domain(blank_white)
    assert res_w.is_valid is False
    assert "blank" in (res_w.rejection_reason or "").lower() or "texture" in (res_w.rejection_reason or "").lower()

    res_b = validate_industrial_domain(blank_black)
    assert res_b.is_valid is False


def test_ood_sky_landscape_rejected():
    """Verify excessive open sky / landscape images are rejected."""
    sky_img = create_sky_image()
    res = validate_industrial_domain(sky_img)
    assert res.is_valid is False
    assert "sky" in (res.rejection_reason or "").lower() or res.domain_score < 0.45


def test_api_predict_rejects_ood_image():
    """Verify POST /predict returns HTTP 400 with INVALID_DOMAIN_INPUT on OOD images."""
    blank_img = create_blank_image(color=(240, 240, 240))
    buf = io.BytesIO()
    blank_img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("blank_test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    detail = data.get("detail", {})
    assert detail.get("error_code") == "INVALID_DOMAIN_INPUT"
    assert "irrelevant" in detail.get("message", "").lower() or "blank" in detail.get("message", "").lower() or "texture" in detail.get("message", "").lower()


def test_api_predict_accepts_valid_industrial_image():
    """Verify POST /predict returns HTTP 200 and inference results on valid industrial frames."""
    ind_surface = generate_industrial_surface("Moderate", width=380, height=380, seed=777)
    buf = io.BytesIO()
    Image.fromarray(ind_surface).save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("pipe_inspection_frame.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "entropy" in data
    assert data["predicted_class"] in ["Low", "Moderate", "High", "Critical"]

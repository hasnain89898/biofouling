#!/usr/bin/env python3
"""
Automated Test Suite for FastAPI Biofouling REST API Backend
------------------------------------------------------------
Tests:
1. Healthcheck endpoint /health and model status.
2. High-confidence inference with automated decision approval.
3. Low-confidence / high-threshold inference triggering HITL escalation.
4. Review queue persistence (human_review_queue.json) and resolution endpoint.

Author: Principal AI Research Engineer
"""

import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

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

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

import database
from database import Base, ReviewQueueRecord, get_db, init_db

# Create isolated in-memory SQLite database with StaticPool for test session sharing
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Bind database module globals to test engine
database.engine = test_engine
database.SessionLocal = TestingSessionLocal

# Initialize test schema
Base.metadata.create_all(bind=test_engine)

from app import app


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


from src.curate_dataset import generate_industrial_surface


def create_dummy_image_bytes(color: tuple = None, category: str = "Moderate", size: tuple = (380, 380)) -> io.BytesIO:
    """Create in-memory industrial frame JPEG bytes for test upload."""
    surface = generate_industrial_surface(category, width=size[0], height=size[1], seed=42)
    img = Image.fromarray(surface)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_root_endpoint():
    """Verify root navigation and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Industrial Biofouling Assessment REST API"
    assert data["version"] == "4.0.0"
    assert "endpoints" in data


def test_healthcheck():
    """Verify healthcheck endpoint and ONNX model loading status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "380x380" in data["image_resolution"]
    assert data["operational_confidence_threshold"] == 0.80
    assert data["uptime_seconds"] >= 0


def test_predict_real_image():
    """Test inference on a real sample from dataset and verify customer UX & telemetry fields."""
    sample_img_path = PROJECT_ROOT / "data" / "raw" / "Critical" / "bf_critical_0001.jpg"
    if not sample_img_path.exists():
        # Fallback to in-memory image
        img_buf = create_dummy_image_bytes()
        files = {"file": ("test_frame.jpg", img_buf, "image/jpeg")}
    else:
        with open(sample_img_path, "rb") as f:
            files = {"file": ("bf_critical_0001.jpg", f.read(), "image/jpeg")}

    # Use low threshold (0.10) to guarantee AUTOMATED_DECISION for testing customer action copy
    response = client.post("/predict?threshold=0.10", files=files)
    assert response.status_code == 200
    data = response.json()

    # 1. Validate Customer & Field Inspector Presentation Layer
    assert "customer_message" in data
    assert "headline" in data
    assert "severity_assessment" in data
    assert "action_recommendation" in data
    assert data["gating_status"] == "AUTOMATED_DECISION"
    assert data["human_review_required"] is False
    assert "Inspection Complete:" in data["customer_message"]
    assert "Recommended action:" in data["customer_message"]
    assert len(data["action_recommendation"]) > 10

    # 2. Validate Backward-Compatible Telemetry
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Low", "Moderate", "High", "Critical"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert 0.0 <= data["probabilities"]["Low"] <= 1.0
    assert 0.0 <= data["probabilities"]["Critical"] <= 1.0
    assert data["latency_ms"] > 0
    assert "request_id" in data

    # 3. Validate Dedicated Technical Telemetry Container
    assert "telemetry" in data
    telem = data["telemetry"]
    assert telem["model_version"] == "4.0.0"
    assert "EfficientNet-B3" in telem["model_backbone"]
    assert telem["inference_engine"] == "ONNX Runtime"
    assert 0.0 <= telem["raw_confidence"] <= 1.0
    assert "%" in telem["confidence_percentage"]
    assert 0.0 <= telem["normalized_entropy"] <= 1.0
    assert "Uncertainty" in telem["uncertainty_level"]
    assert telem["latency_ms"] > 0
    assert telem["top1_class"] in ["Low", "Moderate", "High", "Critical"]


def test_hitl_escalation_and_queue():
    """Test that setting a 99% threshold forces escalation to the HITL review queue in database."""
    with TestingSessionLocal() as db:
        initial_db_count = db.query(ReviewQueueRecord).count()

    img_buf = create_dummy_image_bytes(color=(80, 120, 80))
    files = {"file": ("cooling_tower_pipe_001.jpg", img_buf, "image/jpeg")}

    # Force threshold = 0.99 so confidence is strictly below threshold
    response = client.post("/predict?threshold=0.99", files=files)
    assert response.status_code == 200
    data = response.json()

    # Verify HITL Customer-Facing Message
    assert data["gating_status"] == "HUMAN_EXPERT_REVIEW_REQUIRED"
    assert data["human_review_required"] is True
    assert "Ambiguous boundary" in data["review_reason"] or "below operational" in data["review_reason"]
    assert "Scan received" in data["customer_message"]
    assert "manual review by our inspection specialist" in data["customer_message"]
    assert "Specialist Review Queued" in data["headline"]

    # Verify database was updated with new ReviewQueueRecord
    with TestingSessionLocal() as db:
        updated_db_count = db.query(ReviewQueueRecord).count()
        assert updated_db_count == initial_db_count + 1
        db_record = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.request_id == data["request_id"]).first()
        assert db_record is not None
        assert db_record.filename == "cooling_tower_pipe_001.jpg"
        assert db_record.review_status == "PENDING_REVIEW"
        assert db_record.gating_status == "HUMAN_EXPERT_REVIEW_REQUIRED"
        assert isinstance(db_record.probabilities, dict)
        assert db_record.confidence == data["confidence"]
        assert db_record.created_at is not None
        queue_id = db_record.queue_id

    # Test GET /review-queue
    q_resp = client.get("/review-queue?status_filter=PENDING_REVIEW")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["pending_review_count"] >= 1
    assert any(item["queue_id"] == queue_id for item in q_data["queue_items"])

    # Test resolving the queue item via POST /review-queue/{queue_id}/resolve
    resolve_payload = {
        "resolved_label": "High",
        "resolved_by": "Dr. Smith (Lead Biofouling Specialist)",
        "notes": "Verified thick microalgal film on secondary heat exchanger plates.",
    }
    res_resp = client.post(f"/review-queue/{queue_id}/resolve", json=resolve_payload)
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    assert res_data["status"] == "RESOLVED"
    assert res_data["review_status"] == "RESOLVED"
    assert res_data["resolved_label"] == "High"
    assert res_data["resolved_by"] == "Dr. Smith (Lead Biofouling Specialist)"
    assert res_data["expert_notes"] == "Verified thick microalgal film on secondary heat exchanger plates."
    assert "Inspection Verified: Surface confirmed as High" in res_data["customer_message"]

    # Verify in DB directly
    with TestingSessionLocal() as db:
        resolved_record = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.queue_id == queue_id).first()
        assert resolved_record.review_status == "RESOLVED"
        assert resolved_record.resolved_label == "High"
        assert resolved_record.resolved_by == "Dr. Smith (Lead Biofouling Specialist)"
        assert resolved_record.expert_notes == "Verified thick microalgal film on secondary heat exchanger plates."
        assert resolved_record.resolved_at is not None


def test_database_orm_and_session():
    """Verify ReviewQueueRecord SQLAlchemy model schema and direct DB operations."""
    with TestingSessionLocal() as db:
        rec = ReviewQueueRecord(
            queue_id="Q-TEST-DB01",
            request_id="REQ-TEST-DB01",
            filename="direct_db_test.jpg",
            saved_image_path="data/review_queue_images/direct_db_test.jpg",
            predicted_class="Moderate",
            confidence=0.6850,
            entropy=0.4521,
            probabilities={"Low": 0.15, "Moderate": 0.685, "High": 0.10, "Critical": 0.065},
            gating_status="HUMAN_EXPERT_REVIEW_REQUIRED",
            review_status="PENDING_REVIEW",
            review_reason="Confidence below threshold",
            customer_message="Verification in progress",
            action_recommendation="Inspect biocide dosing",
            expert_notes="Initial test entry",
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        assert rec.id is not None
        assert rec.id > 0
        assert rec.filename == "direct_db_test.jpg"
        assert rec.predicted_class == "Moderate"
        assert rec.confidence == 0.6850
        assert rec.probabilities["Moderate"] == 0.685
        assert rec.gating_status == "HUMAN_EXPERT_REVIEW_REQUIRED"
        assert rec.review_status == "PENDING_REVIEW"
        assert rec.expert_notes == "Initial test entry"
        assert rec.created_at is not None

        # Test resolution using numeric ID endpoint lookup
        resolve_payload = {
            "resolved_label": "Moderate",
            "resolved_by": "Senior Biofouling Engineer",
            "notes": "Confirmed Moderate biofilm colonization.",
        }
        res_resp = client.post(f"/review-queue/{rec.id}/resolve", json=resolve_payload)
        assert res_resp.status_code == 200
        data = res_resp.json()
        assert data["status"] == "RESOLVED"
        assert data["resolved_label"] == "Moderate"


if __name__ == "__main__":
    print("Executing automated test suite for FastAPI biofouling backend...")
    test_root_endpoint()
    print("  ✓ test_root_endpoint PASSED")
    test_healthcheck()
    print("  ✓ test_healthcheck PASSED")
    test_predict_real_image()
    print("  ✓ test_predict_real_image PASSED")
    test_hitl_escalation_and_queue()
    print("  ✓ test_hitl_escalation_and_queue PASSED")
    test_database_orm_and_session()
    print("  ✓ test_database_orm_and_session PASSED")
    print("\nAll FastAPI tests PASSED successfully!")

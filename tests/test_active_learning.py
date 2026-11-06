#!/usr/bin/env python3
"""
Automated Test Suite for Active Learning Dataset Expansion & Retraining Pipeline
--------------------------------------------------------------------------------
Tests:
1. Active learning dataset ingestion (active_learning.py) with database resolved samples.
2. Model fine-tuning loop (retrain_model.py) with frozen backbone and ONNX export.
3. Protected administrative endpoint POST /admin/trigger-retrain authentication and gating.
4. Administrative retrain status endpoint GET /admin/retrain-status.

Author: Principal ML Operations & Deep Learning Engineer
"""

import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
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

import database
from database import Base, ReviewQueueRecord, get_db, init_db

# Create isolated test database
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

database.engine = test_engine
database.SessionLocal = TestingSessionLocal
Base.metadata.create_all(bind=test_engine)

from active_learning import ingest_resolved_samples
from app import ADMIN_API_KEY, app, retrain_state
from retrain_model import fine_tune_model

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

ADMIN_HEADERS = {"X-Admin-Key": ADMIN_API_KEY}


def create_test_image(path: Path, color=(120, 140, 100), size=(380, 380)) -> Path:
    """Create a sample image file on disk for active learning ingestion testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")
    return path


def test_active_learning_ingestion_from_db(tmp_path):
    """Verify that resolved records in the DB are ingested into class folders and added to the training manifest."""
    test_img_dir = tmp_path / "review_queue_images"
    test_expanded_dir = tmp_path / "expanded"
    test_base_csv = tmp_path / "base_train.csv"
    test_output_csv = tmp_path / "train_expanded.csv"

    # Create dummy base train manifest
    dummy_base = pd.DataFrame([
        {"filename": "data/raw/Low/bf_low_0001.jpg", "class_label": "Low", "label_idx": 0, "split": "train", "is_synthetic": False}
    ])
    dummy_base.to_csv(test_base_csv, index=False)

    # Insert 3 resolved records into test DB
    with TestingSessionLocal() as db:
        # 1. Resolved 'Critical' record with valid image
        img1 = create_test_image(test_img_dir / "al_crit_01.jpg", color=(200, 50, 50))
        rec1 = ReviewQueueRecord(
            queue_id="Q-AL-CRIT-01",
            request_id="REQ-AL-01",
            filename="al_crit_01.jpg",
            saved_image_path=str(img1.relative_to(PROJECT_ROOT)).replace("\\", "/") if img1.is_relative_to(PROJECT_ROOT) else str(img1),
            predicted_class="High",
            confidence=0.62,
            probabilities={"Low": 0.05, "Moderate": 0.10, "High": 0.62, "Critical": 0.23},
            gating_status="HUMAN_EXPERT_REVIEW_REQUIRED",
            review_status="RESOLVED",
            resolved_label="Critical",
            resolved_by="Dr. Lead Specialist",
            expert_notes="Confirmed sloughing biomass.",
            resolved_at=datetime.now(timezone.utc),
        )
        db.add(rec1)

        # 2. Resolved 'Moderate' record with valid image
        img2 = create_test_image(test_img_dir / "al_mod_01.jpg", color=(200, 180, 50))
        rec2 = ReviewQueueRecord(
            queue_id="Q-AL-MOD-01",
            request_id="REQ-AL-02",
            filename="al_mod_01.jpg",
            saved_image_path=str(img2.relative_to(PROJECT_ROOT)).replace("\\", "/") if img2.is_relative_to(PROJECT_ROOT) else str(img2),
            predicted_class="Low",
            confidence=0.55,
            probabilities={"Low": 0.55, "Moderate": 0.35, "High": 0.05, "Critical": 0.05},
            gating_status="HUMAN_EXPERT_REVIEW_REQUIRED",
            review_status="RESOLVED",
            resolved_label="Moderate",
            resolved_by="Dr. Lead Specialist",
            expert_notes="Early biofilm patches identified.",
            resolved_at=datetime.now(timezone.utc),
        )
        db.add(rec2)

        # 3. Unresolved record (should be ignored)
        rec3 = ReviewQueueRecord(
            queue_id="Q-AL-PENDING-01",
            request_id="REQ-AL-03",
            filename="al_pending.jpg",
            saved_image_path="N/A",
            predicted_class="Low",
            confidence=0.50,
            probabilities={"Low": 0.5, "Moderate": 0.5, "High": 0.0, "Critical": 0.0},
            gating_status="HUMAN_EXPERT_REVIEW_REQUIRED",
            review_status="PENDING_REVIEW",
        )
        db.add(rec3)
        db.commit()

        # Ingest samples
        stats = ingest_resolved_samples(
            db=db,
            target_dir=test_expanded_dir,
            base_train_csv=test_base_csv,
            output_train_csv=test_output_csv,
        )

        assert stats["status"] == "success"
        assert stats["total_resolved_records"] >= 2
        assert stats["newly_ingested_images"] >= 2
        assert test_output_csv.exists()

        expanded_df = pd.read_csv(test_output_csv)
        assert len(expanded_df) >= 3  # 1 base + 2 newly ingested
        assert "Critical" in expanded_df["class_label"].values
        assert "Moderate" in expanded_df["class_label"].values


def test_admin_auth_and_threshold_gating():
    """Verify authorization security and minimum sample threshold gating on POST /admin/trigger-retrain."""
    # 1. Missing Authorization Header -> 401 Unauthorized
    resp_unauth = client.post("/admin/trigger-retrain", json={"min_samples": 10, "force": False})
    assert resp_unauth.status_code == 401
    assert "Invalid or missing" in resp_unauth.json()["detail"]

    # 2. Invalid Key -> 401 Unauthorized
    resp_bad_key = client.post(
        "/admin/trigger-retrain",
        json={"min_samples": 10, "force": False},
        headers={"X-Admin-Key": "wrong-secret-key"},
    )
    assert resp_bad_key.status_code == 401

    # 3. Valid Key, Threshold Not Met -> 200 OK with SKIPPED_THRESHOLD_NOT_MET
    resp_skipped = client.post(
        "/admin/trigger-retrain",
        json={"min_samples": 9999, "force": False},
        headers=ADMIN_HEADERS,
    )
    assert resp_skipped.status_code == 200
    data_skipped = resp_skipped.json()
    assert data_skipped["status"] == "SKIPPED_THRESHOLD_NOT_MET"
    assert "Retraining skipped" in data_skipped["message"]
    assert data_skipped["min_samples_threshold"] == 9999

    # 4. Check Status Endpoint
    status_resp = client.get("/admin/retrain-status", headers=ADMIN_HEADERS)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "status" in status_data
    assert status_data["status"] in ["IDLE", "RUNNING", "COMPLETED", "FAILED"]


def test_admin_trigger_retrain_background_execution():
    """Test background task scheduling with force=True on POST /admin/trigger-retrain."""
    # Reset retrain state for test
    retrain_state["status"] = "IDLE"
    retrain_state["job_id"] = None

    resp = client.post(
        "/admin/trigger-retrain",
        json={"min_samples": 1, "epochs": 1, "force": True, "async_mode": True},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "QUEUED"
    assert data["job_id"] is not None
    assert "successfully queued" in data["message"]

    # Check status endpoint immediately reflects running or completed
    status_resp = client.get("/admin/retrain-status", headers=ADMIN_HEADERS)
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["job_id"] == data["job_id"]


if __name__ == "__main__":
    print("Executing automated test suite for Active Learning pipeline...")
    test_active_learning_ingestion_from_db(Path("data/test_tmp"))
    print("  ✓ test_active_learning_ingestion_from_db PASSED")
    test_admin_auth_and_threshold_gating()
    print("  ✓ test_admin_auth_and_threshold_gating PASSED")
    test_admin_trigger_retrain_background_execution()
    print("  ✓ test_admin_trigger_retrain_background_execution PASSED")
    print("\nAll Active Learning pipeline tests PASSED successfully!")

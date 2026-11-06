#!/usr/bin/env python3
"""
Automated Test Suite for Biofouling Inspections, Deletion, and Defect Diagnostic Engine (XAI)
--------------------------------------------------------------------------------------------
Tests:
1. DefectDiagnosticEngine across all biofouling classes (Low, Moderate, High, Critical).
2. Prediction endpoint returns defect_analysis, saves file to disk, and persists InspectionRecord.
3. GET /api/inspections listing and filtering.
4. GET /api/inspections/{id} detail view with XAI payload.
5. DELETE /api/inspections/{id} full-stack removal (database record & disk file).
6. 404 error handling on non-existent inspection retrieval and deletion.
"""

import asyncio
import io
import os
import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from starlette.datastructures import UploadFile
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.services.defect_engine import DefectDiagnosticEngine, get_defect_engine
import app
from app import (
    predict_biofouling_frame,
    list_inspections,
    get_inspection_detail,
    delete_inspection,
    UPLOAD_DIR,
)
from database import SessionLocal, init_db, InspectionRecord


class TestDefectDiagnosticEngine(unittest.TestCase):
    """Test DefectDiagnosticEngine explainability logic."""

    def setUp(self):
        self.engine = get_defect_engine()

    def test_all_classes_generate_valid_diagnostics(self):
        classes = ["Critical", "High", "Moderate", "Low"]
        for cls in classes:
            analysis = self.engine.analyze(predicted_class=cls, confidence=0.88, entropy=0.45)
            self.assertIn("primary_issue", analysis)
            self.assertIn("reasoning", analysis)
            self.assertIn("action_required", analysis)
            self.assertTrue(len(analysis["primary_issue"]) > 5)
            self.assertTrue(len(analysis["reasoning"]) > 20)
            self.assertTrue(len(analysis["action_required"]) > 10)
            self.assertEqual(analysis["predicted_class"], cls)

    def test_high_uncertainty_alert(self):
        analysis = self.engine.analyze(predicted_class="Moderate", confidence=0.55, entropy=1.45)
        self.assertIn("Uncertainty Alert", analysis["reasoning"])
        self.assertIn("Human verification", analysis["action_required"])


class TestInspectionEndToEnd(unittest.TestCase):
    """Direct integration tests for biofouling prediction, dynamic listing, and deletion."""

    @classmethod
    def setUpClass(cls):
        init_db()
        app.ensure_initialized()
        cls.db = SessionLocal()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _create_dummy_image_upload(self, filename="test_inspection_sample.png"):
        img_array = (np.random.rand(380, 380, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return UploadFile(filename=filename, file=buf)

    def test_01_predict_returns_defect_analysis_and_persists(self):
        upload_file = self._create_dummy_image_upload("test_hull_section_01.png")
        result = asyncio.run(
            predict_biofouling_frame(
                file=upload_file,
                threshold=0.75,
                db=self.db,
            )
        )

        # Check prediction structure
        self.assertIsNotNone(result.predicted_class)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertIsNotNone(result.inspection_id)

        # Check Defect Analysis explainability fields
        defect = result.defect_analysis
        self.assertIsNotNone(defect)
        self.assertTrue(hasattr(defect, "primary_issue"))
        self.assertTrue(hasattr(defect, "reasoning"))
        self.assertTrue(hasattr(defect, "action_required"))
        self.assertGreater(len(defect.primary_issue), 5)
        self.assertGreater(len(defect.reasoning), 20)
        self.assertGreater(len(defect.action_required), 10)

        # Verify disk persistence in UPLOAD_DIR
        inspection_id = result.inspection_id
        matching_files = list(UPLOAD_DIR.glob(f"*{inspection_id}*"))
        self.assertTrue(len(matching_files) > 0, f"Uploaded file not found in {UPLOAD_DIR}")

        # Store ID and file for subsequent test steps
        TestInspectionEndToEnd.created_inspection_id = inspection_id
        TestInspectionEndToEnd.created_file_path = matching_files[0]

    def test_02_list_inspections(self):
        response = asyncio.run(
            list_inspections(
                severity=None,
                status_filter=None,
                search=None,
                limit=50,
                offset=0,
                db=self.db,
            )
        )
        self.assertGreater(response.total, 0)
        self.assertGreater(len(response.inspections), 0)

        # Confirm the created inspection is in the list
        matched = [i for i in response.inspections if i.inspection_id == self.created_inspection_id]
        self.assertEqual(len(matched), 1)
        item = matched[0]
        self.assertIsNotNone(item.defect_analysis)
        self.assertIn("primary_issue", item.defect_analysis)

    def test_03_get_inspection_detail(self):
        item = asyncio.run(
            get_inspection_detail(
                inspection_id=self.created_inspection_id,
                db=self.db,
            )
        )
        self.assertEqual(item.inspection_id, self.created_inspection_id)
        self.assertIsNotNone(item.defect_analysis)
        self.assertIn("reasoning", item.defect_analysis)

    def test_04_get_nonexistent_inspection_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                get_inspection_detail(
                    inspection_id="nonexistent_xyz_9999",
                    db=self.db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_05_delete_inspection_full_stack(self):
        file_path = self.created_file_path
        self.assertTrue(file_path.exists(), "File must exist on disk before deletion")

        # Call delete endpoint
        del_resp = asyncio.run(
            delete_inspection(
                inspection_id=self.created_inspection_id,
                db=self.db,
            )
        )
        self.assertTrue(del_resp.success)
        self.assertEqual(del_resp.inspection_id, self.created_inspection_id)

        # Verify DB record is gone
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                get_inspection_detail(
                    inspection_id=self.created_inspection_id,
                    db=self.db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)

        # Verify disk file is deleted
        self.assertFalse(file_path.exists(), "File must be removed from disk upon deletion")

    def test_06_delete_nonexistent_inspection_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                delete_inspection(
                    inspection_id="nonexistent_xyz_9999",
                    db=self.db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

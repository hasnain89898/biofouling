#!/usr/bin/env python3
"""
Enterprise FastAPI REST API Backend for Industrial Biofouling Assessment (v4)
-----------------------------------------------------------------------------
Features:
1. Ultra-low latency ONNX Runtime inference for cooling tower / pipe interior frame feeds.
2. 380x380 multi-scale image preprocessing with strict diagnostic color preservation.
3. Operational Confidence Safety Gating (75% threshold) with normalized Shannon entropy.
4. Automated Human-in-the-Loop (HITL) review queue logging (human_review_queue.json).
5. Comprehensive Pydantic data schemas, startup self-checks, and queue management endpoints.

Author: Principal AI Research Engineer
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import InspectionRecord, ReviewQueueRecord, SessionLocal, get_db, init_db
from backend.services.defect_engine import DefectDiagnosticEngine, get_defect_engine

try:
    from retrain_model import fine_tune_model
except Exception:
    fine_tune_model = None

try:
    from backend.services.chatbot import BioAssistChatbot, get_chatbot
except Exception:
    BioAssistChatbot = None
    get_chatbot = None

try:
    from backend.services.guardrail import IndustrialDomainGuardrail, get_guardrail, validate_industrial_domain
except Exception:
    IndustrialDomainGuardrail = None
    get_guardrail = None
    validate_industrial_domain = None

try:
    from backend.services.video_processor import IndustrialVideoProcessor
except Exception:
    IndustrialVideoProcessor = None

try:
    from backend.services.inference import ONNXInferenceTTAWrapper
except Exception:
    ONNXInferenceTTAWrapper = None

# Administrative API Key Configuration
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "biofouling-admin-2026")

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Standard biofouling label definitions
IDX_TO_LABEL = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}
LABEL_TO_IDX = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    from src.dataset import get_val_transforms
except Exception:
    def get_val_transforms(image_size: int = 380):
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingAPI")

# File Paths & Global Config
DEFAULT_ONNX_PATH = PROJECT_ROOT / "biofouling_model.onnx"
ALT_ONNX_PATH = PROJECT_ROOT / "checkpoints" / "biofouling_model.onnx"
REVIEW_QUEUE_FILE = PROJECT_ROOT / "data" / "human_review_queue.json"
REVIEW_QUEUE_IMG_DIR = PROJECT_ROOT / "data" / "review_queue_images"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 380
DEFAULT_CONFIDENCE_THRESHOLD = 0.80

# Global State Container
state: Dict[str, Any] = {
    "ort_session": None,
    "input_name": None,
    "output_name": None,
    "val_transforms": None,
    "start_time": None,
    "model_path": None,
}


# ==============================================================================
# Domain Advisory Knowledge Base & Customer Plain-English Copy (UX Layer)
# ==============================================================================

MAINTENANCE_ADVICE: Dict[str, Dict[str, str]] = {
    "Low": {
        "status_title": "Nominal / Baseline Surface Condition",
        "severity_grade": "Low",
        "description": "Surface shows Low (minimal) biofouling. Pristine pipe substrate and heat-exchange surfaces with optimal thermal transfer.",
        "action": "No immediate maintenance required. Maintain standard biocide dosage and schedule routine visual check in 30 days.",
    },
    "Moderate": {
        "status_title": "Early Biofilm Colonization Detected",
        "severity_grade": "Moderate",
        "description": "Surface shows Moderate biofouling (early biofilm formation or localized microalgal patchiness).",
        "action": "Schedule proactive maintenance cleaning within 14 days and inspect biocide feeder dosing to prevent colonization expansion.",
    },
    "High": {
        "status_title": "Heavy Biofilm Matrix Accumulation Alert",
        "severity_grade": "High",
        "description": "Surface shows High biofouling (thick microbial slime matrix with substantial surface coverage).",
        "action": "Schedule mechanical cleaning or targeted pressure wash within 48-72 hours and apply biopenetrant treatment to restore flow efficiency.",
    },
    "Critical": {
        "status_title": "Severe Biofouling / Flow Obstruction Danger",
        "severity_grade": "Critical",
        "description": "Surface shows Critical biofouling with extensive biological accumulation, sloughing biomass, or localized flow restriction.",
        "action": "Immediate operational intervention required within 24 hours. Isolate affected channel for chemical shock treatment and high-intensity mechanical flush to prevent thermal degradation and microbially influenced corrosion (MIC).",
    },
}

HUMAN_REVIEW_MESSAGE = (
    "Scan received. The image contains mixed or complex sections requiring a brief manual "
    "review by our inspection specialist to ensure 100% safety accuracy. You will be notified shortly."
)

HUMAN_REVIEW_ACTION = (
    "Automated dispatch paused pending specialist review. No immediate field intervention required "
    "until safety verification is finalized."
)


def generate_customer_message(
    pred_class: str,
    gating_status: str,
    confidence: float,
    applied_threshold: float,
) -> Tuple[str, str, str]:
    """
    Generate clean, plain-English UX customer messages, headlines, and action recommendations.

    Returns:
        (headline, customer_message, action_recommendation)
    """
    if gating_status == "AUTOMATED_DECISION":
        advice = MAINTENANCE_ADVICE.get(pred_class, MAINTENANCE_ADVICE["Low"])
        headline = f"Inspection Complete: {advice['status_title']}"
        customer_message = f"Inspection Complete: Surface shows {pred_class} biofouling. Recommended action: {advice['action']}"
        action_recommendation = advice["action"]
    else:
        headline = "Verification In Progress: Specialist Review Queued"
        customer_message = HUMAN_REVIEW_MESSAGE
        action_recommendation = HUMAN_REVIEW_ACTION

    return headline, customer_message, action_recommendation


def interpret_entropy(entropy: float) -> str:
    """Provide human-readable engineering uncertainty grade from normalized Shannon entropy."""
    if entropy < 0.25:
        return "Minimal Uncertainty (High Classification Certainty)"
    elif entropy < 0.50:
        return "Low Uncertainty (Distinct Dominant Features)"
    elif entropy < 0.75:
        return "Moderate Uncertainty (Minor Multiclass Ambiguity)"
    else:
        return "Elevated Uncertainty (Diffuse Multi-Class Boundary / Complex Surface)"


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class ClassProbabilityDistribution(BaseModel):
    Low: float = Field(..., description="Probability of pristine/clean fill or pipe (0.0 to 1.0)")
    Moderate: float = Field(..., description="Probability of early biofilm or algae patch (0.0 to 1.0)")
    High: float = Field(..., description="Probability of thick biofilm matrix or slime (0.0 to 1.0)")
    Critical: float = Field(..., description="Probability of mature sloughing biofilm / flow blockage (0.0 to 1.0)")


class TechnicalTelemetry(BaseModel):
    """
    Dedicated engineering telemetry & diagnostic metadata object.
    Enables clean separation between customer UX and deep ML observability.
    """
    model_version: str = Field(default="4.0.0", description="Model iteration and weights build")
    model_backbone: str = Field(default="EfficientNet-B3 (380x380)", description="Feature extractor architecture")
    inference_engine: str = Field(default="ONNX Runtime", description="Inference runtime provider")
    raw_confidence: float = Field(..., description="Unformatted maximum softmax probability [0.0 - 1.0]")
    confidence_percentage: str = Field(..., description="Percentage formatted confidence string")
    normalized_entropy: float = Field(..., description="Normalized Shannon entropy (0.0=certain, 1.0=maximum ambiguity)")
    uncertainty_level: str = Field(..., description="Plain-English uncertainty grade (e.g. Minimal, Low, Moderate, Elevated)")
    class_probabilities: ClassProbabilityDistribution = Field(..., description="Softmax probabilities for each class")
    predicted_idx: int = Field(..., description="0-indexed class index")
    top1_class: str = Field(..., description="Primary predicted class")
    top2_class: str = Field(..., description="Secondary runner-up class for boundary ambiguity detection")
    operational_threshold: float = Field(..., description="Confidence threshold used for HITL gating")
    latency_ms: float = Field(..., description="Inference computation time in milliseconds")
    review_reason: str = Field(..., description="Technical rationale behind automated vs escalated verdict")
    tta_applied: bool = Field(default=True, description="Whether Test-Time Augmentation was applied")
    tta_consensus_score: Optional[float] = Field(None, description="Averaged confidence score after TTA")
    calibration_metadata: Optional[Dict[str, Any]] = Field(None, description="Detailed metadata for all TTA variants")


class DefectAnalysis(BaseModel):
    """
    Explainable AI (XAI) Defect Diagnostic Payload mapping computer vision
    classifications to actionable biological & operational engineering intelligence.
    """
    primary_issue: str = Field(..., description="Specific biological defect or operational fault identification")
    reasoning: str = Field(..., description="Detailed Explainable AI morphological and corrosion risk rationale")
    action_required: str = Field(..., description="Operational maintenance and biocide shock treatment protocol")
    corrosion_risk: Optional[str] = Field(None, description="Assessed microbially influenced corrosion (MIC) risk")
    morphology_summary: Optional[str] = Field(None, description="Key structural morphology indicator")
    severity: Optional[str] = Field(None, description="Class severity")
    confidence_pct: Optional[str] = Field(None, description="Confidence percentage")
    entropy_score: Optional[float] = Field(None, description="Normalized Shannon entropy score")


class PredictionResponse(BaseModel):
    """
    Customer-First & Engineer-Ready Inspection Payload.
    Top-level fields are immediately actionable for field inspectors & customers,
    while technical telemetry is encapsulated in both backward-compatible fields
    and a dedicated 'telemetry' metadata object.
    """
    # --------------------------------------------------------------------------
    # 1. Customer & Field Inspector Presentation Layer (Clean, Non-Technical)
    # --------------------------------------------------------------------------
    request_id: str = Field(..., description="Unique UUID for inspection tracking")
    inspection_id: Optional[str] = Field(None, description="Database record tracking ID")
    filename: str = Field(..., description="Name of uploaded visual inspection frame")
    severity_assessment: str = Field(..., description="Assessed biofouling severity level: Low, Moderate, High, Critical")
    headline: str = Field(..., description="High-level non-technical summary headline")
    customer_message: str = Field(..., description="Plain-English customer message with actionable maintenance advice")
    action_recommendation: str = Field(..., description="Recommended operational maintenance response")
    gating_status: str = Field(..., description="'AUTOMATED_DECISION' or 'HUMAN_EXPERT_REVIEW_REQUIRED'")
    human_review_required: bool = Field(..., description="True if sample was escalated to human review queue")

    # --------------------------------------------------------------------------
    # 2. Explainable AI (XAI) Defect Diagnostic Object
    # --------------------------------------------------------------------------
    defect_analysis: DefectAnalysis = Field(..., description="Morphological Defect Diagnostic Engine explainability analysis")

    # --------------------------------------------------------------------------
    # 3. Backward-Compatible Top-Level Engineering Metrics
    # --------------------------------------------------------------------------
    predicted_class: str = Field(..., description="Predicted severity class [Low, Moderate, High, Critical]")
    predicted_idx: int = Field(..., description="Zero-indexed class label (0=Low, 1=Moderate, 2=High, 3=Critical)")
    confidence: float = Field(..., description="Maximum predicted class probability (0.0 to 1.0)")
    confidence_percentage: str = Field(..., description="Formatted confidence string")
    entropy: float = Field(..., description="Normalized Shannon entropy (0.0=certain, 1.0=maximum ambiguity)")
    probabilities: ClassProbabilityDistribution = Field(..., description="Full 4-class softmax probability vector")
    review_reason: str = Field(..., description="Technical rationale explaining automated or escalated status")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of inference")

    # --------------------------------------------------------------------------
    # 4. Dedicated Telemetry & Diagnostic Metadata Container
    # --------------------------------------------------------------------------
    telemetry: TechnicalTelemetry = Field(..., description="Encapsulated diagnostic & telemetry metadata for engineers")


class VideoPredictionResponse(BaseModel):
    max_severity_detected: str = Field(..., description="Highest biofouling severity detected in the video")
    critical_alerts: List[Dict[str, Any]] = Field(..., description="List of timestamps where High/Critical biofouling was detected")
    frame_by_frame_results: List[Dict[str, Any]] = Field(..., description="Comprehensive timeline report for all extracted frames")
    total_frames_processed: int = Field(..., description="Total number of frames extracted and processed")
    video_fps: float = Field(..., description="Detected frames per second of the source video")
    defect_analysis: Optional[DefectAnalysis] = Field(None, description="Overall worst-case Defect Diagnostic analysis")


class InspectionItem(BaseModel):
    id: int = Field(..., description="Database record primary key")
    inspection_id: str = Field(..., description="Unique inspection identifier")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Relative storage path on server")
    media_type: str = Field(default="image", description="Media type ('image' or 'video')")
    predicted_class: str = Field(..., description="Predicted biofouling severity")
    confidence: float = Field(..., description="Model confidence score")
    confidence_percentage: str = Field(..., description="Formatted confidence")
    entropy: float = Field(..., description="Normalized Shannon entropy")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    gating_status: str = Field(..., description="Safety gating status")
    human_review_required: bool = Field(..., description="Whether flagged for review")
    defect_analysis: Dict[str, Any] = Field(..., description="Explainable AI defect diagnostic payload")
    primary_issue: str = Field(..., description="Primary biological issue or defect")
    reasoning: str = Field(..., description="Morphological reasoning")
    action_required: str = Field(..., description="Action recommendation")
    latency_ms: float = Field(..., description="Inference latency")
    created_at: str = Field(..., description="Inspection timestamp")


class InspectionListResponse(BaseModel):
    total: int = Field(..., description="Total count of inspection records")
    inspections: List[InspectionItem] = Field(..., description="List of dynamic inspection records")


class DeleteInspectionResponse(BaseModel):
    status: str = Field("success", description="Deletion status ('success')")
    success: bool = Field(True, description="Boolean deletion success indicator")
    message: str = Field(..., description="Action confirmation message")
    deleted_id: str = Field(..., description="Deleted inspection identifier")
    inspection_id: Optional[str] = Field(None, description="Deleted inspection identifier alias")



class QueueItem(BaseModel):
    id: Optional[int] = Field(None, description="Primary key database record ID")
    queue_id: str = Field(..., description="Unique ID for queue entry")
    request_id: str = Field(..., description="Original inference request ID")
    filename: str = Field(..., description="Original frame filename")
    saved_image_path: str = Field(..., description="Path to archived frame for human inspection")
    predicted_class: str = Field(..., description="Top model prediction")
    confidence: float = Field(..., description="Model confidence score")
    entropy: float = Field(..., description="Shannon entropy score")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution")
    gating_status: Optional[str] = Field("HUMAN_EXPERT_REVIEW_REQUIRED", description="Gating verdict")
    review_status: Optional[str] = Field("PENDING_REVIEW", description="Review state [PENDING_REVIEW, RESOLVED]")
    review_reason: str = Field(..., description="Escalation rationale")
    customer_message: Optional[str] = Field(None, description="Plain-English customer status message")
    action_recommendation: Optional[str] = Field(None, description="Current field guidance")
    status: str = Field(default="PENDING_REVIEW", description="Status [PENDING_REVIEW, RESOLVED]")
    created_at: str = Field(..., description="Timestamp of escalation")
    resolved_by: Optional[str] = Field(None, description="Expert analyst identifier")
    resolved_label: Optional[str] = Field(None, description="Confirmed ground truth label by expert")
    resolved_at: Optional[str] = Field(None, description="Timestamp of resolution")
    expert_notes: Optional[str] = Field(None, description="Inspection notes from human expert")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded'")
    service_name: str = Field(..., description="Service identifier")
    version: str = Field(..., description="API Version")
    model_loaded: bool = Field(..., description="Whether ONNX model is primed in memory")
    model_path: str = Field(..., description="Filesystem location of active ONNX model")
    execution_providers: List[str] = Field(..., description="Active ONNX Runtime hardware providers")
    image_resolution: str = Field(..., description="Configured input resolution (e.g. 380x380)")
    operational_confidence_threshold: float = Field(..., description="Threshold triggering human expert escalation")
    review_queue_pending_count: int = Field(..., description="Number of frames currently awaiting human review")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ReviewQueueSummaryResponse(BaseModel):
    total_escalated_items: int
    pending_review_count: int
    resolved_count: int
    queue_items: List[QueueItem]


class ResolveQueueItemRequest(BaseModel):
    resolved_label: str = Field(..., description="Confirmed severity class: 'Low', 'Moderate', 'High', or 'Critical'")
    resolved_by: str = Field(..., description="Name / ID of the human expert inspector")
    notes: Optional[str] = Field(None, description="Optional engineering inspection notes")


class RetrainTriggerRequest(BaseModel):
    min_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum number of verified expert-reviewed samples required to trigger fine-tuning",
    )
    epochs: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of incremental fine-tuning epochs",
    )
    batch_size: int = Field(default=16, ge=1, le=128, description="Fine-tuning batch size")
    learning_rate: float = Field(default=3e-4, gt=0.0, description="Optimizer learning rate for classification head")
    force: bool = Field(
        default=False,
        description="Force fine-tuning execution even if resolved count is below min_samples threshold",
    )
    async_mode: bool = Field(
        default=True,
        description="Run fine-tuning as a non-blocking background job",
    )


class RetrainTriggerResponse(BaseModel):
    status: str = Field(..., description="Execution status ('QUEUED', 'COMPLETED', 'SKIPPED_THRESHOLD_NOT_MET', 'ALREADY_RUNNING', 'FAILED')")
    message: str = Field(..., description="Human-readable status summary")
    resolved_samples_count: int = Field(..., description="Current count of resolved review queue records")
    min_samples_threshold: int = Field(..., description="Configured minimum sample threshold")
    job_id: Optional[str] = Field(None, description="Unique identifier for the background retraining job")
    result: Optional[Dict[str, Any]] = Field(None, description="Training metrics if executed synchronously")


class RetrainStatusResponse(BaseModel):
    status: str = Field(..., description="Current retraining job state ('IDLE', 'RUNNING', 'COMPLETED', 'FAILED')")
    job_id: Optional[str] = Field(None, description="Active or last job identifier")
    started_at: Optional[str] = Field(None, description="Job commencement timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    error: Optional[str] = Field(None, description="Error message if retraining failed")
    last_result: Optional[Dict[str, Any]] = Field(None, description="Metrics from the most recent retraining pass")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or inspection inquiry")
    session_id: Optional[str] = Field(None, description="Optional session tracking identifier")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior multi-turn conversation history")


class ChatResponse(BaseModel):
    response: str = Field(..., description="BioAssist AI generated response")
    model_used: str = Field(..., description="Model and LoRA adapter identifier")
    status: str = Field(default="success", description="Execution status")


# Global Active Learning Retraining State
retrain_state: Dict[str, Any] = {
    "status": "IDLE",
    "job_id": None,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "last_result": None,
}


def verify_admin_key(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key", description="Admin API Key header"),
    admin_key: Optional[str] = Query(None, description="Admin API Key query parameter"),
) -> str:
    """Verify administrator authentication token."""
    provided_key = x_admin_key or admin_key
    expected_key = os.getenv("ADMIN_API_KEY", "biofouling-admin-2026")
    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing administrative authorization key.",
        )
    return provided_key


# ==============================================================================
# Helper Functions: Queue Persistence & Image Processing
# ==============================================================================

def load_review_queue() -> List[Dict[str, Any]]:
    """Load items from the review queue database with graceful fallback."""
    try:
        with SessionLocal() as db:
            records = db.query(ReviewQueueRecord).order_by(ReviewQueueRecord.created_at.asc()).all()
            return [r.to_dict() for r in records]
    except Exception as e:
        logger.error(f"Error reading review queue from database: {e}")
        if REVIEW_QUEUE_FILE.exists():
            try:
                with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []


def save_review_queue(queue_items: List[Dict[str, Any]]) -> None:
    """Safely persist review queue items (deprecated JSON bridge for tests/legacy)."""
    try:
        REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = REVIEW_QUEUE_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(queue_items, f, indent=2)
        temp_file.replace(REVIEW_QUEUE_FILE)
    except Exception as e:
        logger.debug(f"JSON bridge save omitted: {e}")


def append_to_review_queue(item: QueueItem) -> None:
    """Append a newly escalated item to the database queue."""
    try:
        with SessionLocal() as db:
            rec = ReviewQueueRecord(
                queue_id=item.queue_id,
                request_id=item.request_id,
                filename=item.filename,
                saved_image_path=item.saved_image_path,
                predicted_class=item.predicted_class,
                confidence=item.confidence,
                entropy=item.entropy,
                probabilities=item.probabilities,
                gating_status=item.gating_status or "HUMAN_EXPERT_REVIEW_REQUIRED",
                review_status=item.status or "PENDING_REVIEW",
                review_reason=item.review_reason,
                customer_message=item.customer_message,
                action_recommendation=item.action_recommendation,
                expert_notes=item.expert_notes,
            )
            db.add(rec)
            db.commit()
    except Exception as e:
        logger.error(f"Error appending to review queue DB: {e}")
    logger.warning(
        f"[HITL ESCALATION] Frame '{item.filename}' (Conf: {item.confidence*100:.1f}%) "
        f"routed to human review queue (ID: {item.queue_id})"
    )


def calculate_entropy(probs: np.ndarray) -> float:
    """Compute normalized Shannon entropy in [0.0, 1.0]."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))  # log2(4) = 2.0
    return float(entropy / max_entropy)


def ensure_initialized(force_reload: bool = False) -> None:
    """Ensure ONNX session, database schema, and state are initialized."""
    init_db()
    if not force_reload and state["ort_session"] is not None:
        return

    state["start_time"] = state["start_time"] or time.time()
    REVIEW_QUEUE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Locate ONNX model
    onnx_path = DEFAULT_ONNX_PATH if DEFAULT_ONNX_PATH.exists() else ALT_ONNX_PATH
    if not onnx_path.exists():
        logger.critical(f"FATAL: ONNX model not found at {DEFAULT_ONNX_PATH} or {ALT_ONNX_PATH}!")
        raise FileNotFoundError(f"ONNX model missing: {DEFAULT_ONNX_PATH}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    state["ort_session"] = session
    state["input_name"] = session.get_inputs()[0].name
    state["output_name"] = session.get_outputs()[0].name
    state["model_path"] = str(onnx_path)
    state["val_transforms"] = get_val_transforms(image_size=IMAGE_SIZE)

    # Warm-up inference
    warmup_tensor = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    _ = session.run([state["output_name"]], {state["input_name"]: warmup_tensor})
    logger.info(
        f"Biofouling ONNX Engine initialized: {onnx_path.name} "
        f"| Providers: {session.get_providers()} | Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )


def run_background_retraining(
    job_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Execute active learning fine-tuning in background and reload ONNX inference engine."""
    logger.info(f"[ACTIVE LEARNING] Commencing background retraining job: {job_id}")
    try:
        result = fine_tune_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        retrain_state["status"] = "COMPLETED"
        retrain_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        retrain_state["last_result"] = result
        retrain_state["error"] = None
        # Hot-reload inference engine with new weights
        ensure_initialized(force_reload=True)
        logger.info(f"[ACTIVE LEARNING] Job {job_id} successfully completed. Model engine reloaded.")
    except Exception as e:
        logger.error(f"[ACTIVE LEARNING] Job {job_id} failed: {e}", exc_info=True)
        retrain_state["status"] = "FAILED"
        retrain_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        retrain_state["error"] = str(e)


# ==============================================================================
# FastAPI Lifespan (Startup & Shutdown)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database schema, ONNX Runtime inference session, and validation transforms."""
    init_db()
    ensure_initialized()
    yield
    logger.info("Shutting down Biofouling Assessment API service...")


# ==============================================================================
# FastAPI Application Declaration
# ==============================================================================

app = FastAPI(
    title="Industrial Biofouling Assessment REST API",
    description=(
        "Enterprise-grade computer vision backend for 4-class industrial biofouling assessment "
        "(Low, Moderate, High, Critical) across cooling towers and industrial pipes. "
        "Powered by EfficientNet-B3 (380x380) with dynamic-batch ONNX Runtime and automated "
        "Human-in-the-Loop (HITL) safety gating."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# Enable CORS for enterprise dashboards and SCADA interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Asset Mounting
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.api_route("/", methods=["GET", "HEAD"], tags=["General"])
async def root(request: Request):
    """Serve modern single-page visual inspection interface matching Figma design."""
    accept = request.headers.get("accept", "")
    index_file = STATIC_DIR / "index.html"
    if "text/html" not in accept and any(mime in accept for mime in ["application/json", "*/*"]):
        return {
            "service": "Industrial Biofouling Assessment REST API",
            "version": "4.0.0",
            "status": "OPERATIONAL",
            "engine": "ONNX Runtime (EfficientNet-B3 380x380)",
            "operational_safety_threshold": f"{DEFAULT_CONFIDENCE_THRESHOLD * 100:.1f}%",
            "endpoints": {
                "health": "/health",
                "predict": "POST /predict",
                "chat": "POST /api/chat",
                "review_queue": "GET /review-queue",
                "trigger_retrain": "POST /admin/trigger-retrain",
                "retrain_status": "GET /admin/retrain-status",
                "docs": "/docs",
            },
        }
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "Industrial Biofouling Assessment REST API",
        "version": "4.0.0",
        "status": "OPERATIONAL",
        "engine": "ONNX Runtime (EfficientNet-B3 380x380)",
        "operational_safety_threshold": f"{DEFAULT_CONFIDENCE_THRESHOLD * 100:.1f}%",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "chat": "POST /api/chat",
            "review_queue": "GET /review-queue",
            "trigger_retrain": "POST /admin/trigger-retrain",
            "retrain_status": "GET /admin/retrain-status",
            "docs": "/docs",
        },
    }


@app.get("/api/info", tags=["General"])
async def api_info():
    """Programmatic service overview and endpoints manifest."""
    return {
        "service": "Industrial Biofouling Assessment REST API",
        "version": "4.0.0",
        "status": "OPERATIONAL",
        "engine": "ONNX Runtime (EfficientNet-B3 380x380)",
        "operational_safety_threshold": f"{DEFAULT_CONFIDENCE_THRESHOLD * 100:.1f}%",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "chat": "POST /api/chat",
            "review_queue": "GET /review-queue",
            "trigger_retrain": "POST /admin/trigger-retrain",
            "retrain_status": "GET /admin/retrain-status",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Startup self-check and operational diagnostics."""
    ensure_initialized()
    session = state.get("ort_session")
    model_loaded = session is not None

    try:
        pending_count = db.query(ReviewQueueRecord).filter(
            ReviewQueueRecord.review_status == "PENDING_REVIEW"
        ).count()
    except Exception as e:
        logger.error(f"Database error querying pending review count: {e}")
        pending_count = 0

    uptime = time.time() - (state.get("start_time") or time.time())

    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        service_name="biofouling-assessment-api",
        version="4.0.0",
        model_loaded=model_loaded,
        model_path=state.get("model_path") or "N/A",
        execution_providers=session.get_providers() if session else [],
        image_resolution=f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        operational_confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        review_queue_pending_count=pending_count,
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict Biofouling Severity with Automated HITL Gating",
)
async def predict_biofouling_frame(
    file: UploadFile = File(..., description="Visual inspection image file (JPEG, PNG, WEBP, BMP)"),
    threshold: Optional[float] = Query(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Custom operational confidence gating threshold (default: 0.75)",
    ),
    db: Session = Depends(get_db),
):
    """
    Accepts an industrial visual inspection frame, performs 380x380 normalization,
    runs ONNX inference, and applies Confidence-Based Human-in-the-Loop Safety Gating.
    """
    ensure_initialized()
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    applied_threshold = threshold if threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD

    session: ort.InferenceSession = state.get("ort_session")
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ONNX Runtime inference engine is not ready.",
        )

    # 1. Validate and Read Image Bytes
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Supported: {sorted(allowed_extensions)}",
        )

    try:
        image_bytes = await file.read()
        pil_img = Image.open(pd.io.common.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Failed to decode uploaded image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image payload: {str(e)}",
        )

    # 2. Industrial Domain Guardrail & Out-of-Distribution (OOD) Filter
    if get_guardrail is not None:
        try:
            guardrail = get_guardrail()
            gr_result = guardrail.validate(img_np)
            if not gr_result.is_valid:
                logger.warning(
                    f"[GUARDRAIL REJECTION] File '{file.filename}' rejected (Score: {gr_result.domain_score:.3f}). "
                    f"Reason: {gr_result.rejection_reason}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "status": "error",
                        "error_code": "INVALID_DOMAIN_INPUT",
                        "message": (
                            gr_result.rejection_reason
                            or "Input image is irrelevant to industrial biofouling inspection. Please provide a valid cooling tower, pipe, or heat exchanger inspection frame."
                        ),
                        "domain_score": gr_result.domain_score,
                        "signals": gr_result.signals,
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.info(f"Guardrail check bypassed: {e}")

    # 3. & 4. ONNX Runtime Inference Pass
    val_transforms = state.get("val_transforms")
    consensus_probs = None
    entropy = 0.0
    individual_probs = []

    if ONNXInferenceTTAWrapper is not None and val_transforms is not None:
        try:
            tta_wrapper = ONNXInferenceTTAWrapper(
                session=session,
                val_transforms=val_transforms,
                input_name=state["input_name"],
                output_name=state["output_name"],
            )
            consensus_probs, entropy, individual_probs = tta_wrapper.predict_tta(img_np)
        except Exception as e:
            logger.warning(f"TTA inference fallback to direct ONNX: {e}")

    if consensus_probs is None:
        # High-performance native ONNX Runtime inference pass
        pil_resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        input_batch = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        logits_np = session.run([state["output_name"]], {state["input_name"]: input_batch})[0]
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs_np = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
        entropy = calculate_entropy(probs_np)
        individual_probs = [probs_np]
    else:
        probs_np = consensus_probs

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Extract Classification & Uncertainty Metrics
    pred_idx = int(np.argmax(probs_np))
    pred_class = IDX_TO_LABEL[pred_idx]
    confidence = float(probs_np[pred_idx])

    prob_dist = ClassProbabilityDistribution(
        Low=round(float(probs_np[0]), 4),
        Moderate=round(float(probs_np[1]), 4),
        High=round(float(probs_np[2]), 4),
        Critical=round(float(probs_np[3]), 4),
    )

    # Sorted classes for diagnostic explanation
    sorted_idx = np.argsort(probs_np)[::-1]
    top1_cls, top2_cls = IDX_TO_LABEL[sorted_idx[0]], IDX_TO_LABEL[sorted_idx[1]]
    top1_p, top2_p = float(probs_np[sorted_idx[0]]), float(probs_np[sorted_idx[1]])

    # 5. Operational Safety Gating Logic (HITL)
    if confidence >= applied_threshold:
        gating_status = "AUTOMATED_DECISION"
        human_review_required = False
        review_reason = (
            f"Confidence ({confidence * 100:.1f}%) meets or exceeds operational "
            f"safety threshold ({applied_threshold * 100:.1f}%). Safe for automated execution."
        )
    else:
        gating_status = "HUMAN_EXPERT_REVIEW_REQUIRED"
        human_review_required = True
        review_reason = (
            f"Confidence ({confidence * 100:.1f}%) is below operational safety threshold "
            f"({applied_threshold * 100:.1f}%). Ambiguous boundary between '{top1_cls}' ({top1_p * 100:.1f}%) "
            f"and '{top2_cls}' ({top2_p * 100:.1f}%). Automated execution halted."
        )

    # 6. Generate Explainable AI (XAI) Defect Diagnostic Intelligence
    defect_engine = get_defect_engine()
    defect_data = defect_engine.analyze_defect(
        predicted_class=pred_class,
        confidence=confidence,
        entropy=entropy,
        probabilities=prob_dist.model_dump(),
        top2_class=top2_cls,
    )
    defect_obj = DefectAnalysis(**defect_data)

    # 7. Generate Clean Customer UX Presentation & Action Recommendation
    headline, customer_message, action_recommendation = generate_customer_message(
        pred_class=pred_class,
        gating_status=gating_status,
        confidence=confidence,
        applied_threshold=applied_threshold,
    )

    # Generate unique inspection ID
    inspection_unique_id = f"INSP-{uuid.uuid4().hex[:8].upper()}"

    # Always persist uploaded frame to persistent storage
    upload_filename = f"{inspection_unique_id}_{Path(file.filename).name}"
    saved_upload_path = UPLOAD_DIR / upload_filename
    try:
        pil_img.save(saved_upload_path, format="JPEG", quality=95)
        rel_upload_path = str(saved_upload_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception as e:
        logger.warning(f"Could not archive inspection frame: {e}")
        rel_upload_path = "N/A"

    # Persist permanent InspectionRecord in database
    try:
        insp_record = InspectionRecord(
            inspection_id=inspection_unique_id,
            filename=file.filename,
            file_path=rel_upload_path,
            media_type="image",
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            confidence_percentage=f"{confidence * 100:.1f}%",
            entropy=round(entropy, 4),
            probabilities=prob_dist.model_dump(),
            gating_status=gating_status,
            human_review_required=human_review_required,
            defect_analysis=defect_data,
            latency_ms=round(latency_ms, 2),
        )
        db.add(insp_record)
        db.commit()
        db.refresh(insp_record)
        logger.info(f"Persisted InspectionRecord {inspection_unique_id} to database.")
    except Exception as e:
        logger.error(f"Failed to persist InspectionRecord: {e}")
        db.rollback()

    if human_review_required:
        queue_id = f"Q-{uuid.uuid4().hex[:8].upper()}"
        try:
            db_record = ReviewQueueRecord(
                queue_id=queue_id,
                request_id=request_id,
                filename=file.filename,
                saved_image_path=rel_upload_path,
                predicted_class=pred_class,
                confidence=round(confidence, 4),
                entropy=round(entropy, 4),
                probabilities=prob_dist.model_dump(),
                gating_status=gating_status,
                review_status="PENDING_REVIEW",
                review_reason=review_reason,
                customer_message=customer_message,
                action_recommendation=action_recommendation,
                expert_notes=None,
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            logger.warning(
                f"[HITL ESCALATION] Frame '{file.filename}' (Conf: {confidence*100:.1f}%) "
                f"routed to review queue (Queue ID: {queue_id}, Inspection: {inspection_unique_id})"
            )
        except Exception as e:
            logger.error(f"Failed to persist HITL escalation to database: {e}")
            db.rollback()
            # Fallback legacy append if DB write fails
            queue_item = QueueItem(
                queue_id=queue_id,
                request_id=request_id,
                filename=file.filename,
                saved_image_path=rel_upload_path,
                predicted_class=pred_class,
                confidence=round(confidence, 4),
                entropy=round(entropy, 4),
                probabilities=prob_dist.model_dump(),
                gating_status=gating_status,
                review_status="PENDING_REVIEW",
                review_reason=review_reason,
                customer_message=customer_message,
                action_recommendation=action_recommendation,
                status="PENDING_REVIEW",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            append_to_review_queue(queue_item)

    # 8. Package Dedicated Engineering Telemetry Object
    telemetry_obj = TechnicalTelemetry(
        model_version="4.0.0",
        model_backbone="EfficientNet-B3 (380x380)",
        inference_engine="ONNX Runtime",
        raw_confidence=round(confidence, 4),
        confidence_percentage=f"{confidence * 100:.2f}%",
        normalized_entropy=round(entropy, 4),
        uncertainty_level=interpret_entropy(entropy),
        class_probabilities=prob_dist,
        predicted_idx=pred_idx,
        top1_class=top1_cls,
        top2_class=top2_cls,
        operational_threshold=applied_threshold,
        latency_ms=round(latency_ms, 2),
        review_reason=review_reason,
        tta_applied=True,
        tta_consensus_score=round(confidence, 4),
        calibration_metadata={"individual_tta_probs": [p.tolist() for p in individual_probs]},
    )

    # 9. Return Comprehensive & Clean Top-Level Response with Defect Diagnostic Analysis
    return PredictionResponse(
        request_id=request_id,
        inspection_id=inspection_unique_id,
        filename=file.filename,
        severity_assessment=pred_class,
        headline=headline,
        customer_message=customer_message,
        action_recommendation=action_recommendation,
        gating_status=gating_status,
        human_review_required=human_review_required,
        defect_analysis=defect_obj,
        predicted_class=pred_class,
        predicted_idx=pred_idx,
        confidence=round(confidence, 4),
        confidence_percentage=f"{confidence * 100:.2f}%",
        entropy=round(entropy, 4),
        probabilities=prob_dist,
        review_reason=review_reason,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        telemetry=telemetry_obj,
    )



@app.post(
    "/predict-video",
    response_model=VideoPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict Biofouling Severity on Video Feeds",
)
async def predict_video(
    file: UploadFile = File(..., description="Industrial drone or ROV video file (MP4, AVI)"),
    fps_sample_rate: float = Query(1.0, gt=0.0, description="Frames to sample per second of video"),
    threshold: Optional[float] = Query(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Custom operational confidence gating threshold",
    ),
):
    """
    Accepts an industrial drone/ROV video, extracts frames at the specified rate,
    applies robust real-world augmentations, and runs ONNX inference.
    Returns a comprehensive timeline report.
    """
    ensure_initialized()
    
    session: ort.InferenceSession = state.get("ort_session")
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ONNX Runtime inference engine is not ready.",
        )
        
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Supported: {sorted(allowed_extensions)}",
        )
        
    temp_video_path = PROJECT_ROOT / f"temp_{uuid.uuid4()}{file_ext}"
    try:
        with open(temp_video_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # Read in chunks of 1MB
                buffer.write(content)
                
        processor = IndustrialVideoProcessor(
            session=session,
            val_transforms=state["val_transforms"],
            input_name=state["input_name"],
            output_name=state["output_name"],
            idx_to_label=IDX_TO_LABEL
        )
        
        applied_threshold = threshold if threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
        
        results = processor.process_video(
            video_path=str(temp_video_path),
            fps_sample_rate=fps_sample_rate,
            threshold=applied_threshold
        )
        
        return VideoPredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}",
        )
    finally:
        if temp_video_path.exists():
            temp_video_path.unlink()


# ==============================================================================
# Full-Stack Dynamic Inspection Management Endpoints
# ==============================================================================

@app.get(
    "/api/inspections",
    response_model=InspectionListResponse,
    tags=["Inspection Management"],
    summary="List All Dynamic Biofouling Inspections with XAI Telemetry",
)
async def list_inspections(
    severity: Optional[str] = Query(None, description="Filter by biofouling severity: Low, Moderate, High, Critical"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by gating status: AUTOMATED_DECISION or HUMAN_EXPERT_REVIEW_REQUIRED"),
    search: Optional[str] = Query(None, description="Search term for filename or inspection ID"),
    limit: int = Query(50, ge=1, le=200, description="Pagination page limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """
    Retrieve all dynamically recorded biofouling inspections from the database.
    Supports filtering by severity level, operational gating status, and search query.
    """
    query = db.query(InspectionRecord)
    if severity and severity.upper() != "ALL":
        query = query.filter(InspectionRecord.predicted_class.ilike(severity))
    if status_filter and status_filter.upper() != "ALL":
        query = query.filter(InspectionRecord.gating_status == status_filter)
    if search:
        s = f"%{search}%"
        query = query.filter(
            (InspectionRecord.filename.ilike(s)) |
            (InspectionRecord.inspection_id.ilike(s)) |
            (InspectionRecord.predicted_class.ilike(s))
        )

    total = query.count()
    records = query.order_by(InspectionRecord.created_at.desc()).offset(offset).limit(limit).all()
    items = [InspectionItem(**r.to_dict()) for r in records]
    return InspectionListResponse(total=total, inspections=items)


@app.get(
    "/api/inspections/{inspection_id}",
    response_model=InspectionItem,
    tags=["Inspection Management"],
    summary="Get Specific Inspection by ID",
)
async def get_inspection_detail(
    inspection_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve full inspection record including Explainable AI (XAI) defect analysis."""
    record = db.query(InspectionRecord).filter(InspectionRecord.inspection_id == inspection_id).first()
    if not record and inspection_id.isdigit():
        record = db.query(InspectionRecord).filter(InspectionRecord.id == int(inspection_id)).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    return InspectionItem(**record.to_dict())


@app.delete(
    "/api/inspections/{inspection_id}",
    response_model=DeleteInspectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inspection Management"],
    summary="Delete Inspection Record and Server Storage Files",
)
async def delete_inspection(
    inspection_id: str,
    db: Session = Depends(get_db),
):
    """
    Securely removes an inspection record from the database and permanently deletes
    all associated image/video media files from server storage.
    """
    record = db.query(InspectionRecord).filter(InspectionRecord.inspection_id == inspection_id).first()
    if not record and inspection_id.isdigit():
        record = db.query(InspectionRecord).filter(InspectionRecord.id == int(inspection_id)).first()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record '{inspection_id}' not found.",
        )

    deleted_id = record.inspection_id

    # 1. Delete associated physical file from storage if present
    if record.file_path and record.file_path != "N/A":
        try:
            target_path = PROJECT_ROOT / record.file_path
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                logger.info(f"Removed inspection file: {target_path}")
        except Exception as file_err:
            logger.warning(f"Error removing inspection file {record.file_path}: {file_err}")

    # 2. Delete any matching review queue records and review images
    try:
        q_records = db.query(ReviewQueueRecord).filter(
            (ReviewQueueRecord.request_id == deleted_id) |
            (ReviewQueueRecord.filename == record.filename)
        ).all()
        for q in q_records:
            if q.saved_image_path and q.saved_image_path != "N/A":
                try:
                    q_file = PROJECT_ROOT / q.saved_image_path
                    if q_file.exists() and q_file.is_file():
                        q_file.unlink()
                except Exception:
                    pass
            db.delete(q)
    except Exception as q_err:
        logger.warning(f"Error removing linked review queue record: {q_err}")

    # 3. Delete database record
    db.delete(record)
    db.commit()
    logger.info(f"Inspection record '{deleted_id}' permanently deleted from database.")

    return DeleteInspectionResponse(
        status="success",
        success=True,
        message=f"Inspection '{deleted_id}' and all associated files deleted successfully.",
        deleted_id=deleted_id,
        inspection_id=deleted_id,
    )


@app.get(
    "/review-queue",
    response_model=ReviewQueueSummaryResponse,
    tags=["Human-in-the-Loop Management"],
    summary="List Flagged Inspection Frames Awaiting Human Review",
)
async def get_human_review_queue(
    status_filter: Optional[str] = Query(
        None, description="Filter queue by status: 'PENDING_REVIEW' or 'RESOLVED'"
    ),
    db: Session = Depends(get_db),
):
    """Retrieve all visual inspection frames that were flagged and escalated to human experts."""
    query = db.query(ReviewQueueRecord)
    if status_filter:
        query = query.filter(ReviewQueueRecord.review_status == status_filter.upper())
    records = query.order_by(ReviewQueueRecord.created_at.desc()).all()

    total_count = db.query(ReviewQueueRecord).count()
    pending_count = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.review_status == "PENDING_REVIEW").count()
    resolved_count = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.review_status == "RESOLVED").count()

    queue_items = [QueueItem(**rec.to_dict()) for rec in records]

    return ReviewQueueSummaryResponse(
        total_escalated_items=total_count if not status_filter else len(queue_items),
        pending_review_count=pending_count,
        resolved_count=resolved_count,
        queue_items=queue_items,
    )


@app.post(
    "/review-queue/{queue_id}/resolve",
    response_model=QueueItem,
    tags=["Human-in-the-Loop Management"],
    summary="Resolve an Escalated Frame with Human Expert Verification",
)
async def resolve_queue_item(
    queue_id: str,
    payload: ResolveQueueItemRequest,
    db: Session = Depends(get_db),
):
    """Mark a flagged inspection frame as resolved with human expert ground truth classification."""
    if payload.resolved_label not in LABEL_TO_IDX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid label '{payload.resolved_label}'. Allowed: {list(LABEL_TO_IDX.keys())}",
        )

    record = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.queue_id == queue_id).first()
    if not record and queue_id.isdigit():
        record = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.id == int(queue_id)).first()
    if not record:
        record = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.request_id == queue_id).first()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue item with ID '{queue_id}' not found.",
        )

    # Update item
    record.review_status = "RESOLVED"
    record.resolved_label = payload.resolved_label
    record.resolved_by = payload.resolved_by
    record.resolved_at = datetime.now(timezone.utc)
    advice = MAINTENANCE_ADVICE.get(payload.resolved_label, MAINTENANCE_ADVICE["Low"])
    record.customer_message = (
        f"Inspection Verified: Surface confirmed as {payload.resolved_label} biofouling by "
        f"specialist {payload.resolved_by}. Recommended action: {advice['action']}"
    )
    record.action_recommendation = advice["action"]
    if payload.notes:
        record.expert_notes = payload.notes

    db.commit()
    db.refresh(record)
    logger.info(f"Queue item {queue_id} (ID: {record.id}) resolved by {payload.resolved_by} as '{payload.resolved_label}'")

    return QueueItem(**record.to_dict())


# ==============================================================================
# BioAssist Domain LLM Chat Endpoint
# ==============================================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["BioAssist AI Chatbot"],
    summary="Domain-Specific Biofouling & Marine Inspection LLM Chat",
)
async def chat_with_bioassist(payload: ChatRequest):
    """
    Accepts user inquiries regarding industrial biofouling detection, marine inspection,
    biocide treatment regimens, and MLOps maintenance. Formats queries through the
    domain-specific chat pipeline and returns technical responses with zero cloud dependency.
    """
    t0 = time.perf_counter()
    logger.info(
        f"Incoming /api/chat request (session_id={payload.session_id}, msg_len={len(payload.message)})"
    )

    try:
        chatbot = get_chatbot()
        reply, model_used = chatbot.generate_reply(
            user_message=payload.message,
            history=payload.history,
            max_new_tokens=350,
            temperature=0.3,
            top_p=0.9,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Chatbot response generated in {latency_ms:.2f}ms using {model_used}")

        return ChatResponse(
            response=reply,
            model_used=model_used,
            status="success",
        )
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        # Robust offline fallback to guarantee zero crash
        fallback_msg = (
            f"**BioAssist Technical Advisory Response:**\n\n"
            f"For inquiry: *\"{payload.message}\"*\n\n"
            f"• **Inspection Assessment:** Maintain 380x380 standardized visual evaluation.\n"
            f"• **Action Guidance:** Schedule immediate chemical shock dosing for Critical/High biofilm matrices "
            f"and verify confidence safety gating (threshold >= 75%)."
        )
        return ChatResponse(
            response=fallback_msg,
            model_used="BioAssist Domain Engine (Offline Fallback)",
            status="success",
        )


# ==============================================================================
# MLOps & Active Learning Administrative Endpoints
# ==============================================================================

@app.post(
    "/admin/trigger-retrain",
    response_model=RetrainTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["MLOps & Active Learning"],
    summary="Trigger Active Learning Model Retraining on Expert-Resolved Dataset",
)
async def trigger_model_retraining(
    payload: RetrainTriggerRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin_key: str = Depends(verify_admin_key),
):
    """
    Protected MLOps trigger endpoint that queries verified expert-resolved inspection frames,
    expands the active learning training dataset, and kicks off incremental fine-tuning
    with automatic ONNX model export and engine hot-reloading once the sample threshold is met.
    """
    # Count verified resolved samples in database
    resolved_count = (
        db.query(ReviewQueueRecord)
        .filter(
            ReviewQueueRecord.review_status == "RESOLVED",
            ReviewQueueRecord.resolved_label.in_(list(LABEL_TO_IDX.keys())),
        )
        .count()
    )

    if resolved_count < payload.min_samples and not payload.force:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=RetrainTriggerResponse(
                status="SKIPPED_THRESHOLD_NOT_MET",
                message=(
                    f"Retraining skipped: {resolved_count} resolved sample(s) available, which is below "
                    f"the required minimum threshold of {payload.min_samples} samples. Use force=True to override."
                ),
                resolved_samples_count=resolved_count,
                min_samples_threshold=payload.min_samples,
                job_id=None,
                result=None,
            ).model_dump(),
        )

    if retrain_state["status"] == "RUNNING":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=RetrainTriggerResponse(
                status="ALREADY_RUNNING",
                message=f"A model fine-tuning job is currently in progress (Job ID: {retrain_state['job_id']}).",
                resolved_samples_count=resolved_count,
                min_samples_threshold=payload.min_samples,
                job_id=retrain_state["job_id"],
                result=None,
            ).model_dump(),
        )

    job_id = f"RETRAIN-{uuid.uuid4().hex[:8].upper()}"
    retrain_state["status"] = "RUNNING"
    retrain_state["job_id"] = job_id
    retrain_state["started_at"] = datetime.now(timezone.utc).isoformat()
    retrain_state["completed_at"] = None
    retrain_state["error"] = None
    retrain_state["last_result"] = None

    if payload.async_mode:
        background_tasks.add_task(
            run_background_retraining,
            job_id=job_id,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
        )
        return RetrainTriggerResponse(
            status="QUEUED",
            message=(
                f"Active learning fine-tuning job {job_id} successfully queued with {resolved_count} "
                f"expert-resolved training samples. Progress can be monitored via GET /admin/retrain-status."
            ),
            resolved_samples_count=resolved_count,
            min_samples_threshold=payload.min_samples,
            job_id=job_id,
            result=None,
        )
    else:
        # Synchronous execution
        try:
            result = fine_tune_model(
                epochs=payload.epochs,
                batch_size=payload.batch_size,
                learning_rate=payload.learning_rate,
                db_session=db,
            )
            retrain_state["status"] = "COMPLETED"
            retrain_state["completed_at"] = datetime.now(timezone.utc).isoformat()
            retrain_state["last_result"] = result
            retrain_state["error"] = None
            ensure_initialized(force_reload=True)

            return RetrainTriggerResponse(
                status="COMPLETED",
                message=f"Model fine-tuning job {job_id} completed synchronously. Validation Macro-F1: {result['final_val_f1']:.4f}.",
                resolved_samples_count=resolved_count,
                min_samples_threshold=payload.min_samples,
                job_id=job_id,
                result=result,
            )
        except Exception as e:
            retrain_state["status"] = "FAILED"
            retrain_state["completed_at"] = datetime.now(timezone.utc).isoformat()
            retrain_state["error"] = str(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Fine-tuning pipeline execution failed: {str(e)}",
            )


@app.get(
    "/admin/retrain-status",
    response_model=RetrainStatusResponse,
    tags=["MLOps & Active Learning"],
    summary="Get Status and Metrics of Model Retraining Jobs",
)
async def get_retrain_status(
    admin_key: str = Depends(verify_admin_key),
):
    """Retrieve the current execution state, completion timestamp, and telemetry of retraining jobs."""
    return RetrainStatusResponse(
        status=retrain_state["status"],
        job_id=retrain_state["job_id"],
        started_at=retrain_state["started_at"],
        completed_at=retrain_state["completed_at"],
        error=retrain_state["error"],
        last_result=retrain_state["last_result"],
    )


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

#!/usr/bin/env python3
"""
Database Configuration & ORM Models for Industrial Biofouling Assessment
-------------------------------------------------------------------------
Provides:
1. PostgreSQL engine & session factory with fallback to local connection string.
2. SQLAlchemy ORM model `ReviewQueueRecord` for Human-in-the-Loop (HITL) storage.
3. Clean FastAPI database session dependency generator `get_db`.
4. Schema initialization helper `init_db`.

Author: Principal Backend Engineer
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from pathlib import Path

# Environment variable configuration with fallback to local SQLite
SQLITE_DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "biofouling_local.db"
SQLITE_DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_URL = f"sqlite:///{SQLITE_DEFAULT_PATH.as_posix()}"

DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = None
if DATABASE_URL:
    try:
        connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
    except Exception:
        engine = None

if engine is None:
    engine = create_engine(DEFAULT_SQLITE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class ReviewQueueRecord(Base):
    """
    SQLAlchemy ORM Model representing an escalated inspection frame
    awaiting or finalized by Human-in-the-Loop (HITL) expert verification.
    """

    __tablename__ = "review_queue_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    queue_id = Column(String(50), unique=True, index=True, nullable=False)
    request_id = Column(String(100), index=True, nullable=True)
    filename = Column(String(255), nullable=False)
    saved_image_path = Column(String(500), nullable=True)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    entropy = Column(Float, nullable=True)
    probabilities = Column(JSON, nullable=False)
    gating_status = Column(String(50), nullable=False)
    review_status = Column(String(50), nullable=False, default="PENDING_REVIEW", index=True)
    review_reason = Column(Text, nullable=True)
    customer_message = Column(Text, nullable=True)
    action_recommendation = Column(Text, nullable=True)
    expert_notes = Column(Text, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolved_label = Column(String(50), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record into a dictionary compatible with Pydantic QueueItem."""
        return {
            "id": self.id,
            "queue_id": self.queue_id,
            "request_id": self.request_id or f"REQ-{self.id}",
            "filename": self.filename,
            "saved_image_path": self.saved_image_path or "N/A",
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "entropy": self.entropy if self.entropy is not None else 0.0,
            "probabilities": self.probabilities if isinstance(self.probabilities, dict) else {},
            "gating_status": self.gating_status,
            "review_status": self.review_status,
            "status": self.review_status,  # Backwards compatibility alias for QueueItem.status
            "review_reason": self.review_reason or "",
            "customer_message": self.customer_message,
            "action_recommendation": self.action_recommendation,
            "expert_notes": self.expert_notes,
            "resolved_by": self.resolved_by,
            "resolved_label": self.resolved_label,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else datetime.now(timezone.utc).isoformat(),
        }


class InspectionRecord(Base):
    """
    SQLAlchemy ORM Model representing an industrial biofouling visual inspection.
    Stores complete inspection telemetry, file paths, safety gating status, and
    detailed explainable defect reasoning (XAI).
    """

    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(String(100), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    media_type = Column(String(50), default="image", nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    confidence_percentage = Column(String(20), nullable=True)
    entropy = Column(Float, nullable=True)
    probabilities = Column(JSON, nullable=True)
    gating_status = Column(String(50), nullable=False)
    human_review_required = Column(Boolean, default=False, nullable=False)
    defect_analysis = Column(JSON, nullable=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record into a dictionary compatible with API schemas."""
        analysis = self.defect_analysis if isinstance(self.defect_analysis, dict) else {}
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "filename": self.filename,
            "file_path": self.file_path or "N/A",
            "media_type": self.media_type or "image",
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "confidence_percentage": self.confidence_percentage or f"{self.confidence * 100:.1f}%",
            "entropy": self.entropy if self.entropy is not None else 0.0,
            "probabilities": self.probabilities if isinstance(self.probabilities, dict) else {},
            "gating_status": self.gating_status,
            "human_review_required": self.human_review_required,
            "defect_analysis": analysis,
            "primary_issue": analysis.get("primary_issue", "N/A"),
            "reasoning": analysis.get("reasoning", "N/A"),
            "action_required": analysis.get("action_required", "N/A"),
            "latency_ms": self.latency_ms if self.latency_ms is not None else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else datetime.now(timezone.utc).isoformat(),
        }


def init_db(target_engine=None) -> bool:
    """Initialize database tables defined in metadata with automatic local SQLite fallback."""
    global engine, SessionLocal
    eng = target_engine or engine
    try:
        with eng.connect() as conn:
            pass
        Base.metadata.create_all(bind=eng)
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger("BiofoulingDB")
        logger.warning(
            f"Primary database ({eng.url.drivername}) unavailable ({e}). Falling back to local SQLite database (./data/biofouling_local.db)."
        )
        try:
            from pathlib import Path
            sqlite_path = Path(__file__).resolve().parent / "data" / "biofouling_local.db"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            Base.metadata.create_all(bind=engine)
            return True
        except Exception as sq_err:
            logger.error(f"SQLite fallback failed: {sq_err}")
            return False


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional SQLAlchemy session per request
    and guarantees proper closure upon request completion.
    """
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

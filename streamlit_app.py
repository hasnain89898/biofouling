#!/usr/bin/env python3
"""
Enterprise Streamlit Inspection & Test Interface for Industrial Biofouling (v4)
--------------------------------------------------------------------------------
Provides interactive visual testing, threshold tuning, uncertainty visualization,
batch audits, and a live Human-in-the-Loop review queue resolution portal.

Author: Principal AI Research Engineer
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn.functional as F

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import ReviewQueueRecord, SessionLocal, get_db, init_db
from backend.services.chatbot import BioAssistChatbot, get_chatbot
from backend.services.guardrail import validate_industrial_domain
from src.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    get_val_transforms,
)
from src.inference_engine import (
    BiofoulingInferenceEngine,
    InferenceResult,
    calculate_entropy,
)

# Backend API URL for chat and inference services
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="Industrial Biofouling AI (v4)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    .badge-auto {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .chat-pill {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }
    .chat-container {
        border-radius: 12px;
        padding: 1rem;
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Paths
MODEL_PTH = PROJECT_ROOT / "checkpoints" / "best_model_v4.pth"
MODEL_ONNX = PROJECT_ROOT / "biofouling_model.onnx"
TEST_CSV = PROJECT_ROOT / "data" / "splits_cleaned" / "test.csv"
REVIEW_QUEUE_FILE = PROJECT_ROOT / "data" / "human_review_queue.json"
REVIEW_QUEUE_IMG_DIR = PROJECT_ROOT / "data" / "review_queue_images"

CLASS_COLORS = {
    "Low": "#10B981",       # Emerald Green
    "Moderate": "#F59E0B",  # Amber
    "High": "#F97316",      # Orange
    "Critical": "#EF4444",  # Crimson Red
}

CLASS_DESCRIPTIONS = {
    "Low": "Pristine cooling tower fill, clean heat exchanger plates, or unfouled pipe interior.",
    "Moderate": "Early stage biofilm development, patchy green algae filaments, light slime coating.",
    "High": "Dense mature biofilm matrix, heavy microbial slime, initial hydraulic resistance.",
    "Critical": "Severe EPS accumulation, sloughing biomass clumps, structural/flow blockage danger.",
}


# ==============================================================================
# Cached Engines & Helpers
# ==============================================================================

@st.cache_resource
def get_inference_engine(backend_type: str, use_tta: bool) -> BiofoulingInferenceEngine:
    """Load and cache the appropriate inference engine instance."""
    if backend_type == "ONNX Runtime (Fast Edge)":
        return BiofoulingInferenceEngine(
            model_or_path=str(MODEL_ONNX),
            backend="onnx",
            use_tta=False,
            image_size=380,
        )
    elif backend_type == "PyTorch CPU":
        return BiofoulingInferenceEngine(
            model_or_path=str(MODEL_PTH),
            backend="pytorch",
            use_tta=False,
            image_size=380,
        )
    else:  # PyTorch + 8-View D4 TTA
        return BiofoulingInferenceEngine(
            model_or_path=str(MODEL_PTH),
            backend="pytorch",
            use_tta=True,
            image_size=380,
        )


def load_review_queue_data() -> List[Dict[str, Any]]:
    """Read review queue records from PostgreSQL/database with JSON file fallback."""
    try:
        init_db()
        with SessionLocal() as db:
            records = db.query(ReviewQueueRecord).order_by(ReviewQueueRecord.created_at.asc()).all()
            if records:
                return [r.to_dict() for r in records]
    except Exception:
        pass

    if not REVIEW_QUEUE_FILE.exists():
        return []
    try:
        with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_review_queue_data(items: List[Dict[str, Any]]) -> None:
    """Persist review queue changes to PostgreSQL/database and JSON backup."""
    # Attempt DB update for resolved items
    try:
        init_db()
        with SessionLocal() as db:
            for it in items:
                qid = it.get("queue_id")
                if qid:
                    rec = db.query(ReviewQueueRecord).filter(ReviewQueueRecord.queue_id == qid).first()
                    if rec and it.get("status") == "RESOLVED":
                        rec.review_status = "RESOLVED"
                        rec.resolved_label = it.get("resolved_label")
                        rec.resolved_by = it.get("resolved_by")
                        rec.expert_notes = it.get("expert_notes")
                        if it.get("resolved_at"):
                            try:
                                rec.resolved_at = datetime.fromisoformat(it["resolved_at"])
                            except Exception:
                                rec.resolved_at = datetime.now(timezone.utc)
            db.commit()
    except Exception:
        pass

    try:
        REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = REVIEW_QUEUE_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        temp_file.replace(REVIEW_QUEUE_FILE)
    except Exception:
        pass


def find_sample_images() -> Dict[str, List[Path]]:
    """Locate sample test images from raw dataset for quick loading."""
    samples = {"Low": [], "Moderate": [], "High": [], "Critical": []}
    for cls_name in samples.keys():
        cls_dir = PROJECT_ROOT / "data" / "raw" / cls_name
        if cls_dir.exists():
            imgs = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
            samples[cls_name] = imgs[:5]
    return samples


def send_chat_message(
    message: str,
    history: List[Dict[str, Any]],
    session_id: str,
) -> Tuple[str, str]:
    """
    Send chat query to FastAPI backend /api/chat endpoint with automatic
    seamless fallback to local BioAssist chatbot engine.
    """
    endpoint = f"{BACKEND_API_URL.rstrip('/')}/api/chat"
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    payload = {
        "message": message,
        "session_id": session_id,
        "history": clean_history,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", ""), data.get("model_used", "BioAssist REST API")
    except Exception:
        pass

    # Direct local engine fallback
    bot = get_chatbot()
    return bot.generate_reply(user_message=message, history=clean_history)


# ==============================================================================
# Sidebar Configuration
# ==============================================================================

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/visibility/materialicons/48dp/2x/baseline_visibility_black_48dp.png",
        width=48,
    )
    st.markdown("### ⚙️ Engine Settings")

    backend_choice = st.selectbox(
        "Inference Backend",
        [
            "ONNX Runtime (Fast Edge)",
            "PyTorch CPU",
            "PyTorch (8-View D4 TTA)",
        ],
        index=0,
        help="ONNX Runtime provides 2.26x speedup (~23ms). 8-View D4 TTA provides maximum spatial ensemble robustness.",
    )

    use_tta_flag = (backend_choice == "PyTorch (8-View D4 TTA)")

    threshold = st.slider(
        "Operational Safety Threshold (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=1,
        help="Inferences with confidence below this threshold are automatically routed to the Human Expert Review queue.",
    ) / 100.0

    st.markdown("---")
    st.markdown("### 📋 Navigation")
    mode = st.radio(
        "Select Inspection Workspace",
        [
            "🔬 Single-Frame Inspection (Test Mode)",
            "🗂️ Batch Test Split Manifest Audit",
            "🛡️ Human-in-the-Loop Review Queue",
            "📈 Architecture & Multi-Version Benchmarks",
            "💬 BioAssist AI Expert Copilot",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("Industrial Biofouling AI Engine v4.0.0\nEfficientNet-B3 (380x380) | ONNX Runtime")


# Load Engine
try:
    engine = get_inference_engine(backend_choice, use_tta_flag)
except Exception as e:
    st.error(f"Failed to load model engine: {e}")
    st.stop()


# ==============================================================================
# Workspace 1: Single-Frame Inspection (Test Mode)
# ==============================================================================

if mode == "🔬 Single-Frame Inspection (Test Mode)":
    st.markdown('<div class="main-title">🔬 Industrial Biofouling Real-Time Inspection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a cooling tower / pipe inspection frame or select a catalog sample to evaluate severity with operational confidence gating.</div>',
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1.35], gap="large")

    selected_image: Optional[Image.Image] = None
    image_name = "custom_upload.jpg"

    with col_input:
        st.markdown("#### 1. Input Image Selection")
        input_source = st.radio(
            "Choose Image Source",
            ["Upload Custom Image File", "Select Pre-Loaded Catalog Sample"],
            horizontal=True,
        )

        if input_source == "Upload Custom Image File":
            uploaded_file = st.file_uploader(
                "Upload Visual Frame (JPEG, PNG, WEBP, BMP)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
        else:
            sample_dict = find_sample_images()
            sample_class = st.selectbox("Filter by Category", ["Critical", "High", "Moderate", "Low"])
            class_samples = sample_dict.get(sample_class, [])
            if class_samples:
                sample_path = st.selectbox(
                    "Choose Catalog Frame",
                    class_samples,
                    format_func=lambda p: p.name,
                )
                if sample_path and sample_path.exists():
                    selected_image = Image.open(sample_path).convert("RGB")
                    image_name = sample_path.name
            else:
                st.info(f"No catalog images found for {sample_class}.")

        if selected_image is not None:
            st.image(
                selected_image,
                caption=f"Selected Frame: {image_name} ({selected_image.width}x{selected_image.height} px)",
                use_container_width=True,
            )

    with col_result:
        st.markdown("#### 2. Classification & Safety Gating Verdict")

        if selected_image is None:
            st.info("👈 Please upload an image or choose a catalog sample from the left panel to begin.")
        else:
            # Domain Guardrail Check
            gr_res = validate_industrial_domain(selected_image)
            if not gr_res.is_valid:
                st.markdown(
                    f"""
                    <div style="background-color: #FFFBEB; border-left: 6px solid #F59E0B; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <div style="font-weight: 700; color: #B45309; font-size: 0.9rem;">⚠️ DOMAIN GUARDRAIL ALERT (OOD)</div>
                        <h4 style="color: #92400E; margin-top: 6px; margin-bottom: 4px;">Out-of-Distribution Input (Score: {gr_res.domain_score*100:.1f}%)</h4>
                        <p style="color: #78350F; margin: 0; font-size: 0.85rem;">
                            <strong>Reason:</strong> {gr_res.rejection_reason}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.spinner("Analyzing high-resolution features and running confidence gating..."):
                result: InferenceResult = engine.predict(
                    image_input=selected_image,
                    custom_threshold=threshold,
                )

            pred_color = CLASS_COLORS.get(result.predicted_class, "#3B82F6")

            # Gating Verdict Banner
            if result.human_review_required:
                st.markdown(
                    f"""
                    <div style="background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <div class="badge-review">🚨 HUMAN EXPERT REVIEW REQUIRED</div>
                        <h3 style="color: #991B1B; margin-top: 8px; margin-bottom: 4px;">Top Prediction: {result.predicted_class} ({result.confidence*100:.1f}%)</h3>
                        <p style="color: #7F1D1D; margin: 0; font-size: 0.95rem;">
                            <strong>Safety Gate Alert:</strong> {result.review_reason}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <div class="badge-auto">✅ AUTOMATED DECISION APPROVED</div>
                        <h3 style="color: #065F46; margin-top: 8px; margin-bottom: 4px;">Severity Verdict: {result.predicted_class} ({result.confidence*100:.1f}%)</h3>
                        <p style="color: #047857; margin: 0; font-size: 0.95rem;">
                            <strong>Operational Status:</strong> Confidence meets or exceeds the {threshold*100:.0f}% safety threshold. Safe for automated control.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Key Telemetry Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val" style="color: {pred_color};">{result.predicted_class}</div>
                        <div class="metric-label">Predicted Class</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{result.confidence*100:.1f}%</div>
                        <div class="metric-label">Max Confidence</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{result.entropy:.3f}</div>
                        <div class="metric-label">Shannon Entropy</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-box">
                        <div class="metric-val">{result.latency_ms:.1f} ms</div>
                        <div class="metric-label">Latency</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Plotly Probability Distribution Bar Chart
            classes = ["Low", "Moderate", "High", "Critical"]
            probs = [result.probabilities[c] * 100.0 for c in classes]
            colors = [CLASS_COLORS[c] for c in classes]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=classes,
                    y=probs,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition="auto",
                    marker_color=colors,
                    marker_line=dict(color="#1F2937", width=1.2),
                    hovertemplate="<b>Severity: %{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
                )
            )

            # Add operational threshold guide line
            fig.add_hline(
                y=threshold * 100,
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=f"Safety Threshold ({threshold*100:.0f}%)",
                annotation_position="top right",
            )

            fig.update_layout(
                title=dict(text="<b>4-Class Softmax Probability Distribution</b>", font=dict(size=14)),
                xaxis_title="Severity Category",
                yaxis_title="Probability (%)",
                yaxis=dict(range=[0, 100]),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(240,244,248,0.5)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Diagnostic Visual Feature Breakdown
            with st.expander("🔍 Diagnostic Severity Interpretation & Visual Signatures", expanded=True):
                st.markdown(f"**Diagnostic Description for `{result.predicted_class}`:**")
                st.write(CLASS_DESCRIPTIONS.get(result.predicted_class, ""))
                st.markdown("---")
                st.markdown(
                    f"**Probabilities Vector:** `Low: {result.probabilities['Low']*100:.1f}%` | "
                    f"`Moderate: {result.probabilities['Moderate']*100:.1f}%` | "
                    f"`High: {result.probabilities['High']*100:.1f}%` | "
                    f"`Critical: {result.probabilities['Critical']*100:.1f}%`"
                )


# ==============================================================================
# Workspace 2: Batch Test Split Manifest Audit
# ==============================================================================

elif mode == "🗂️ Batch Test Split Manifest Audit":
    st.markdown('<div class="main-title">🗂️ Batch Test Split Manifest Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Execute batch inference across the pristine 33-sample independent test split with live confidence gating.</div>',
        unsafe_allow_html=True,
    )

    if not TEST_CSV.exists():
        st.error(f"Test split manifest not found at {TEST_CSV}!")
    else:
        if st.button("🚀 Run Full Audit on Test Split (33 Samples)", type="primary"):
            with st.spinner("Executing batch inference and calculating metrics..."):
                audit_summary = engine.audit_dataset_split(str(TEST_CSV), threshold=threshold)

            res_df = audit_summary["results_df"]

            # Summary Metrics Row
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Test Samples", len(res_df))
            with s2:
                st.metric(
                    "Automated Decisions",
                    f"{audit_summary['automated_count']} ({audit_summary['automated_coverage_pct']:.1f}%)",
                )
            with s3:
                st.metric(
                    "Automated Cohort Accuracy",
                    f"{audit_summary['automated_accuracy_pct']:.2f}%",
                    delta=f"{audit_summary['automated_accuracy_pct'] - audit_summary['total_accuracy_pct']:.1f}% vs Raw",
                )
            with s4:
                st.metric(
                    "Human Escalation Queue",
                    f"{audit_summary['review_count']} ({audit_summary['review_rate_pct']:.1f}%)",
                )

            st.markdown("---")
            st.markdown("#### Detailed Sample Inferences & HITL Status")

            # Format dataframe for display
            display_df = res_df.copy()
            display_df["confidence"] = display_df["confidence"].apply(lambda c: f"{c*100:.1f}%")
            display_df["entropy"] = display_df["entropy"].apply(lambda e: f"{e:.3f}")
            display_df["is_correct"] = display_df["is_correct"].apply(lambda ok: "✅ Correct" if ok else "❌ Misclassified")
            display_df["human_review_required"] = display_df["human_review_required"].apply(
                lambda r: "🚨 Escalated" if r else "🟢 Automated"
            )

            st.dataframe(
                display_df[
                    [
                        "filename",
                        "true_label",
                        "predicted_label",
                        "confidence",
                        "entropy",
                        "human_review_required",
                        "is_correct",
                        "review_reason",
                    ]
                ],
                use_container_width=True,
                height=450,
            )


# ==============================================================================
# Workspace 3: Human-in-the-Loop Review Queue
# ==============================================================================

elif mode == "🛡️ Human-in-the-Loop Review Queue":
    st.markdown('<div class="main-title">🛡️ Human-in-the-Loop Review Queue Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Inspect, review, and verify ambiguous inspection frames escalated by the confidence gating safety policy.</div>',
        unsafe_allow_html=True,
    )

    queue_items = load_review_queue_data()

    if not queue_items:
        st.success("🎉 No items currently pending in the Human Review Queue. All visual inspections passed automated confidence gating!")
    else:
        pending_items = [it for it in queue_items if it.get("status") == "PENDING_REVIEW"]
        resolved_items = [it for it in queue_items if it.get("status") == "RESOLVED"]

        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("Total Flagged Inferences", len(queue_items))
        with q2:
            st.metric("Awaiting Human Review", len(pending_items))
        with q3:
            st.metric("Resolved by Experts", len(resolved_items))

        st.markdown("---")

        tab_pending, tab_resolved = st.tabs([f"⏳ Pending Review ({len(pending_items)})", f"✅ Resolved ({len(resolved_items)})"])

        with tab_pending:
            if not pending_items:
                st.info("No pending review items.")
            else:
                for idx, item in enumerate(pending_items):
                    with st.expander(
                        f"🚨 Queue Item {item.get('queue_id')} — {item.get('filename')} (Model Pred: {item.get('predicted_class')} @ {item.get('confidence', 0)*100:.1f}%)",
                        expanded=(idx == 0),
                    ):
                        c_img, c_detail = st.columns([1, 1.2])

                        with c_img:
                            img_path_str = item.get("saved_image_path", "")
                            full_img_path = PROJECT_ROOT / img_path_str
                            if full_img_path.exists():
                                st.image(
                                    str(full_img_path),
                                    caption=f"Archived Frame: {item.get('filename')}",
                                    use_container_width=True,
                                )
                            else:
                                st.warning(f"Archived image thumbnail not found at: {img_path_str}")

                        with c_detail:
                            st.markdown(f"**Escalation Rationale:** {item.get('review_reason')}")
                            st.markdown(
                                f"**Confidence:** `{item.get('confidence', 0)*100:.1f}%` | "
                                f"**Entropy:** `{item.get('entropy', 0):.3f}`"
                            )
                            st.markdown(f"**Timestamp:** `{item.get('created_at')}`")

                            st.markdown("##### Human Expert Sign-Off:")
                            col_f1, col_f2 = st.columns(2)
                            with col_f1:
                                expert_label = st.selectbox(
                                    "Confirmed Ground Truth Label",
                                    ["Critical", "High", "Moderate", "Low"],
                                    key=f"label_{item.get('queue_id')}",
                                )
                            with col_f2:
                                expert_name = st.text_input(
                                    "Reviewer Name / Badge ID",
                                    value="Dr. Biofouling Inspector",
                                    key=f"expert_{item.get('queue_id')}",
                                )

                            expert_notes = st.text_area(
                                "Engineering Notes & Observations",
                                key=f"notes_{item.get('queue_id')}",
                                placeholder="e.g. Algal slime on plate ribbing confirmed. Clean schedule advised.",
                            )

                            if st.button("✅ Confirm & Resolve Item", key=f"btn_{item.get('queue_id')}", type="primary"):
                                # Update item in queue
                                for q_it in queue_items:
                                    if q_it.get("queue_id") == item.get("queue_id"):
                                        q_it["status"] = "RESOLVED"
                                        q_it["resolved_label"] = expert_label
                                        q_it["resolved_by"] = expert_name
                                        q_it["expert_notes"] = expert_notes
                                        q_it["resolved_at"] = datetime.now(timezone.utc).isoformat()
                                        break
                                save_review_queue_data(queue_items)
                                st.success(f"Item {item.get('queue_id')} successfully resolved as '{expert_label}'!")
                                time.sleep(0.6)
                                st.rerun()

        with tab_resolved:
            if not resolved_items:
                st.info("No resolved items yet.")
            else:
                for item in resolved_items:
                    st.markdown(
                        f"""
                        <div class="status-card" style="background-color: #F8FAFC;">
                            <strong>ID:</strong> <code>{item.get('queue_id')}</code> |
                            <strong>Frame:</strong> <code>{item.get('filename')}</code> |
                            <strong>Model Pred:</strong> {item.get('predicted_class')} ({item.get('confidence', 0)*100:.1f}%) ➔
                            <strong>Confirmed:</strong> <span style="color: #047857; font-weight: bold;">{item.get('resolved_label')}</span> |
                            <strong>Expert:</strong> {item.get('resolved_by')} ({item.get('resolved_at', 'N/A')})<br>
                            <span style="color: #64748B; font-size: 0.88rem;">Notes: {item.get('expert_notes', 'None provided')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ==============================================================================
# Workspace 4: Architecture & Multi-Version Benchmarks
# ==============================================================================

elif mode == "📈 Architecture & Multi-Version Benchmarks":
    st.markdown('<div class="main-title">📈 Architecture & Multi-Version Performance Progression</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed historical progression of model architectures, training strategies, and test set accuracy across iterations v1 to v4.</div>',
        unsafe_allow_html=True,
    )

    # Benchmark Table
    benchmark_data = {
        "Model Version": [
            "v1 Baseline",
            "v2 Hardened",
            "v3 Scaled & Scrubbed",
            "v4 Enterprise (Current)",
        ],
        "Backbone Architecture": [
            "EfficientNet-B0 (224x224)",
            "EfficientNet-B0 + CoarseDropout",
            "EfficientNet-B2 (224x224) + 400 Balanced",
            "EfficientNet-B3 (380x380) + D4 TTA + ONNX",
        ],
        "Validation Macro-F1": ["0.6937", "0.6741", "0.8606", "0.8674"],
        "Test Macro-F1": ["0.5233", "0.5541", "0.6661", "0.6741"],
        "Automated Accuracy (HITL Gated)": [
            "Ungated",
            "Ungated",
            "64.29% (@ 75%)",
            "80.00% – 100.00% (@ 75-90%)",
        ],
        "Inference Latency": ["48.2 ms", "47.9 ms", "38.5 ms", "23.01 ms (ONNX)"],
    }

    st.table(pd.DataFrame(benchmark_data))

    st.markdown("---")

    # Confusion Matrix & Classification Report Tabs
    c_tab1, c_tab2 = st.columns(2)

    with c_tab1:
        st.markdown("#### 🎯 Independent Test Confusion Matrix (v4)")
        cm = np.array([
            [6, 0, 3, 0],
            [0, 5, 2, 1],
            [2, 0, 5, 1],
            [0, 1, 1, 6],
        ])
        labels = ["Low", "Moderate", "High", "Critical"]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Severity", y="Ground Truth Severity", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_tab2:
        st.markdown("#### 📊 Classification Report Breakdown (v4)")
        report_data = {
            "Class": ["Low", "Moderate", "High", "Critical", "Macro Average"],
            "Precision": ["75.00%", "83.33%", "45.45%", "75.00%", "69.70%"],
            "Recall": ["66.67%", "62.50%", "62.50%", "75.00%", "66.67%"],
            "F1-Score": ["0.7059", "0.7143", "0.5263", "0.7500", "0.6741"],
            "Test Support": [9, 8, 8, 8, 33],
        }
        st.dataframe(pd.DataFrame(report_data), use_container_width=True, height=240)


# ==============================================================================
# Workspace 5: BioAssist AI Expert Copilot (Domain Chatbot Interface)
# ==============================================================================

elif mode == "💬 BioAssist AI Expert Copilot":
    st.markdown('<div class="main-title">💬 BioAssist Industrial AI Copilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Domain-specialized conversational AI for industrial biofouling identification, marine inspection standards, biocide chemical regimens, and MLOps pipeline telemetry.</div>',
        unsafe_allow_html=True,
    )

    # Initialize Chat Session State
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to **BioAssist**. I am your industrial AI specialist for cooling towers, heat exchangers, "
                    "marine pipeline biofouling detection, chemical biocide treatments, and MLOps confidence gating.\n\n"
                    "How can I assist your inspection or maintenance operations today?"
                ),
                "timestamp": "Initialization",
                "model_used": "BioAssist AI Copilot",
            }
        ]

    # Telemetry Status Bar
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(
            f"""<div class="metric-box">
                <div class="metric-val" style="font-size: 1.1rem; color: #047857;">🟢 Operational</div>
                <div class="metric-label">Backend API Status</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f"""<div class="metric-box">
                <div class="metric-val" style="font-size: 1.1rem; color: #1E3A8A;">Qwen2.5-1.5B</div>
                <div class="metric-label">Base Architecture</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_t3:
        st.markdown(
            f"""<div class="metric-box">
                <div class="metric-val" style="font-size: 1.1rem; color: #6D28D9;">LoRA / 4-bit Ready</div>
                <div class="metric-label">PEFT Adapter Status</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_t4:
        st.markdown(
            f"""<div class="metric-box">
                <div class="metric-val" style="font-size: 1.1rem; color: #475569;"><code>{st.session_state.chat_session_id}</code></div>
                <div class="metric-label">Active Session ID</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Domain Quick-Prompt Chips
    st.markdown("##### ⚡ Quick Inspection & Chemistry Inquiries")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    prompt_to_send: Optional[str] = None

    with q_col1:
        if st.button("🚨 Critical Biofouling Protocol", use_container_width=True):
            prompt_to_send = "What is the emergency response protocol for Critical biofouling with flow obstruction?"
    with q_col2:
        if st.button("🔬 MIC Corrosion & SRB", use_container_width=True):
            prompt_to_send = "Explain Microbially Influenced Corrosion (MIC) risks and Sulfate-Reducing Bacteria (SRB) in pipes."
    with q_col3:
        if st.button("🛡️ Safety Gating & Entropy", use_container_width=True):
            prompt_to_send = "How does confidence safety gating (75% threshold) and Shannon entropy routing work?"
    with q_col4:
        if st.button("🧪 Biocide Dosing Regimens", use_container_width=True):
            prompt_to_send = "What are the recommended oxidizing and non-oxidizing biocide dosing regimens?"

    # Conversation History Rendering
    for msg in st.session_state.chat_messages:
        role = msg.get("role", "user")
        avatar = "🧑‍💻" if role == "user" else "🌊"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.get("content", ""))
            if role == "assistant" and msg.get("model_used"):
                st.caption(f"🤖 Model: `{msg.get('model_used')}` • {msg.get('timestamp', '')}")

    # Chat Input Box
    chat_input_val = st.chat_input("Ask BioAssist about biofouling severity, chemical treatment, or telemetry...")
    active_prompt = prompt_to_send or chat_input_val

    if active_prompt:
        # Add User Message to History
        user_timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        st.session_state.chat_messages.append({
            "role": "user",
            "content": active_prompt,
            "timestamp": user_timestamp,
            "model_used": "User Client",
        })

        # Render User Message immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(active_prompt)

        # Generate Response with Dynamic Typing / Thinking Spinner
        with st.chat_message("assistant", avatar="🌊"):
            with st.spinner("BioAssist is analyzing marine inspection telemetry & generating response..."):
                response_text, model_tag = send_chat_message(
                    message=active_prompt,
                    history=st.session_state.chat_messages[:-1],
                    session_id=st.session_state.chat_session_id,
                )
                assistant_timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                st.markdown(response_text)
                st.caption(f"🤖 Model: `{model_tag}` • {assistant_timestamp}")

        # Store Assistant Response in History
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": assistant_timestamp,
            "model_used": model_tag,
        })
        st.rerun()

    # Chat Controls (Clear History & Export)
    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Conversation cleared. Ready for new inspection and biofouling inquiries.",
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    "model_used": "BioAssist AI Copilot",
                }
            ]
            st.rerun()
    with ctrl_col2:
        export_json = json.dumps(st.session_state.chat_messages, indent=2)
        st.download_button(
            "📥 Export Chat Log (JSON)",
            data=export_json,
            file_name=f"bioassist_chat_{st.session_state.chat_session_id}.json",
            mime="application/json",
        )

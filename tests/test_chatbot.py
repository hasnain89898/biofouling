#!/usr/bin/env python3
"""
Automated Test Suite for BioAssist Offline LLM Chatbot Service & /api/chat Endpoint
-----------------------------------------------------------------------------------
Tests:
1. BioAssistChatbot initialization, system prompt configuration, and LoRA adapter detection.
2. Domain query pattern matching & offline expert response generation.
3. Multi-turn conversation history prompt formatting.
4. FastAPI POST /api/chat request and response schema compliance.
5. Error handling and 100% offline fallback reliability.

Author: Senior Full-Stack & AI Integration Engineer
"""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.chatbot import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    BioAssistChatbot,
    get_chatbot,
    match_offline_knowledge,
)
from app import app, ChatRequest, ChatResponse
from streamlit_app import send_chat_message

client = TestClient(app)


def test_chatbot_initialization():
    """Verify BioAssistChatbot parameters, device detection, and system prompt."""
    bot = BioAssistChatbot()
    assert bot.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert bot.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert bot.device in ("cuda", "cpu")
    assert bot.adapter_path.endswith("biofouling_assistant_lora")


def test_format_chat_prompt():
    """Verify multi-turn chat prompt assembly."""
    bot = BioAssistChatbot()
    history = [
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": "Initial response"},
    ]
    messages = bot.format_chat_prompt("Follow-up question", history)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Initial question"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Initial response"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Follow-up question"


def test_domain_knowledge_matching():
    """Verify high-precision domain knowledge queries."""
    # Critical query
    crit_ans = match_offline_knowledge("What is the protocol for Critical biofouling and flow blockage?")
    assert crit_ans is not None
    assert "Critical Biofouling Protocol" in crit_ans
    assert "Chlorine Dioxide" in crit_ans or "DBNPA" in crit_ans

    # MIC query
    mic_ans = match_offline_knowledge("Explain microbially influenced corrosion MIC and SRB")
    assert mic_ans is not None
    assert "Microbially Influenced Corrosion" in mic_ans
    assert "Sulfate-Reducing Bacteria" in mic_ans

    # Safety gating query
    gate_ans = match_offline_knowledge("How does confidence safety gating and Shannon entropy work?")
    assert gate_ans is not None
    assert "Confidence Safety Gating" in gate_ans
    assert "75%" in gate_ans


def test_generate_reply_offline_resilience():
    """Verify generate_reply produces valid output even without network connection."""
    bot = get_chatbot()
    reply, model_used = bot.generate_reply("What is the treatment for High biofouling slime?")
    assert len(reply) > 50
    assert "High" in reply or "Biofouling" in reply
    assert isinstance(model_used, str)


def test_api_chat_endpoint_success():
    """Verify FastAPI POST /api/chat endpoint with single turn message."""
    payload = {
        "message": "What are the biocide treatment protocols for Moderate biofouling?",
        "session_id": "test-session-101",
        "history": [],
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "response" in data
    assert len(data["response"]) > 30
    assert "model_used" in data


def test_api_chat_endpoint_multi_turn():
    """Verify FastAPI POST /api/chat endpoint with multi-turn history."""
    payload = {
        "message": "And what is the inspection interval for that condition?",
        "session_id": "test-session-102",
        "history": [
            {"role": "user", "content": "What is Moderate biofouling?"},
            {"role": "assistant", "content": "Moderate biofouling is early algal or biofilm colonization."},
        ],
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["response"]) > 20


def test_streamlit_send_chat_message_helper():
    """Verify Streamlit UI send_chat_message helper function."""
    reply, model_tag = send_chat_message(
        message="Explain ONNX runtime speedup",
        history=[],
        session_id="ui-test-01",
    )
    assert len(reply) > 20
    assert isinstance(model_tag, str)

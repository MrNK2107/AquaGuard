"""
FastAPI Automated Test Suite for SeaClear RT-DETR Detection & Reasoning API
===========================================================================
Tests /health, /detect (Part A), and /reason (Part B) including intent routing,
structured detection synthesis, and confidence/unanswerable guardrails.
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_dummy_image():
    img = Image.new("RGB", (640, 640), color=(20, 60, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "classes" in data
    assert len(data["classes"]) == 6

def test_detect_endpoint_valid_image():
    buf = create_dummy_image()
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "count" in data
    assert "inference_ms" in data
    assert isinstance(data["detections"], list)

def test_reason_intent_routing_unrelated():
    buf = create_dummy_image()
    response = client.post(
        "/reason",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"question": "Hello, what can you do?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["used_detector"] is False
    assert "marine debris" in data["answer"].lower() or "assistant" in data["answer"].lower()

def test_reason_guardrail_unanswerable():
    buf = create_dummy_image()
    response = client.post(
        "/reason",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"question": "What is the exact water depth in meters and weight of this tire?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["insufficient"] is True
    assert "Insufficient information" in data["answer"]

def test_reason_visual_query():
    buf = create_dummy_image()
    response = client.post(
        "/reason",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"question": "How many plastic bottles and tires are in this image?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "used_detector" in data

# 🌊 AquaGuard: API Usage & Deployment Reference Manual

This document provides complete instructions for executing, evaluating, and integrating the **AquaGuard SeaClear RT-DETR Detection and Reasoning API** (FastAPI backend & Gradio interactive frontend).

---

## 📑 Official Deliverable PDF Documentation

All required screening deliverables have been compiled into vector-grade publication PDFs:
1. [**`AquaGuard_Technical_Memo_and_Audit.pdf`**](file:///c:/Users/nanda/Desktop/RAP/memo/AquaGuard_Technical_Memo_and_Audit.pdf): **Strict 2-page Written Memo & Failure Audit** as required by Deliverable 3.
2. [**`AquaGuard_Complete_System_Documentation.pdf`**](file:///c:/Users/nanda/Desktop/RAP/memo/AquaGuard_Complete_System_Documentation.pdf): Comprehensive 5-page Technical Dossier with architecture diagrams, loss curves, and confusion matrix.
3. [**`AquaGuard_API_Reference_and_Usage_Guide.pdf`**](file:///c:/Users/nanda/Desktop/RAP/memo/AquaGuard_API_Reference_and_Usage_Guide.pdf): Dedicated API Reference manual & Autonomous ROV integration SDK.

---

## 🚀 Quickstart & Server Launch

### 1. Launch FastAPI Backend (Port 8000)
```bash
# Windows PowerShell
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Interactive Landing Dashboard: `http://localhost:8000/`

### 2. Launch Cyber-Marine Web Studio (Port 7860)
```bash
# Launch modernized Gradio Frontend
.\.venv\Scripts\python app_ui.py
```
- Local Studio URL: `http://localhost:7860`
- Live Hugging Face ZeroGPU Space: [https://huggingface.co/spaces/MrPhantom07/aquaguard](https://huggingface.co/spaces/MrPhantom07/aquaguard)

---

## 📡 REST API Endpoints Specification

### 1. `GET /health` — Telemetry & CUDA Check
```bash
curl http://localhost:8000/health
```
**Sample Response:**
```json
{
  "status": "ok",
  "model": "weights/best.pt",
  "cuda": true,
  "classes": [
    "can_metal",
    "bottle_plastic",
    "bottle_glass",
    "net_plastic",
    "bag_plastic",
    "tire_rubber"
  ]
}
```

---

### 2. `POST /detect` — Part A: Object Detection
Accepts an image file and returns detected bounding boxes, class labels, and confidence scores.

```bash
curl -X POST "http://localhost:8000/detect?conf=0.25" \
     -H "Accept: application/json" \
     -F "file=@data/yolo/images/test/slano_0001.jpg"
```
**Sample Response:**
```json
{
  "count": 2,
  "inference_ms": 15.7,
  "model": "weights/best.pt",
  "detections": [
    {
      "class_id": 1,
      "class_name": "bottle_plastic",
      "confidence": 0.935,
      "x1": 142.5,
      "y1": 88.0,
      "x2": 280.1,
      "y2": 310.4
    },
    {
      "class_id": 5,
      "class_name": "tire_rubber",
      "confidence": 0.978,
      "x1": 310.0,
      "y1": 150.2,
      "x2": 490.6,
      "y2": 390.8
    }
  ]
}
```

---

### 3. `POST /reason` — Part B: Framework-Free Natural Language Reasoning

#### Case A: Visual Inquiry (Vision Model Invoked)
```bash
curl -X POST "http://localhost:8000/reason" \
     -F "file=@data/yolo/images/test/slano_0001.jpg" \
     -F "question=How many plastic bottles are in this image?" \
     -F "conf=0.35"
```
**Response:**
```json
{
  "answer": "There is 1 bottle_plastic detected.",
  "used_detector": true,
  "count": 2,
  "insufficient": false,
  "reason": "structured reasoning over detections"
}
```

#### Case B: Non-Visual Inquiry (Intent Routing Bypass — 0 ms, 0 GPU)
```bash
curl -X POST "http://localhost:8000/reason" \
     -F "file=@data/yolo/images/test/slano_0001.jpg" \
     -F "question=Hello! What can you do?"
```
**Response:**
```json
{
  "answer": "I detect marine debris (can_metal, bottle_plastic, bottle_glass, net_plastic, bag_plastic, tire_rubber) and answer questions about counts, presence, and locations.",
  "used_detector": false,
  "count": 0,
  "insufficient": false,
  "reason": "intent routing"
}
```

#### Case C: Unobservable Metadata / Insufficient Info Guardrail
```bash
curl -X POST "http://localhost:8000/reason" \
     -F "file=@data/yolo/images/test/slano_0001.jpg" \
     -F "question=What is the water depth in meters and weight of this tire?"
```
**Response:**
```json
{
  "answer": "Insufficient information: The requested attribute (e.g. depth, weight, brand, or temperature) cannot be determined from visual bounding box detections.",
  "used_detector": true,
  "count": 1,
  "insufficient": true,
  "reason": "confidence guardrail: insufficient to answer"
}
```

---

## 🐍 Python Client SDK Example

```python
import requests

class AquaGuardClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def detect(self, image_path: str, conf: float = 0.25):
        with open(image_path, "rb") as f:
            r = requests.post(f"{self.base_url}/detect", files={"file": f}, params={"conf": conf})
        r.raise_for_status()
        return r.json()

    def reason(self, image_path: str, question: str, conf: float = 0.35):
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/reason",
                files={"file": f},
                data={"question": question, "conf": conf}
            )
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    client = AquaGuardClient()
    res = client.reason("data/yolo/images/test/slano_0001.jpg", "How many bottles are detected?")
    print("Answer:", res["answer"])
```

---

## 🐳 Docker Containerization

```bash
# Build Docker image
docker build -t aquaguard:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name aquaguard aquaguard:latest

# Run automated tests inside container or locally
pytest tests/test_api.py -v
```

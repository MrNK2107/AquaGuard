# AquaGuard API guide

This document provides complete instructions for executing, evaluating, and integrating the **AquaGuard SeaClear RT-DETR Detection and Reasoning API** (FastAPI backend & Gradio interactive frontend).

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

## Authentication and rate limits

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `AQUAGUARD_API_KEYS` | *(unset)* | Comma-separated keys. Unset or empty means auth is off; every response then carries `X-AquaGuard-Auth: disabled`. |
| `AQUAGUARD_RATE_LIMIT` | `60/minute` | Token bucket per key (or per client IP when auth is off). `N/second`, `N/minute`, `N/hour`, or `off`. |
| `AQUAGUARD_MISSIONS_DIR` | `missions` | Where mission folders are stored. |
| `AQUAGUARD_MAX_VIDEO_MB` / `AQUAGUARD_MAX_FRAMES` | `500` / `600` | Upload and sampling caps. |

Send the key as `X-API-Key: <key>` or `Authorization: Bearer <key>`. `/`, `/health`, `/docs` and the `/console` page are public. Their data calls are not.

| Status | When | Headers |
| :--- | :--- | :--- |
| `401` | Missing or wrong key | `WWW-Authenticate: Bearer realm="aquaguard"` |
| `429` | Bucket empty | `Retry-After: <seconds>`, `X-RateLimit-Limit`, `X-RateLimit-Remaining` |

Costs per call, in tokens: `/detect` and `/reason` 1, mission creation 10, HTML report 1, PDF 2, corrections zip 2, CSV/JSONL 0.5, review actions and reads 0.1, frame images 0.05.

## Missions (survey reports and review)

```bash
KEY=deck-1-key
# 1. Start a mission from one video (sampled every 2 s) or many stills
curl -s -X POST http://localhost:8000/api/v1/missions -H "X-API-Key: $KEY"   -F files=@dive.mp4 -F name="Harbour wall T3" -F site="Pier 4" -F every_s=2 -F max_frames=120
# -> 202 {"id": "m-20261103-010203-a1b2c3", "status": "processing", ...}

# 2. Poll until status == "ready"; the inventory is included
curl -s http://localhost:8000/api/v1/missions/m-20261103-010203-a1b2c3 -H "X-API-Key: $KEY"

# 3. Reports
curl -s -H "X-API-Key: $KEY" -o report.pdf  http://localhost:8000/api/v1/missions/<id>/report.pdf
curl -s -H "X-API-Key: $KEY" -o report.html http://localhost:8000/api/v1/missions/<id>/report.html
curl -s -H "X-API-Key: $KEY" -o boxes.csv   http://localhost:8000/api/v1/missions/<id>/detections.csv
```

Form fields for `POST /api/v1/missions`: `files` (required: one video `.mp4/.mov/.avi/.mkv/.webm`, or up to 300 images), `name`, `site`, `operator`, `conf` (auto-count threshold, default 0.50), `floor` (drop below, default 0.25, must be ≤ `conf`), `every_s` (video sampling, default 1.0), `max_frames` (default 120).

Inventory fields: `totals` (`counted`, `auto`, `verified`, `pending`, `rejected`, `added`), `classes[]` (`counted`, `verified`, `pending`, `frames_present`, `peak_in_frame`, `mean_conf`), `estimate` (`low`/`high`, a range for video), `timeline[]` (per frame counts), `hotspots[]`, `review` (`reviewed`, `reviewable`, `pct`).

### Review

```bash
# Most ambiguous first
curl -s -H "X-API-Key: $KEY" "http://localhost:8000/api/v1/missions/<id>/review/queue?limit=20"

# Decide: confirm | reject | relabel (needs class_name) | undo
curl -s -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json"   -d '{"detection_id": "00003-01", "action": "relabel", "class_name": "bottle_glass", "reviewer": "deck-1"}'   http://localhost:8000/api/v1/missions/<id>/review

# Accept the rest of a frame, or add a missed box (stored-frame pixels)
curl -s -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{"reviewer":"deck-1"}'   http://localhost:8000/api/v1/missions/<id>/frames/3/confirm
curl -s -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json"   -d '{"x1": 40, "y1": 60, "x2": 210, "y2": 220, "class_name": "tire_rubber"}'   http://localhost:8000/api/v1/missions/<id>/frames/3/boxes

# Retraining export: corrections.jsonl + YOLO labels/images for fully reviewed frames + data.yaml
curl -s -H "X-API-Key: $KEY" -o corrections.zip http://localhost:8000/api/v1/missions/<id>/corrections.zip
```

Each `corrections.jsonl` line records `box_id`, `bbox_xyxy`, `decision` (`confirmed`/`relabeled`/`rejected`/`added`), `model_class`, `model_confidence`, `final_class`, `reviewer` and `at`.

## Offline CLI

```bash
python scripts/mission_report.py dive.mp4 --name "Harbour wall T3" --every 2 --pdf   # CPU by default
python scripts/mission_report.py stills/ --name "Lokrum stills" --out missions
```

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

"""
AquaGuard: SeaClear RT-DETR High-Throughput REST API
====================================================
Enterprise-grade FastAPI implementation with:
- Multi-worker asynchronous request handling
- Structured JSON telemetry & RFC 7807 error formatting
- Request correlation ID tracing (X-Request-ID)
- Low-latency Part A RT-DETR-L object detection
- Zero-framework Part B cognitive reasoning & guardrails
"""

import io
import time
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import RTDETR

from .reasoning import needs_detector, answer_from_detections, CONF_THRESHOLD

# ------------------------------------------------------------------------------
# 1. Industrial Structured JSON Logging Configuration
# ------------------------------------------------------------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Formats logs as single-line JSON objects for Datadog / Fluentd ingestion."""
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger("aquaguard.api")
logger.setLevel(logging.INFO)

# Console Handler
ch = logging.StreamHandler()
ch.setFormatter(JSONFormatter())
logger.addHandler(ch)

# File Handler (5MB max per file, 3 backups)
fh = RotatingFileHandler(LOGS_DIR / "api.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
fh.setFormatter(JSONFormatter())
logger.addHandler(fh)

# ------------------------------------------------------------------------------
# 2. FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="AquaGuard: SeaClear RT-DETR API",
    description="Underwater Anthropogenic Marine Debris Perception & Framework-Free Cognitive Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for edge web clients & robotics dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 3. Model Weight Resolution & Singleton Caching
# ------------------------------------------------------------------------------
def get_model_path() -> Path:
    candidates = [
        Path("weights/best.pt"),
        Path("runs/detect/train-2/weights/best.pt"),
        Path("runs/detect/train/weights/best.pt"),
        Path("rtdetr-l.pt")
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("rtdetr-l.pt")

MODEL_PATH = get_model_path()
FALLBACK = Path("rtdetr-l.pt")
NAMES = ['can_metal', 'bottle_plastic', 'bottle_glass', 'net_plastic', 'bag_plastic', 'tire_rubber']

_model = None

def get_model():
    global _model
    if _model is None:
        p = str(get_model_path())
        logger.info(f"Initializing RT-DETR-L Neural Engine from: {p}")
        _model = RTDETR(p)
        # Warmup forward pass
        _model.predict(source=Image.new("RGB", (512, 512)), verbose=False)
        logger.info("RT-DETR-L Engine Warmup Complete (Ready for inference)")
    return _model

# ------------------------------------------------------------------------------
# 4. Telemetry Middleware & Correlation ID Tracing
# ------------------------------------------------------------------------------
@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.time()
    
    response = await call_next(request)
    
    latency_ms = (time.time() - t0) * 1000
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Compute-Device"] = "CUDA" if torch.cuda.is_available() else "CPU"
    
    # Structured access log entry
    extra = {
        "request_id": req_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "client_ip": request.client.host if request.client else "unknown"
    }
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({latency_ms:.1f}ms)", extra={"extra_data": extra})
    return response

# ------------------------------------------------------------------------------
# 5. Standardized RFC 7807 Error Handling
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "https://errors.aquaguard.io/http-error",
            "title": "HTTP Exception",
            "status": exc.status_code,
            "detail": exc.detail,
            "instance": request.url.path
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "https://errors.aquaguard.io/validation-error",
            "title": "Payload Validation Failed",
            "status": 422,
            "detail": exc.errors(),
            "instance": request.url.path
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled system exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "type": "https://errors.aquaguard.io/internal-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An internal error occurred during visual perception inference.",
            "instance": request.url.path
        }
    )

# ------------------------------------------------------------------------------
# 6. REST API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AquaGuard: SeaClear RT-DETR API & Perception Engine</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #050b14;
                --card-bg: rgba(13, 23, 42, 0.85);
                --cyan: #00f0ff;
                --emerald: #00ff9d;
                --text: #f1f5f9;
                --muted: #94a3b8;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                margin: 0;
                padding: 40px 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                max-width: 900px;
                width: 100%;
            }
            .header-box {
                background: linear-gradient(135deg, rgba(6, 18, 38, 0.9) 0%, rgba(2, 44, 75, 0.9) 100%);
                border: 1px solid rgba(0, 240, 255, 0.3);
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 24px;
                box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15);
            }
            h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 8px 0; color: #fff; }
            h2 { font-size: 1.1rem; color: var(--cyan); margin: 0 0 16px 0; font-weight: 500; }
            .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
            .badge { background: rgba(0,240,255,0.12); border: 1px solid rgba(0,240,255,0.3); color: #e0f7fa; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 24px; }
            .card {
                background: var(--card-bg);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                padding: 20px;
                transition: transform 0.2s;
            }
            .card:hover { transform: translateY(-3px); border-color: var(--cyan); }
            .card h3 { color: var(--cyan); margin: 0 0 8px 0; font-size: 1.15rem; }
            .card p { color: var(--muted); font-size: 0.88rem; line-height: 1.5; margin: 0 0 14px 0; }
            .btn {
                display: inline-block;
                background: linear-gradient(135deg, #0284c7, #00d2ff);
                color: #050b14;
                text-decoration: none;
                font-weight: 700;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 0.85rem;
            }
            .btn:hover { background: linear-gradient(135deg, #00d2ff, #00ff9d); }
            code { font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.4); padding: 2px 5px; border-radius: 4px; color: #38bdf8; font-size: 0.82rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-box">
                <h1>🌊 AquaGuard: SeaClear RT-DETR API</h1>
                <h2>Underwater Anthropogenic Marine Debris Perception & Framework-Free Cognitive Engine</h2>
                <div class="badges">
                    <span class="badge">RT-DETR-L Hybrid Attention</span>
                    <span class="badge">92.32% mAP@50</span>
                    <span class="badge">15.7 ms GPU Latency</span>
                    <span class="badge">Industrial Multi-Stage Docker</span>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>🔍 Part A: Object Detection</h3>
                    <p>Post underwater images to extract 6 non-COCO debris classes, bounding box coordinates, and confidences.</p>
                    <code>POST /detect</code> or <code>POST /api/v1/detect</code><br><br>
                    <a href="/docs#/default/detect_detect_post" class="btn">Swagger Docs</a>
                </div>

                <div class="card">
                    <h3>🧠 Part B: Cognitive Reasoning</h3>
                    <p>Execute framework-free natural language spatial reasoning, intent routing, and confidence guardrails.</p>
                    <code>POST /reason</code> or <code>POST /api/v1/reason</code><br><br>
                    <a href="/docs#/default/reason_reason_post" class="btn">Swagger Docs</a>
                </div>

                <div class="card">
                    <h3>📊 System Telemetry & Health</h3>
                    <p>Audit loaded model checkpoints, GPU CUDA status, and class definitions.</p>
                    <code>GET /health</code><br><br>
                    <a href="/health" class="btn">Health Check</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
@app.get("/api/v1/health")
def health():
    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "service": "aquaguard-api",
        "version": "2.0.0",
        "model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK),
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "CPU",
        "classes": NAMES
    }

@app.post("/detect")
@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    if file.content_type and not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
        
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image size exceeds 15MB limit.")
        
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image binary could not be decoded.")
        
    t0 = time.time()
    model = get_model()
    results = model.predict(source=img, conf=conf, verbose=False)
    dt = (time.time() - t0) * 1000
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
            detections.append({
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
                "class_id": cid,
                "class_name": cname,
                "confidence": round(float(box.conf[0].item()), 3)
            })
            
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    return {
        "count": len(detections),
        "inference_ms": round(dt, 1),
        "model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK),
        "detections": detections
    }

@app.post("/reason")
@app.post("/api/v1/reason")
async def reason(file: UploadFile = File(...), question: str = Form(...), conf: float = 0.35):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Form field 'question' is required.")
        
    q = question.strip()
    
    # 1. Intent Routing Check (Bypass Detector for Non-Visual Queries)
    if not needs_detector(q):
        if "hello" in q.lower() or "hi" in q.lower():
            answer = "Hello! I am AquaGuard, an autonomous marine debris perception & reasoning assistant. Upload an underwater seabed image and ask about detected litter, counts, spatial coordinates, or debris classifications."
        elif "what can you do" in q.lower() or "who are you" in q.lower():
            answer = "I identify 6 fine-grained non-COCO seabed debris classes (can_metal, bottle_plastic, bottle_glass, net_plastic, bag_plastic, tire_rubber) and perform spatial reasoning over bounding boxes and class distributions."
        else:
            answer = "This question does not require visual perception. The vision detection pipeline was automatically bypassed to conserve GPU compute."
            
        return {
            "answer": answer,
            "used_detector": False,
            "count": 0,
            "detections": [],
            "insufficient": False,
            "reason": "intent routing: non-visual query bypassed"
        }
        
    # 2. Vision Detection Pass
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    model = get_model()
    results = model.predict(source=img, conf=0.20, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
            detections.append({
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
                "class_id": cid,
                "class_name": cname,
                "confidence": float(box.conf[0].item())
            })
            
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    # 3. Grounded Structured Reasoning & Guardrail Evaluation
    answer, insufficient = answer_from_detections(q, detections, conf_thresh=conf)
    
    return {
        "answer": answer,
        "used_detector": True,
        "count": len(detections),
        "insufficient": insufficient,
        "reason": "structured reasoning over detections" if not insufficient else "confidence guardrail: insufficient to answer",
        "detections": detections[:10]
    }

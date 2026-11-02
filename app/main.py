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
import math
import threading
from contextlib import asynccontextmanager
import time
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from .reasoning import needs_detector, answer_from_detections, is_greeting, is_capability_question, CONF_THRESHOLD
from .mission import detections_from_result
from .security import AccessControl, fingerprint

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
@asynccontextmanager
async def lifespan(_app):
    from .missions_api import recover_interrupted  # imported here: missions_api imports this module

    n = recover_interrupted()
    if n:
        logger.info(f"Marked {n} interrupted mission(s) as failed")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="AquaGuard API",
    description=(
        "Debris survey for ROV cleanup crews. `/detect` and `/reason` work on a single frame; "
        "`/api/v1/missions` turns a dive video or a batch of stills into a reviewed debris inventory "
        "with HTML/PDF reports and a retraining export. Send `X-API-Key` when the server has "
        "`AQUAGUARD_API_KEYS` set."
    ),
    version="2.1.0",
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
# One RT-DETR instance serves /detect, /reason and mission processing; ultralytics
# predictors keep per-call state, so every predict (and the first load) goes through this lock.
infer_lock = threading.RLock()


def get_model():
    global _model
    if _model is not None:
        return _model
    with infer_lock:
        if _model is not None:
            return _model
        p = str(get_model_path())
        logger.info(f"Initializing RT-DETR-L Neural Engine from: {p}")
        # Imported lazily so the API module (and its tests) can load without
        # pulling in ultralytics until a prediction is actually needed.
        from ultralytics import RTDETR
        _model = RTDETR(p)
        # Warmup forward pass
        _model.predict(source=Image.new("RGB", (512, 512)), verbose=False)
        logger.info("RT-DETR-L Engine Warmup Complete (Ready for inference)")
    return _model

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def _locked_predict(img, conf):
    with infer_lock:
        return get_model().predict(source=img, conf=conf, verbose=False)


# ------------------------------------------------------------------------------
# Access control: API keys + per-client rate limit (see app/security.py)
# ------------------------------------------------------------------------------
access = AccessControl.from_env()


def require_access(cost: float = 1.0):
    """Dependency factory: authenticate the caller and charge ``cost`` rate-limit tokens."""
    def dependency(request: Request, response: Response):
        ac = access  # module global, so tests and operators can swap it
        key = ac.extract_key(request.headers)
        client = request.client.host if request.client else "unknown"
        if not ac.check_key(key):
            if ac.limiter is not None:  # failed attempts spend the caller's IP bucket, so key guessing is throttled
                d = ac.limiter.hit(f"ip:{client}", 1.0)
                if not d.allowed:
                    raise HTTPException(status_code=429, detail="Too many failed authentication attempts.",
                                        headers={"Retry-After": str(max(1, math.ceil(d.retry_after)))})
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key. Send it as 'X-API-Key: <key>' or 'Authorization: Bearer <key>'.",
                headers={"WWW-Authenticate": 'Bearer realm="aquaguard"'},
            )
        if not ac.enabled:
            response.headers["X-AquaGuard-Auth"] = "disabled"
        ident = f"key:{fingerprint(key)}" if (ac.enabled and key) else f"ip:{client}"
        request.state.client_id = ident
        if ac.limiter is not None:
            d = ac.limiter.hit(ident, cost)
            headers = {"X-RateLimit-Limit": str(d.limit), "X-RateLimit-Remaining": str(max(d.remaining, 0))}
            if not d.allowed:
                headers["Retry-After"] = str(max(1, math.ceil(d.retry_after)))
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Retry after the number of seconds in Retry-After.", headers=headers)
            response.headers.update(headers)
        return ident
    return dependency


async def read_image(file: UploadFile) -> Image.Image:
    """Validate and decode an uploaded image, raising RFC 7807-friendly HTTP errors."""
    if file.content_type and not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image size exceeds 15MB limit.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image binary could not be decoded.")


def extract_detections(results) -> list:
    """Flatten ultralytics Results into JSON-serialisable dicts, sorted by confidence."""
    detections = []
    for r in results:
        detections.extend(detections_from_result(r, NAMES))
    return sorted(detections, key=lambda x: x["confidence"], reverse=True)


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
        headers=getattr(exc, "headers", None),
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
        status_code=422,
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
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse((Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8"))

@app.get("/health")
@app.get("/api/v1/health")
def health():
    cuda = torch.cuda.is_available()
    try:
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
    except (AssertionError, RuntimeError):  # e.g. CUDA_VISIBLE_DEVICES="" on a CUDA build
        cuda, device = False, "CPU"
    return {
        "status": "ok",
        "service": "aquaguard-api",
        "version": app.version,
        "auth": "api-key" if access.enabled else "disabled",
        "model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK),
        "cuda": cuda,
        "device": device,
        "classes": NAMES
    }

@app.post("/detect")
@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.0, le=1.0),
                 _client: str = Depends(require_access(1.0))):
    img = await read_image(file)

    t0 = time.time()
    results = await run_in_threadpool(_locked_predict, img, conf)
    dt = (time.time() - t0) * 1000

    detections = extract_detections(results)
    for d in detections:
        d["confidence"] = round(d["confidence"], 3)
    return {
        "count": len(detections),
        "inference_ms": round(dt, 1),
        "model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK),
        "detections": detections
    }

@app.post("/reason")
@app.post("/api/v1/reason")
async def reason(file: UploadFile = File(...), question: str = Form(...), conf: float = Query(CONF_THRESHOLD, ge=0.0, le=1.0),
                 _client: str = Depends(require_access(1.0))):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Form field 'question' is required.")
        
    q = question.strip()
    
    # 1. Intent Routing Check (Bypass Detector for Non-Visual Queries)
    if not needs_detector(q):
        if is_greeting(q):
            answer = "Hello! I am AquaGuard, an autonomous marine debris perception & reasoning assistant. Upload an underwater seabed image and ask about detected litter, counts, spatial coordinates, or debris classifications."
        elif is_capability_question(q):
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
    img = await read_image(file)
    results = await run_in_threadpool(_locked_predict, img, 0.20)
    detections = extract_detections(results)

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


# ------------------------------------------------------------------------------
# 7. Missions: survey reports + human review (app/missions_api.py)
# ------------------------------------------------------------------------------
from .missions_api import console_router, router as missions_router  # noqa: E402

app.include_router(missions_router)
app.include_router(console_router)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io, time
from pathlib import Path
from PIL import Image
import torch

from ultralytics import RTDETR
from .reasoning import needs_detector, answer_from_detections, CONF_THRESHOLD

app = FastAPI(title="SeaClear RT-DETR API", version="1.0")

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
DATA_YAML = Path("data/yolo/data.yaml")
NAMES = ['can_metal', 'bottle_plastic', 'bottle_glass', 'net_plastic', 'bag_plastic', 'tire_rubber']

_model = None

def get_model():
    global _model
    if _model is None:
        p = str(get_model_path())
        _model = RTDETR(p)
        # Warmup
        _model.predict(source=Image.new("RGB",(640,640)), verbose=False)
    return _model

@app.get("/health")
def health():
    cuda = torch.cuda.is_available()
    return {"status":"ok","model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK),"cuda": cuda, "classes": NAMES}

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    if file.content_type and not file.content_type.startswith("image"):
        raise HTTPException(400, "File must be an image")
    data = await file.read()
    if len(data) > 10*1024*1024:
        raise HTTPException(413, "Image too large (max 10MB)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    t0 = time.time()
    model = get_model()
    results = model.predict(source=img, conf=conf, verbose=False)
    dt = (time.time()-t0)*1000
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
            detections.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "class_id": cid, "class_name": cname,
                "confidence": float(box.conf[0].item())
            })
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    return {"detections": detections, "count": len(detections), "inference_ms": round(dt,1), "model": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK)}

@app.post("/reason")
async def reason(file: UploadFile = File(...), question: str = Form(...), conf: float = 0.35):
    if not question or not question.strip():
        raise HTTPException(400, "question is required")
    q = question.strip()
    if not needs_detector(q):
        # Answer without detector
        if "hello" in q.lower() or "hi" in q.lower():
            return {"answer": "Hello! I am a marine debris detection assistant. Upload an image and ask about objects like bottles, tires, or nets.", "used_detector": False, "detections": [], "insufficient": False, "reason": "intent routing: unrelated to image content"}
        if "what can you do" in q.lower() or "who are you" in q.lower():
            return {"answer": "I detect marine debris (can_metal, bottle_plastic, bottle_glass, net_plastic, bag_plastic, tire_rubber) and answer questions about counts, presence, and locations.", "used_detector": False, "detections": [], "insufficient": False, "reason": "intent routing"}
        return {"answer": "This question does not require image analysis, so no detection was performed.", "used_detector": False, "detections": [], "insufficient": False, "reason": "intent routing: no vision keywords"}
    # Needs detector
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    model = get_model()
    results = model.predict(source=img, conf=0.25, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
            detections.append({"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),"class_id":cid,"class_name":cname,"confidence":float(box.conf[0].item())})
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    answer, insufficient = answer_from_detections(q, detections, conf_thresh=conf)
    return {"answer": answer, "used_detector": True, "detections": detections[:20], "count": len(detections), "insufficient": insufficient, "reason": "structured reasoning over detections" if not insufficient else "confidence guardrail: insufficient to answer"}

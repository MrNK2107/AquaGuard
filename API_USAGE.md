# API Usage

## Run locally
```
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
### GET /health
```
curl http://localhost:8000/health
```
Response:
```json
{"status":"ok","model":"runs/detect/train/weights/best.pt","cuda":false,"classes":["can_metal","bottle_plastic","bottle_glass","net_plastic","bag_plastic","tire_rubber"]}
```

### POST /detect
```
curl -X POST http://localhost:8000/detect -F "file=@data/yolo/images/val/1.jpg" -F "conf=0.25"
```
Response:
```json
{"detections":[{"x1":100,"y1":200,"x2":300,"y2":400,"class_id":2,"class_name":"bottle_glass","confidence":0.82}],"count":1,"inference_ms":120.5}
```

### POST /reason
Vision question:
```
curl -X POST http://localhost:8000/reason -F "file=@data/yolo/images/val/1.jpg" -F "question=How many bottles are there?"
```
Response:
```json
{"answer":"There are 2 bottle_glass(s) detected.","used_detector":true,"detections":[...],"insufficient":false}
```

Non-vision (no detector):
```
curl -X POST http://localhost:8000/reason -F "file=@data/yolo/images/val/1.jpg" -F "question=Hello, what can you do?"
```
Response:
```json
{"answer":"Hello! I am a marine debris detection assistant...","used_detector":false,"insufficient":false}
```

Insufficient guardrail (low conf):
```
curl -X POST http://localhost:8000/reason -F "file=@data/yolo/images/val/empty.jpg" -F "question=Is there a tire?"
```
Response:
```json
{"answer":"Insufficient information: no objects were detected with sufficient confidence...","used_detector":true,"insufficient":true}
```

## Docker
```
docker build -t seaclear-rtdetr .
docker run -p 8000:8000 seaclear-rtdetr
```

from ultralytics import RTDETR
import torch, time, platform, json
from pathlib import Path

DATA = Path("data/yolo/data.yaml")
MODEL = "rtdetr-l.pt"
EPOCHS = 30
IMGSZ = 512
BATCH = 8
DEVICE = 0
WORKERS = 0

print(f"Device cuda: {torch.cuda.is_available()}  torch {torch.__version__}  platform {platform.platform()}")
print(f"Training RT-DETR-L on {DATA}  epochs {EPOCHS} imgsz {IMGSZ} batch {BATCH}")

if __name__ == "__main__":
    model = RTDETR(MODEL)
    start = time.time()
    results = model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=42,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=1e-4,
        weight_decay=1e-4,
        warmup_epochs=3,
        patience=15,
        save=True,
        plots=True,
        val=True,
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed/3600:.2f}h")
    meta = {
        "model": MODEL,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "time_hours": elapsed/3600,
        "hardware": platform.platform() + " RTX 5060 8GB CUDA " + torch.version.cuda,
    }
    Path("runs/detect/train/meta.json").write_text(json.dumps(meta, indent=2))

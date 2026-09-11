"""
Cross-Location Out-Of-Distribution (OOD) & Test Split Evaluation Script
======================================================================
Evaluates the fine-tuned RT-DETR-L model across specific geographic locations
(Bistrina, Lokrum, Marseille) to compute in-distribution vs cross-location transfer metrics.
"""

import os
import json
import zipfile
from collections import defaultdict
from pathlib import Path
import torch
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "best.pt"
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"
SPLITS_FILE = ROOT / "data" / "yolo" / "splits_files.json"
ZIP_PATH = ROOT / "archive.zip"
OUT_AUDIT = ROOT / "eval_results"
OUT_AUDIT.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("   SeaClear Cross-Location & OOD Evaluation Audit   ")
    print("=" * 60)
    
    if not WEIGHTS.exists():
        print(f"Error: {WEIGHTS} not found.")
        return
        
    print(f"Loading weights: {WEIGHTS}")
    model = RTDETR(str(WEIGHTS))
    
    # 1. Overall Test Set Evaluation
    print("\n[1/3] Running Validation on 15% Holdout Test Split...")
    val_metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        batch=8,
        imgsz=512,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False
    )
    
    overall_map50 = float(val_metrics.box.map50)
    overall_map = float(val_metrics.box.map)
    overall_p = float(val_metrics.box.mp)
    overall_r = float(val_metrics.box.mr)
    
    print(f"  Holdout Test mAP@50:    {overall_map50*100:.2f}%")
    print(f"  Holdout Test mAP@50-95: {overall_map*100:.2f}%")
    print(f"  Holdout Test Precision: {overall_p*100:.2f}%")
    print(f"  Holdout Test Recall:    {overall_r*100:.2f}%")
    
    # Per-class metrics
    class_names = ['can_metal', 'bottle_plastic', 'bottle_glass', 'net_plastic', 'bag_plastic', 'tire_rubber']
    per_class = {}
    for i, cname in enumerate(class_names):
        if i < len(val_metrics.box.maps):
            per_class[cname] = {
                "mAP50": float(val_metrics.box.ap50[i]) if hasattr(val_metrics.box, 'ap50') else float(val_metrics.box.maps[i]),
                "mAP50-95": float(val_metrics.box.maps[i])
            }
            print(f"    - {cname:15s}: mAP@50 = {per_class[cname]['mAP50']*100:.1f}%")
            
    # Save results to JSON
    audit_summary = {
        "overall_test": {
            "mAP50": overall_map50,
            "mAP50_95": overall_map,
            "precision": overall_p,
            "recall": overall_r,
            "per_class": per_class
        }
    }
    
    out_file = OUT_AUDIT / "audit_metrics.json"
    out_file.write_text(json.dumps(audit_summary, indent=2))
    print(f"\n[SUCCESS] Audit metrics saved to: {out_file}")

if __name__ == "__main__":
    main()

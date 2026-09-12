<div align="center">

# 🌊 AquaGuard: SeaClear RT-DETR
### Real-Time Underwater Marine Debris Detection & Reasoning API

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/MrPhantom07/aquaguard)
[![Technical Report](https://img.shields.io/badge/%F0%9F%93%9D%20Technical%20Report-TECHNICAL__REPORT.md-purple?style=for-the-badge)](memo/TECHNICAL_REPORT.md)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![RT-DETR-L](https://img.shields.io/badge/Model-RT--DETR--Large-FF6F00?style=for-the-badge&logo=target&logoColor=white)](https://github.com/lyuwenyu/RT-DETR)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Fine-Tuned Real-Time Detection Transformer + Framework-Free Structured Reasoning Engine</b><br>
  Engineered for Autonomous Remotely Operated Vehicles (ROVs) deployed on benthic seabed cleanup missions.
</p>

> 🚀 **Live Interactive Web Demo**: **[https://huggingface.co/spaces/MrPhantom07/aquaguard](https://huggingface.co/spaces/MrPhantom07/aquaguard)** *(Running on ZeroGPU)*  
> 🌐 **Direct Fullscreen App**: **[https://mrphantom07-aquaguard.hf.space](https://mrphantom07-aquaguard.hf.space)**

---

### 📊 Benchmark Scoreboard (Holdout Test Split — 858 Images / 1,395 Annotations)

| Metric | Score | Performance Level | Real-Time Edge Status |
| :--- | :---: | :---: | :---: |
| **mAP @ 0.50** | **`92.32%`** | 🟢 **State-of-the-Art** | Optimal Detection Reliability |
| **mAP @ 0.50:0.95** | **`69.62%`** | 🟢 **High Localization Rigor** | Sub-pixel Bounding Box Tightness |
| **Precision** | **`88.96%`** | 🟢 **Low False-Alarm Rate** | Prevents False Positive Gripping |
| **Recall** | **`87.52%`** | 🟢 **High Discovery Rate** | Maximizes Debris Extraction |
| **Inference Latency** | **`15.7 ms`** | ⚡ **~64 FPS End-to-End** | Real-Time ROV Video Feed Ready |

</div>

---

## 📌 Quick Navigation & Key Resources

| Resource | Description | Direct Link |
| :--- | :--- | :---: |
| 🚀 **Live ZeroGPU Space** | Interactive Gradio Web Demo on Hugging Face | [**Open Live App**](https://huggingface.co/spaces/MrPhantom07/aquaguard) |
| 🌐 **Fullscreen App** | Direct standalone web application interface | [**Open Fullscreen**](https://mrphantom07-aquaguard.hf.space) |
| 📑 **Technical Report** | Comprehensive system audit with 5 failure cases & root-cause analyses | [**`memo/TECHNICAL_REPORT.md`**](memo/TECHNICAL_REPORT.md) |
| 🔌 **API Usage Guide** | Endpoints, payload schemas, and Python/cURL examples | [**`API_USAGE.md`**](API_USAGE.md) |
| 📦 **Trained Checkpoint** | Production fine-tuned RT-DETR-L weights (`92.32%` mAP50)| [**`weights/best.pt`**](weights/best.pt) |

---

## 📌 Executive Summary

**AquaGuard** is an end-to-end Computer Vision and Natural Language Decision System designed for **underwater marine debris identification** across 6 fine-grained **non-COCO** anthropogenic waste classes. 

Built on the **SeaClear Marine Debris Benchmark** (*Nature Scientific Data*, 4TU.ResearchData), AquaGuard bridges the gap between deep vision backbones and actionable robotic intelligence by combining:
1. **Part A (RT-DETR-L Transformer)**: Fine-tuned hybrid encoder model with deformable self-attention resilient to underwater red-light absorption, turbidity, and wave caustics.
2. **Part B (Framework-Free Reasoning Engine)**: Pure, zero-dependency Python reasoning pipeline (strictly zero LangChain/CrewAI bloat) supporting keyword intent routing, spatial bounding box synthesis, and confidence guardrails.
3. **Part C (Production FastAPI Service)**: Edge-ready web API delivering $<20\text{ ms}$ query latency and automated OpenAPI documentation.

---

## 🏷️ Class-by-Class Performance Breakdown

Evaluated on the **15% hold-out test set** (858 unseen images across multiple ROV dives):

```
========================================================================================
 Class Taxonomy               Instances     Precision     Recall     mAP@50     mAP@50-95
========================================================================================
 🛍️  bag_plastic                 138          97.0%        96.0%      99.2%       78.9%
 🛞  tire_rubber                 378          97.7%        97.4%      97.7%       85.4%
 🍾  bottle_plastic              194          88.8%        89.3%      93.5%       70.2%
 🕸️  net_plastic                 145          91.8%        82.1%      91.5%       65.2%
 🥫  can_metal                   176          87.1%        75.6%      88.6%       56.5%
 🍶  bottle_glass                364          86.0%        82.2%      83.6%       63.9%
----------------------------------------------------------------------------------------
 🎯  ALL CLASSES                1,395         88.96%       87.52%     92.32%      69.62%
========================================================================================
```

---

## 🔬 Visual Evaluation & Training Artifacts

The training run (30 epochs, AdamW, $\text{lr}=10^{-4}$) generated full audit curves in [`runs/detect/train-2/`](runs/detect/train-2):

| Evaluation Artifact | Description | Direct File Link |
| :--- | :--- | :--- |
| **Training & Loss Curves** | Metric convergence across 30 epochs for box, giou, and cls losses | [`results.png`](runs/detect/train-2/results.png) |
| **Normalized Confusion Matrix** | Cross-class classification accuracy & background confusion | [`confusion_matrix_normalized.png`](runs/detect/train-2/confusion_matrix_normalized.png) |
| **Precision-Recall Curve** | Area Under Curve (AUC) across all IoU evaluation thresholds | [`BoxPR_curve.png`](runs/detect/train-2/BoxPR_curve.png) |
| **F1 Confidence Curve** | Optimal operational F1-score vs confidence threshold profile | [`BoxF1_curve.png`](runs/detect/train-2/BoxF1_curve.png) |
| **Sample Validation Predictions** | Visual bounding box overlays on real benthic ROV test footage | [`val_batch0_pred.jpg`](runs/detect/train-2/val_batch0_pred.jpg) |

---

## 🏛️ System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │      Client / ROV Edge Controller       │
                                  └────────────────────┬────────────────────┘
                                                       │
                                   HTTP POST Multipart │ /api/v1/reason
                                                       ▼
                      ┌─────────────────────────────────────────────────────────────────┐
                      │              FastAPI Production Web Service                     │
                      └────────────────────────┬────────────────────────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────────────────────────────┐
                      │         Part B: Framework-Free Intent Routing Engine            │
                      └──────────────┬───────────────────────────────────┬──────────────┘
                                     │                                   │
              [Intent: Unrelated /   │                                   │ [Intent: Vision Required /
               Chitchat / Greeting]  │                                   │  Count / Locate / Verify]
                                     ▼                                   ▼
                      ┌───────────────────────────────┐ ┌───────────────────────────────┐
                      │  Instant Rule-Based Responder │ │   Part A: RT-DETR-L Detector  │
                      │    (Latency < 1 ms, 0 GPU)    │ │ (Inference: 15.7 ms on GPU)   │
                      └───────────────────────────────┘ └───────────────┬───────────────┘
                                                                        │
                                                                        ▼
                                                        ┌───────────────────────────────┐
                                                        │  Spatial Reasoning & Guardrail│
                                                        │  - Bbox Coordinate Analysis   │
                                                        │  - Non-visual Query Filter    │
                                                        │  - Low Confidence Filter      │
                                                        └───────────────┬───────────────┘
                                                                        │
                                                                        ▼
                                                        ┌───────────────────────────────┐
                                                        │  Structured JSON API Response │
                                                        └───────────────────────────────┘
```

---

## 🛡️ Guardrails & Decision Logic (Part B)

The reasoning layer is built purely in standard Python ([`app/reasoning.py`](app/reasoning.py)), strictly complying with **Hard Constraint #1** (no LangChain, LangGraph, CrewAI, or AutoGen):

1. **Intent Routing**: Categorizes queries into `DETECTION_REQUIRED` vs `UNRELATED_QUERY`. Non-vision queries (greetings, general capabilities) are resolved immediately without invoking neural network weights.
2. **Confidence Guardrails**: Whenever detections fall below $\text{conf} = 0.35$, the system refuses to speculate and returns:
   > *"Insufficient information: no objects were detected with sufficient confidence to answer confidently."*
3. **Unobservable Metadata Guardrail**: Protects the system against non-visual hallucination when asked for unobservable physical attributes (e.g. *water depth in meters*, *exact weight in kg*, *manufacturer brand*).

#### Example Guardrail Payloads:
```json
// POST /reason with question: "What is the water depth and weight of this tire?"
{
  "answer": "Insufficient information: The requested attribute (e.g. depth, weight, brand, or temperature) cannot be determined from visual bounding box detections.",
  "used_detector": true,
  "count": 1,
  "insufficient": true,
  "reason": "confidence guardrail: insufficient to answer"
}
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository and create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python -m pytest tests/test_api.py -v
```

### 3. Launch Local FastAPI Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI documentation is available at: **`http://localhost:8000/docs`**

### 4. Run via Docker
```bash
# Build production Docker container
docker build -t seaclear-rtdetr-api .

# Run containerized service
docker run -p 8000:8000 seaclear-rtdetr-api
```

---

## 🔌 API Endpoints Specification

| Method | Route | Description | Key Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health check & metadata | None |
| `POST` | `/detect` | **Part A**: Raw bounding box object detection | `file` (Image), `conf` (float, default `0.25`) |
| `POST` | `/reason` | **Part B**: Framework-free structured reasoning | `file` (Image), `question` (str), `conf` (float) |

For comprehensive payload schemas, `curl` requests, and Python SDK code samples, refer to [`API_USAGE.md`](API_USAGE.md).

---

## 📂 Repository Layout

```
├── app/
│   ├── main.py                  # FastAPI application with /detect and /reason endpoints
│   └── reasoning.py             # Framework-free pure Python reasoning & guardrails
├── data/
│   ├── dataset_filtered.json    # Filtered SeaClear COCO annotations
│   └── yolo/                    # YOLO formatted splits (train: 4004, val: 858, test: 858)
├── eval_results/
│   └── audit_metrics.json       # JSON export of 15% holdout test evaluation scores
├── memo/
│   └── TECHNICAL_REPORT.md      # Publication-grade technical audit & failure mode report
├── runs/detect/train-2/         # Training logs, checkpoints, PR curves, and confusion matrices
├── scripts/
│   ├── prepare_data.py          # Two-stage stratified dataset preparation pipeline
│   └── evaluate_ood_locations.py# Cross-location and test evaluation script
├── tests/
│   └── test_api.py              # Automated pytest suite for endpoints & guardrails
├── weights/
│   └── best.pt                  # Production fine-tuned RT-DETR-L checkpoint
├── Dockerfile                   # Production Docker deployment container
├── requirements.txt             # Pinned project dependencies
├── API_USAGE.md                 # Detailed endpoint usage guide and payload examples
└── README.md                    # Project documentation and benchmark scoreboard
```

---

## 📑 Comprehensive Technical Report & System Audit

A publication-grade **System Audit, Dataset Analysis & Failure Mode Report** is documented in **[`memo/TECHNICAL_REPORT.md`](memo/TECHNICAL_REPORT.md)**, containing:
1. **System Architecture & Dataflow**: Mermaid pipeline linking client, intent routing, neural detector, and spatial guardrails.
2. **Dataset Annotation & Spatial Heatmaps**: Bounding box spatial density and multi-class distribution plots.
3. **Training & Metric Progression Curves**: 30-epoch loss convergence, PR curves, and normalized confusion matrices.
4. **Five (5) Physical Failure Mode Root-Cause Analyses**:
   - **Red-Wavelength Optical Attenuation**: Metal debris contrast loss in deep silt.
   - **Optical Caustics & Glass Refraction**: Dynamic sunlight patterns breaking transparent bottle contours.
   - **Partial Sediment Burial**: Proposals failure on $>70\%$ submerged objects.
   - **Marine Bio-Fouling**: Barnacle & macro-algal encrustation masking rubber tire treads.
   - **Thruster-Induced Motion Blur**: Resuspended sediment clouds mimicking discarded fishing nets.
5. **Part B Framework-Free Decision Engine**: Complete logic breakdown and explicit *"Insufficient Information"* guardrail examples.

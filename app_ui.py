"""
AquaGuard: SeaClear RT-DETR Interactive Web Interface (Gradio)
=============================================================
Compatible with Hugging Face Spaces (Free Tier), local development, and public sharing.
Provides interactive interfaces for:
- Part A: RT-DETR Object Detection (bounding boxes, confidences, classes)
- Part B: Framework-Free Natural Language Reasoning & Guardrails
- System Metrics & Failure Mode Analysis
"""

import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import torch
from ultralytics import RTDETR

import sys
app_dir = str(Path(__file__).parent / "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from reasoning import needs_detector, answer_from_detections, is_unanswerable_query

try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(fn):
        return fn

# Resolve model weights
def get_weights_path():
    candidates = [
        Path("weights/best.pt"),
        Path("runs/detect/train-2/weights/best.pt"),
        Path("runs/detect/train/weights/best.pt"),
        Path("rtdetr-l.pt")
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "rtdetr-l.pt"

WEIGHTS_PATH = get_weights_path()
print(f"Loading RT-DETR Model from: {WEIGHTS_PATH}")
model = RTDETR(WEIGHTS_PATH)

# Class color palette for clear contrast on underwater images
CLASS_NAMES = ['can_metal', 'bottle_plastic', 'bottle_glass', 'net_plastic', 'bag_plastic', 'tire_rubber']
CLASS_COLORS = {
    'can_metal': (255, 87, 34),       # Bright Orange
    'bottle_plastic': (33, 150, 243),  # Blue
    'bottle_glass': (0, 188, 212),     # Cyan
    'net_plastic': (255, 235, 59),     # Yellow
    'bag_plastic': (233, 30, 99),      # Pink/Magenta
    'tire_rubber': (76, 175, 80)       # Green
}

def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    """Draws styled bounding boxes and class tags on image."""
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cname = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cname, (255, 255, 0))
        
        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label banner
        label_text = f"{cname} {conf:.2f}"
        # Small background for label
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 8 + 6, max(0, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill=(255, 255, 255))
        
    return img_draw

@gpu_decorator
def run_detection(input_image: Image.Image, conf_threshold: float, iou_threshold: float):
    if input_image is None:
        return None, "Please upload an image.", []
    
    t0 = time.time()
    results = model.predict(source=input_image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    latency_ms = (time.time() - t0) * 1000
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            conf = float(box.conf[0].item())
            detections.append({
                "class_name": cname,
                "confidence": round(conf, 3),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1)
            })
            
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    annotated = annotate_image(input_image, detections)
    
    summary_text = (
        f"**Detections Found**: {len(detections)} object(s)\n\n"
        f"**Inference Latency**: {latency_ms:.1f} ms ({'GPU' if torch.cuda.is_available() else 'CPU'})\n\n"
        f"**Active Checkpoint**: `{Path(WEIGHTS_PATH).name}`"
    )
    
    table_data = [[d["class_name"], d["confidence"], f"[{d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}]"] for d in detections]
    return annotated, summary_text, table_data

@gpu_decorator
def run_reasoning(input_image: Image.Image, question: str, conf_threshold: float):
    if not question or not question.strip():
        return "Please enter a question.", "N/A", "N/A", False
        
    q = question.strip()
    
    # 1. Intent Routing Check
    if not needs_detector(q):
        if "hello" in q.lower() or "hi" in q.lower():
            answer = "Hello! I am AquaGuard, an autonomous marine debris detection and reasoning assistant. Upload an underwater seabed image and ask about detected litter, counts, or locations."
        elif "what can you do" in q.lower() or "who are you" in q.lower():
            answer = "I identify 6 classes of underwater anthropogenic debris (can_metal, bottle_plastic, bottle_glass, net_plastic, bag_plastic, tire_rubber) and perform spatial reasoning over counts and bounding boxes."
        else:
            answer = "This question does not require visual perception. No image detection was invoked."
        return answer, "Bypassed (0 ms, 0 GPU)", "0 objects", False

    # 2. Vision Pipeline
    if input_image is None:
        return "Error: Please upload an image to answer visual queries.", "N/A", "N/A", True
        
    results = model.predict(source=input_image, conf=0.20, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            conf = float(box.conf[0].item())
            detections.append({
                "class_name": cname,
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
            
    # 3. Decision Reasoning & Guardrail Evaluation
    answer, insufficient = answer_from_detections(q, detections, conf_thresh=conf_threshold)
    intent_status = "Visual Reasoning Invoked" if not is_unanswerable_query(q) else "Unanswerable / Non-visual Query Filtered"
    
    return answer, intent_status, f"{len(detections)} candidate object(s)", insufficient

# Find sample images if available
sample_images = []
sample_dir = Path("data/yolo/images/test")
if sample_dir.exists():
    for f in list(sample_dir.glob("*.jpg"))[:4]:
        sample_images.append(str(f))

# Custom Theme Styling
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(title="AquaGuard: SeaClear RT-DETR") as demo:
    gr.Markdown(
        """
        # 🌊 AquaGuard: SeaClear RT-DETR
        ### Underwater Anthropogenic Marine Debris Detection & Reasoning System
        *Fine-tuned **RT-DETR-L** (92.32% mAP@50) + Pure Python Framework-Free Reasoning Engine*
        """
    )
    
    with gr.Tabs():
        # TAB 1: OBJECT DETECTION
        with gr.TabItem("🔍 Part A: Object Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    det_input_img = gr.Image(type="pil", label="Underwater Seabed Image")
                    det_conf_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.35, step=0.05, label="Confidence Threshold")
                    det_iou_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.45, step=0.05, label="NMS IoU Threshold")
                    det_btn = gr.Button("Detect Debris", variant="primary")
                    
                    if sample_images:
                        gr.Examples(examples=sample_images, inputs=det_input_img, label="Sample SeaClear ROV Test Images")
                        
                with gr.Column(scale=1):
                    det_output_img = gr.Image(type="pil", label="Annotated Bounding Boxes")
                    det_summary_md = gr.Markdown(label="Detection Summary")
                    det_table = gr.Dataframe(
                        headers=["Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"],
                        label="Structured Detections Table"
                    )
                    
            det_btn.click(
                fn=run_detection,
                inputs=[det_input_img, det_conf_slider, det_iou_slider],
                outputs=[det_output_img, det_summary_md, det_table]
            )

        # TAB 2: VISUAL REASONING & GUARDRAILS
        with gr.TabItem("🧠 Part B: Reasoning & Guardrails"):
            with gr.Row():
                with gr.Column(scale=1):
                    reas_input_img = gr.Image(type="pil", label="Input Image")
                    reas_question = gr.Textbox(
                        lines=2,
                        placeholder="e.g. How many plastic bottles are in this image? OR What is the water depth?",
                        label="Natural Language Question"
                    )
                    reas_conf_slider = gr.Slider(minimum=0.10, maximum=0.80, value=0.35, step=0.05, label="Guardrail Confidence Threshold")
                    reas_btn = gr.Button("Submit Question", variant="primary")
                    
                    gr.Markdown("#### 💡 Try Sample Prompts:")
                    ex_btn1 = gr.Button("1. 'How many plastic bottles are in this image?'", size="sm")
                    ex_btn2 = gr.Button("2. 'Is there a rubber tire present?'", size="sm")
                    ex_btn3 = gr.Button("3. 'Where are the objects located?'", size="sm")
                    ex_btn4 = gr.Button("4. 'What is the water depth and weight of this tire?' (Guardrail Test)", size="sm")
                    ex_btn5 = gr.Button("5. 'Hello, what can you do?' (Intent Routing Bypass)", size="sm")
                    
                with gr.Column(scale=1):
                    reas_answer = gr.Textbox(lines=4, label="System Answer", interactive=False)
                    reas_intent = gr.Textbox(label="Intent Router Status", interactive=False)
                    reas_count = gr.Textbox(label="Detected Candidates", interactive=False)
                    reas_guardrail_flag = gr.Checkbox(label="Guardrail Triggered (Insufficient Info)", interactive=False)
                    
            reas_btn.click(
                fn=run_reasoning,
                inputs=[reas_input_img, reas_question, reas_conf_slider],
                outputs=[reas_answer, reas_intent, reas_count, reas_guardrail_flag]
            )
            
            ex_btn1.click(lambda: "How many plastic bottles are in this image?", None, reas_question)
            ex_btn2.click(lambda: "Is there a rubber tire present?", None, reas_question)
            ex_btn3.click(lambda: "Where are the objects located?", None, reas_question)
            ex_btn4.click(lambda: "What is the exact water depth in meters and weight of this tire?", None, reas_question)
            ex_btn5.click(lambda: "Hello, what can you do?", None, reas_question)

        # TAB 3: BENCHMARK & SYSTEM AUDIT
        with gr.TabItem("📊 Part C: Benchmark & Audit"):
            gr.Markdown(
                """
                ### 🏆 SeaClear Benchmark Scoreboard (15% Hold-out Test Set)
                
                | Metric | Overall Performance |
                | :--- | :---: |
                | **mAP@50** | **`92.32%`** |
                | **mAP@50-95** | **`69.62%`** |
                | **Precision** | **`88.96%`** |
                | **Recall** | **`87.52%`** |
                | **Inference Latency** | **`15.7 ms`** (~64 FPS) |
                
                ---
                
                ### 📦 Per-Class Holdout Test Scores
                
                | Class | Instances | mAP@50 | mAP@50-95 |
                | :--- | :---: | :---: | :---: |
                | `bag_plastic` | 138 | **99.2%** | 78.9% |
                | `tire_rubber` | 378 | **97.7%** | 85.4% |
                | `bottle_plastic` | 194 | **93.5%** | 70.2% |
                | `net_plastic` | 145 | **91.5%** | 65.2% |
                | `can_metal` | 176 | **88.6%** | 56.5% |
                | `bottle_glass` | 364 | **83.6%** | 63.9% |
                
                ---
                
                ### 🛡️ Architecture & Guardrails Principles
                - **No Heavy Frameworks**: Pure standard Python logic for intent routing and spatial reasoning.
                - **Physical Optical Nuances**: Documented red-light water attenuation, optical caustics on transparent glass, and bio-fouling degradation.
                """
            )

if __name__ == "__main__":
    # Launch locally on port 7860
    demo.launch(theme=theme, server_name="0.0.0.0", server_port=7860, share=False)

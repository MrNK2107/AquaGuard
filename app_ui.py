"""
AquaGuard: SeaClear RT-DETR Perception & Reasoning Studio
==========================================================
Inspired by clean full-width HF perception studios.
Supports optional visual question answering & dynamic UI customization.
Pure Python reasoning layer without external agent frameworks.
"""

import os
import time
import json
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

# Model Weights Resolution
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

CLASS_NAMES = ['can_metal', 'bottle_plastic', 'bottle_glass', 'net_plastic', 'bag_plastic', 'tire_rubber']
CLASS_COLORS = {
    'can_metal': (255, 107, 0),       # Coral Orange
    'bottle_plastic': (0, 195, 255),  # Cyan Blue
    'bottle_glass': (46, 204, 113),   # Mint Green
    'net_plastic': (241, 196, 15),    # Gold Yellow
    'bag_plastic': (231, 76, 60),     # Red Coral
    'tire_rubber': (155, 89, 182)     # Purple Violet
}

def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    """Draws sleek bounding boxes with readable tags matching the legend."""
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
        
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cname = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cname, (0, 200, 255))
        
        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Pill banner
        label_text = f" {cname} {conf*100:.1f}% "
        bbox = draw.textbbox((x1, max(0, y1 - 20)), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, max(0, y1 - 20)), label_text, fill=(10, 15, 30), font=font)
        
    return img_draw


@gpu_decorator
def run_detect(image: Image.Image, conf_threshold: float):
    if image is None:
        return None, "", [], {}

    t0 = time.time()
    results = model.predict(source=image, conf=conf_threshold, iou=0.45, verbose=False)
    latency_ms = (time.time() - t0) * 1000

    detections = []
    class_counts = {}
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
                "confidence_pct": f"{conf * 100:.1f}%",
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "box_coords": f"[{round(x1, 1)}, {round(y1, 1)}, {round(x2, 1)}, {round(y2, 1)}]"
            })
            class_counts[cname] = class_counts.get(cname, 0) + 1

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    annotated = annotate_image(image, detections)

    count_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()]) if class_counts else "None"
    status_pills = f"""
    <div class="pills">
        <span class="pill pill-ok">🟢 {len(detections)} object(s) detected</span>
        <span class="pill pill-info">⚡ {latency_ms:.1f} ms latency</span>
        <span class="pill pill-muted">📦 {count_str}</span>
    </div>
    """

    table_data = [[d["class_name"], d["confidence_pct"], d["box_coords"]] for d in detections]
    
    raw_json = {
        "status": "success",
        "objects_count": len(detections),
        "latency_ms": round(latency_ms, 1),
        "detections": detections
    }

    return annotated, status_pills, table_data, raw_json


@gpu_decorator
def run_ask(image: Image.Image, question: str, conf_threshold: float):
    q = question.strip() if question else ""
    
    # 1. Non-visual inquiry bypass check
    if q and not needs_detector(q):
        if "hello" in q.lower() or "hi" in q.lower():
            ans = "Hello! I am AquaGuard, an autonomous underwater marine debris perception & reasoning system. Ask me about detected litter, counts, spatial positions, or classifications."
        elif "what can you do" in q.lower() or "who are you" in q.lower():
            ans = "I detect 6 non-COCO seabed debris classes (can_metal, bottle_plastic, bottle_glass, net_plastic, bag_plastic, tire_rubber) and perform grounded spatial reasoning without external agent frameworks."
        else:
            ans = "This query does not require visual perception. The vision detection pipeline was bypassed to conserve edge GPU compute."
            
        status_pills = """
        <div class="pills">
            <span class="pill pill-info">⚡ Bypassed Detector (0 ms, 0 GPU)</span>
            <span class="pill pill-ok">🟢 Direct Response</span>
        </div>
        """
        raw_json = {
            "status": "success",
            "question": q,
            "answer": ans,
            "used_detector": False,
            "insufficient": False
        }
        return image, ans, status_pills, [], raw_json

    if image is None:
        return None, "Please upload an underwater image or select a hold-out test sample.", "", [], {}

    # 2. Neural Detection
    t0 = time.time()
    results = model.predict(source=image, conf=conf_threshold, iou=0.45, verbose=False)
    latency_ms = (time.time() - t0) * 1000

    detections = []
    class_counts = {}
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
                "confidence_pct": f"{conf * 100:.1f}%",
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "box_coords": f"[{round(x1, 1)}, {round(y1, 1)}, {round(x2, 1)}, {round(y2, 1)}]"
            })
            class_counts[cname] = class_counts.get(cname, 0) + 1

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    annotated = annotate_image(image, detections)

    # 3. Grounded Reasoning
    if q:
        answer, insufficient = answer_from_detections(q, detections, conf_thresh=conf_threshold)
    else:
        if detections:
            tally_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
            answer = f"Detected {len(detections)} debris item(s) on the seabed ({tally_str}). Ask any question to perform spatial reasoning over counts, locations, or debris types."
        else:
            answer = "No debris objects detected at the current confidence threshold."
        insufficient = False

    if insufficient:
        guard_pill = '<span class="pill pill-warn">🛡️ Guardrail Refusal (Insufficient Info)</span>'
    else:
        guard_pill = '<span class="pill pill-ok">🟢 Grounded Reasoning OK</span>'

    status_pills = f"""
    <div class="pills">
        {guard_pill}
        <span class="pill pill-info">⚡ {latency_ms:.1f} ms</span>
        <span class="pill pill-muted">🎯 {len(detections)} objects</span>
    </div>
    """

    table_data = [[d["class_name"], d["confidence_pct"], d["box_coords"]] for d in detections]

    raw_json = {
        "status": "success",
        "question": q if q else None,
        "answer": answer,
        "objects_count": len(detections),
        "latency_ms": round(latency_ms, 1),
        "insufficient": insufficient,
        "detections": detections
    }

    return annotated, answer, status_pills, table_data, raw_json


def toggle_qa_visibility(enabled: bool):
    """Dynamically shows or hides the Question & Answer controls."""
    return gr.update(visible=enabled), gr.update(visible=enabled)


# Curated clear sample test images covering all 6 non-COCO marine classes
sample_images = []
for p in [
    "data/yolo/images/test/110.jpg",
    "data/yolo/images/test/1876.jpg",
    "data/yolo/images/test/431.jpg",
    "data/yolo/images/test/1946.jpg",
    "data/yolo/images/test/1360.jpg",
    "data/yolo/images/test/Cam1_16_26_03_10_11_2020.mp4_00248.jpg"
]:
    if Path(p).exists():
        sample_images.append(p)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS & STYLING (Inspiration from HF PPE RT-DETR space)
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
:root {
    --bg-dark: #090e17;
    --card-dark: #111827;
    --border-dark: #1f293d;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --accent-orange: #ff6b00;
    --accent-cyan: #00c3ff;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 96% !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding: 18px 0 32px !important;
}

/* Header typography */
.hero {
    padding: 6px 0 16px;
}
.hero h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.02em;
    color: #ffffff;
}
.hero p {
    margin: 0;
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.55;
}
.hero p b {
    color: #ffffff;
}
.hero a {
    color: var(--accent-orange);
    text-decoration: none;
    font-weight: 600;
}
.hero a:hover {
    text-decoration: underline;
}

/* Orange/Coral Label Badges */
.gr-form, .block {
    background: transparent !important;
    border: none !important;
}

label span, .label-wrap span {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Category Legend */
.legend {
    font-size: 0.85rem;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    padding: 8px 2px 8px;
    color: #d1d5db;
}
.legend span {
    display: inline-flex;
    align-items: center;
}
.sw {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-right: 6px;
}
.sw-can { background: #ff6b00; }
.sw-bplastic { background: #00c3ff; }
.sw-bglass { background: #2ecc71; }
.sw-net { background: #f1c40f; }
.sw-bag { background: #e74c3c; }
.sw-tire { background: #9b59b6; }

/* Status Pills */
.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 6px 0;
}
.pill {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.4;
    border: 1px solid transparent;
}
.pill-ok {
    background: #123322;
    color: #7ddba3;
    border-color: #1f5236;
}
.pill-warn {
    background: #3a2f10;
    color: #f2cd6a;
    border-color: #5c4a1a;
}
.pill-info {
    background: #16263d;
    color: #8fb8ee;
    border-color: #234270;
}
.pill-muted {
    background: #1e293b;
    color: #94a3b8;
    border-color: #334155;
}

/* Answer box styling */
#answer textarea {
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
    background: #111827 !important;
    border: 1px solid #1f293d !important;
    color: #f9fafb !important;
    border-radius: 8px !important;
}

/* Primary and Secondary buttons */
button.primary-btn {
    background: #ff6b00 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
button.secondary-btn {
    background: #1f2937 !important;
    color: #e5e7eb !important;
    font-weight: 600 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}

.toggle-wrap {
    margin: 8px 0 4px;
    padding: 4px 0;
}

/* Footer */
.foot {
    opacity: 0.7;
    font-size: 0.85rem;
    text-align: center;
    padding-top: 18px;
    margin-top: 20px;
    border-top: 1px solid #1f293d;
    color: #9ca3af;
}

footer {
    display: none !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="sm",
).set(
    body_background_fill="#090e17",
    body_background_fill_dark="#090e17",
    block_background_fill="#101725",
    block_background_fill_dark="#101725",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    input_background_fill="#0c1322",
    input_background_fill_dark="#0c1322",
    input_border_color="#1e293b",
    input_border_color_dark="#1e293b",
)

# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AquaGuard · Marine Debris Perception & Reasoning",
    analytics_enabled=False,
) as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>Underwater Debris Perception & Reasoning</h1>
            <p>
                RT-DETR fine-tuned for <b>can_metal</b>, <b>bottle_plastic</b>, <b>bottle_glass</b>, <b>net_plastic</b>, <b>bag_plastic</b>, and <b>tire_rubber</b> on seabed imagery, with a hand-written reasoning layer that answers questions from the detections and <b>refuses when they cannot support an answer</b>.
                &nbsp;·&nbsp; <a href="/docs" target="_blank">API docs</a>
                &nbsp;·&nbsp; <a href="memo/AquaGuard_Technical_Memo_and_Audit.pdf" target="_blank">memo</a>
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ───────────────── LEFT COLUMN: INPUTS ─────────────────
        with gr.Column(scale=5, min_width=340):
            input_image = gr.Image(
                label="Seabed image",
                sources=["upload", "webcam", "clipboard"],
                type="pil",
                height=360,
            )

            if sample_images:
                gr.Examples(
                    examples=sample_images,
                    inputs=input_image,
                    label="Held-out test images (never trained on)",
                    examples_per_page=6,
                )

            slider_conf = gr.Slider(
                minimum=0.05,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Confidence threshold",
                info="boxes below this are dropped; the guardrail floor is 0.30",
            )

            detect_btn = gr.Button(
                "Detect objects",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"],
            )

            enable_qa = gr.Checkbox(
                label="Enable question answering & spatial reasoning",
                value=True,
                elem_classes=["toggle-wrap"],
            )

            # Conditional Question Input Area
            with gr.Column(visible=True) as qa_input_group:
                input_question = gr.Textbox(
                    label="Ask a question about the image",
                    placeholder="e.g. How many plastic bottles are in this image?",
                    value="How many plastic bottles are in this image?",
                    lines=2,
                    max_lines=3,
                )

                try_these = gr.Examples(
                    examples=[
                        ["How many plastic bottles are in this image?"],
                        ["How many debris objects are in this image?"],
                        ["Where are the detected objects located?"],
                        ["What is the exact water depth in meters and weight of this tire?"],
                        ["Hello! What can you do?"],
                    ],
                    inputs=input_question,
                    label="Try these",
                    examples_per_page=5,
                )

                ask_btn = gr.Button(
                    "Ask",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

        # ───────────────── RIGHT COLUMN: OUTPUTS ─────────────────
        with gr.Column(scale=6, min_width=340):
            output_image = gr.Image(
                label="Detections",
                type="pil",
                interactive=False,
                height=360,
            )

            gr.HTML(
                """
                <div class="legend">
                    <span><i class="sw sw-can"></i>can_metal</span>
                    <span><i class="sw sw-bplastic"></i>bottle_plastic</span>
                    <span><i class="sw sw-bglass"></i>bottle_glass</span>
                    <span><i class="sw sw-net"></i>net_plastic</span>
                    <span><i class="sw sw-bag"></i>bag_plastic</span>
                    <span><i class="sw sw-tire"></i>tire_rubber</span>
                </div>
                """
            )

            output_pills = gr.HTML()

            # Conditional Answer Box Area
            output_answer = gr.Textbox(
                label="Answer",
                lines=3,
                max_lines=4,
                interactive=False,
                elem_id="answer",
                visible=True,
            )

            with gr.Accordion("Structured evidence (detections)", open=False):
                output_table = gr.Dataframe(
                    headers=["Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"],
                    interactive=False,
                )

            with gr.Accordion("Raw response (same fields as the API)", open=False):
                output_json = gr.JSON(
                    label="",
                )

    gr.HTML(
        """
        <div class="foot">
            Same model and code as <code>POST /detect</code> and <code>POST /reason</code>. Pure Python rule-based reasoning engine with zero external agent frameworks.
        </div>
        """
    )

    # ───────────────── Event Wiring ─────────────────

    # Toggle Q&A visibility dynamically
    enable_qa.change(
        fn=toggle_qa_visibility,
        inputs=[enable_qa],
        outputs=[qa_input_group, output_answer],
    )

    # 1. Detect objects button: runs detection only
    detect_btn.click(
        fn=run_detect,
        inputs=[input_image, slider_conf],
        outputs=[output_image, output_pills, output_table, output_json],
        show_progress="full",
    )

    # 2. Ask button / Textbox submit: runs detection + grounded question answering
    ask_btn.click(
        fn=run_ask,
        inputs=[input_image, input_question, slider_conf],
        outputs=[output_image, output_answer, output_pills, output_table, output_json],
        show_progress="full",
    )

    input_question.submit(
        fn=run_ask,
        inputs=[input_image, input_question, slider_conf],
        outputs=[output_image, output_answer, output_pills, output_table, output_json],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=CUSTOM_CSS,
    )

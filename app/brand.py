"""AquaGuard identity: one place for the palette, class colours and box drawing.

The class colours are the Okabe-Ito colour-blind-safe set, so the six debris
classes stay distinguishable on a sunlit deck screen and in a printed report.
"""

from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont

PRODUCT = "AquaGuard"
TAGLINE = "Debris survey for ROV cleanup crews"

# Core palette (also mirrored as CSS custom properties in the console + report).
PALETTE = {
    "ink": "#0B1D2A",      # text, chart axes
    "deep": "#0E3A4F",     # header band, primary surfaces
    "paper": "#F6F4EF",    # page background (print friendly)
    "line": "#D9D4C7",     # hairlines
    "muted": "#5B6B75",    # secondary text
    "signal": "#FF5F1F",   # the one accent: actions, "needs review"
    "ok": "#1E7D57",       # verified
}

CLASS_NAMES = ["can_metal", "bottle_plastic", "bottle_glass", "net_plastic", "bag_plastic", "tire_rubber"]

CLASS_COLORS_HEX: Dict[str, str] = {
    "can_metal": "#E69F00",
    "bottle_plastic": "#56B4E9",
    "bottle_glass": "#009E73",
    "net_plastic": "#CC79A7",
    "bag_plastic": "#D55E00",
    "tire_rubber": "#0072B2",
}

CLASS_LABELS: Dict[str, str] = {
    "can_metal": "Metal can",
    "bottle_plastic": "Plastic bottle",
    "bottle_glass": "Glass bottle",
    "net_plastic": "Net / rope",
    "bag_plastic": "Plastic bag",
    "tire_rubber": "Tyre",
}


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def class_color(name: str) -> str:
    return CLASS_COLORS_HEX.get(name, PALETTE["signal"])


def _font(size: int):
    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(image: Image.Image, detections: Iterable[dict], width: int = 3) -> Image.Image:
    """Draw class-coloured boxes with a short label; returns a new RGB image."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = _font(max(12, out.width // 45))
    for det in detections:
        color = hex_to_rgb(class_color(det.get("class_name", "")))
        box = [det["x1"], det["y1"], det["x2"], det["y2"]]
        draw.rectangle(box, outline=color, width=width)
        conf = det.get("confidence")
        score = "added" if conf is None else f"{conf * 100:.0f}%"
        label = f" {CLASS_LABELS.get(det.get('class_name', ''), det.get('class_name', '?'))} {score} "
        ty = max(0, det["y1"] - font.size - 6) if hasattr(font, "size") else max(0, det["y1"] - 16)
        tb = draw.textbbox((det["x1"], ty), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((det["x1"], ty), label, fill=(11, 29, 42), font=font)
    return out

import re
from typing import List, Dict, Tuple

UNRELATED_PATTERNS = [
    r"hello", r"hi\b", r"how are you", r"what is your name", r"who are you",
    r"what can you do", r"tell me a joke", r"weather", r"capital of"
]

UNANSWERABLE_PATTERNS = [
    r"\bdepth\b", r"\bweight\b", r"\bhow heavy\b", r"\bhow deep\b",
    r"\bbrand\b", r"\bmanufacturer\b", r"\btemperature\b", r"\bage of\b",
    r"\bhow old\b", r"\bowner\b", r"\bcost\b", r"\bprice\b"
]

VISION_KEYWORDS = [
    "how many", "count", "number of", "is there", "are there", "detect",
    "where", "locate", "most common", "what.*object", "visible", "present",
    "can you see",
    r"\bcan_metal\b", r"\btire_rubber\b", r"\bbottle_plastic\b", r"\bbottle_glass\b", r"\bnet_plastic\b", r"\bbag_plastic\b",
    r"\btire\b", r"\bbottle\b", r"\bnet\b", r"\bbag\b", r"\bglass\b", r"\bplastic\b", r"\bmetal\b", r"\brubber\b"
]

CONF_THRESHOLD = 0.35
LOW_CONF_THRESHOLD = 0.30

def is_unanswerable_query(question: str) -> bool:
    q = question.lower()
    return any(re.search(pat, q) for pat in UNANSWERABLE_PATTERNS)

def needs_detector(question: str) -> bool:
    q = question.lower()
    if is_unanswerable_query(q):
        return True
    for pat in UNRELATED_PATTERNS:
        if re.search(pat, q):
            if not any(re.search(v, q) for v in VISION_KEYWORDS):
                return False
    if any(re.search(v, q) for v in VISION_KEYWORDS):
        return True
    if re.search(r"\b(how many|count|detect|find|locate|object|thing)\b", q):
        return True
    return False

def answer_from_detections(question: str, detections: List[Dict], conf_thresh: float = CONF_THRESHOLD) -> Tuple[str, bool]:
    q = question.lower().strip()
    if is_unanswerable_query(q):
        return "Insufficient information: The requested attribute (e.g. depth, weight, brand, or temperature) cannot be determined from visual bounding box detections.", True
    if not detections:
        return "Insufficient information: no objects were detected with sufficient confidence to answer confidently.", True
    # Filter by confidence
    high = [d for d in detections if d["confidence"] >= conf_thresh]
    if not high:
        maxc = max(d["confidence"] for d in detections)
        return f"Insufficient information: highest detection confidence {maxc:.2f} is below threshold {conf_thresh:.2f}, cannot answer confidently.", True

    counts = {}
    for d in high:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    total = len(high)

    # How many
    if re.search(r"how many|count|number of", q):
        # Check for specific class mention
        for cls in counts:
            if cls.replace("_"," ") in q or cls.split("_")[0] in q:
                return f"There are {counts[cls]} {cls}(s) detected.", False
        # Generic count
        detail = ", ".join(f"{c}: {n}" for c,n in counts.items())
        return f"There are {total} objects detected in total ({detail}).", False

    if re.search(r"is there|are there|present|visible", q):
        for cls in counts:
            if cls.replace("_"," ") in q or cls.split("_")[0] in q:
                if counts[cls] > 0:
                    return f"Yes, {cls} is present ({counts[cls]} detected).", False
                else:
                    return f"No, {cls} was not detected with sufficient confidence.", False
        # generic presence
        return f"Yes, objects are present: {', '.join(counts.keys())}.", False

    if re.search(r"most common|most frequent", q):
        most = max(counts, key=counts.get)
        return f"The most common object is {most} with {counts[most]} instances.", False

    if re.search(r"where|locate|position|bounding", q):
        locs = [f"{d['class_name']} at [{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}] (conf {d['confidence']:.2f})" for d in high[:5]]
        return "Detected locations: " + "; ".join(locs) + (f" and {len(high)-5} more." if len(high)>5 else ""), False

    if re.search(r"what.*object|what is in", q):
        detail = ", ".join(f"{c} ({n})" for c,n in counts.items())
        return f"The image contains: {detail}.", False

    # Fallback generic
    detail = ", ".join(f"{c}: {n}" for c,n in counts.items())
    return f"Detected {total} objects: {detail}.", False

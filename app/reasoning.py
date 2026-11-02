import re
from typing import List, Dict, Tuple

UNRELATED_PATTERNS = [
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"how are you", r"what is your name", r"who are you",
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

CLASS_NAMES = ["can_metal", "bottle_plastic", "bottle_glass", "net_plastic", "bag_plastic", "tire_rubber"]

# Object nouns (singular + plural) -> candidate classes
_NOUNS = {
    "can": ["can_metal"], "cans": ["can_metal"],
    "bottle": ["bottle_plastic", "bottle_glass"], "bottles": ["bottle_plastic", "bottle_glass"],
    "net": ["net_plastic"], "nets": ["net_plastic"],
    "bag": ["bag_plastic"], "bags": ["bag_plastic"],
    "tire": ["tire_rubber"], "tires": ["tire_rubber"], "tyre": ["tire_rubber"], "tyres": ["tire_rubber"],
}
# Material adjectives -> classes made of that material
_MATERIALS = {
    "metal": ["can_metal"], "metallic": ["can_metal"], "aluminum": ["can_metal"], "aluminium": ["can_metal"], "tin": ["can_metal"],
    "plastic": ["bottle_plastic", "net_plastic", "bag_plastic"],
    "glass": ["bottle_glass"],
    "rubber": ["tire_rubber"],
}

_MODAL_FOLLOWERS = {"you", "i", "we", "they", "it", "he", "she", "be", "see", "find", "detect", "spot", "tell", "identify", "not", "t"}


def _is_modal_can(tokens: List[str], i: int) -> bool:
    """True when 'can' is the auxiliary verb ("can you see...") rather than the object."""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    return i == 0 or nxt in _MODAL_FOLLOWERS


def mentioned_classes(question: str) -> List[str]:
    """Return the debris classes a question refers to, in canonical order.

    Object nouns ("bottles", "tires") select classes; material words
    ("plastic", "glass") narrow a noun's candidates, or select every class of
    that material when no noun is present. Exact class ids (e.g. ``bottle_glass``)
    always match. Returns [] when no class is mentioned.
    """
    q = question.lower()
    exact = {c for c in CLASS_NAMES if c in q}
    tokens = re.findall(r"[a-z]+", q.replace("_", " "))
    noun_hits = [
        set(_NOUNS[t]) for i, t in enumerate(tokens)
        if t in _NOUNS and not (t == "can" and _is_modal_can(tokens, i))
    ]
    material_hits = set()
    for t in tokens:
        material_hits.update(_MATERIALS.get(t, []))

    found = set(exact)
    if noun_hits:
        for cands in noun_hits:
            narrowed = cands & material_hits
            found |= narrowed if narrowed else cands
    elif not exact:
        found |= material_hits
    return [c for c in CLASS_NAMES if c in found]


GREETING_RE = re.compile(r"\b(hello|hi|hey)\b")
CAPABILITY_RE = re.compile(r"\b(what can you do|who are you)\b")


def is_greeting(question: str) -> bool:
    """Whole-word greeting check: "hi" matches "hi there", not "sushi" or "this"."""
    return bool(GREETING_RE.search(question.lower()))


def is_capability_question(question: str) -> bool:
    return bool(CAPABILITY_RE.search(question.lower()))


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
    asked = mentioned_classes(q)

    # How many
    if re.search(r"how many|count|number of", q):
        if asked:
            if len(asked) == 1:
                cls = asked[0]
                return f"There are {counts.get(cls, 0)} {cls}(s) detected.", False
            detail = ", ".join(f"{c}: {counts.get(c, 0)}" for c in asked)
            n = sum(counts.get(c, 0) for c in asked)
            return f"There are {n} matching objects detected ({detail}).", False
        detail = ", ".join(f"{c}: {n}" for c, n in counts.items())
        return f"There are {total} objects detected in total ({detail}).", False

    if re.search(r"is there|are there|present|visible", q):
        if asked:
            parts = []
            for cls in asked:
                if counts.get(cls, 0) > 0:
                    parts.append(f"Yes, {cls} is present ({counts[cls]} detected).")
                else:
                    parts.append(f"No, {cls} was not detected with sufficient confidence.")
            return " ".join(parts), False
        return f"Yes, objects are present: {', '.join(counts.keys())}.", False

    if re.search(r"most common|most frequent", q):
        most = max(counts, key=counts.get)
        return f"The most common object is {most} with {counts[most]} instances.", False

    if re.search(r"where|locate|position|bounding", q):
        pool = [d for d in high if d["class_name"] in asked] if asked else high
        if not pool:
            return f"No {', '.join(asked)} was detected with sufficient confidence, so it cannot be located.", False
        locs = [f"{d['class_name']} at [{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}] (conf {d['confidence']:.2f})" for d in pool[:5]]
        return "Detected locations: " + "; ".join(locs) + (f" and {len(pool)-5} more." if len(pool) > 5 else ""), False

    if re.search(r"what.*object|what is in", q):
        detail = ", ".join(f"{c} ({n})" for c, n in counts.items())
        return f"The image contains: {detail}.", False

    # Fallback generic
    detail = ", ".join(f"{c}: {n}" for c, n in counts.items())
    return f"Detected {total} objects: {detail}.", False

"""Unit tests for the framework-free reasoning layer (app/reasoning.py)."""

import pytest

from app.reasoning import (
    CONF_THRESHOLD,
    answer_from_detections,
    is_unanswerable_query,
    mentioned_classes,
    needs_detector,
)


def det(cls, conf=0.9, box=(10, 20, 110, 220)):
    x1, y1, x2, y2 = box
    return {"class_name": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


SCENE = [det("can_metal", 0.92), det("can_metal", 0.81), det("bottle_glass", 0.77), det("bag_plastic", 0.66)]


# --- is_unanswerable_query -----------------------------------------------------

@pytest.mark.parametrize("q", [
    "How deep is this?", "What is the weight of the tire?", "Which brand is the can?",
    "What's the water temperature?", "How old is this bottle?", "What is the price?",
])
def test_unanswerable_queries(q):
    assert is_unanswerable_query(q)


@pytest.mark.parametrize("q", ["How many cans?", "Where is the net?", "Is there a bag?"])
def test_answerable_queries(q):
    assert not is_unanswerable_query(q)


# --- needs_detector (intent routing) -------------------------------------------

@pytest.mark.parametrize("q", [
    "Hello", "hi there", "What can you do?", "Tell me a joke", "What is the capital of France?",
    "What's the weather like?",
])
def test_non_visual_queries_bypass_detector(q):
    assert needs_detector(q) is False


@pytest.mark.parametrize("q", [
    "How many tires are there?", "Where is the plastic bag?", "Is there a fishing net?",
    "Count the objects", "Hello, how many bottles can you see?",
    "How deep is the water?",  # unanswerable still routes to detector -> guardrail
])
def test_visual_queries_use_detector(q):
    assert needs_detector(q) is True


def test_greeting_patterns_use_word_boundaries():
    # "which" contains "hi" but is not a greeting; the vision keyword must win.
    assert needs_detector("which bottle is closest?") is True
    assert needs_detector("sushi") is False


# --- mentioned_classes -----------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("How many tires?", ["tire_rubber"]),
    ("Any tyres here?", ["tire_rubber"]),
    ("How many plastic bottles?", ["bottle_plastic"]),
    ("How many glass bottles?", ["bottle_glass"]),
    ("How many bottles?", ["bottle_plastic", "bottle_glass"]),
    ("Is there a bottle_plastic?", ["bottle_plastic"]),
    ("Count plastic items", ["bottle_plastic", "net_plastic", "bag_plastic"]),
    ("Anything made of glass?", ["bottle_glass"]),
    ("How many cans?", ["can_metal"]),
    ("Can you see any bottles?", ["bottle_plastic", "bottle_glass"]),
    ("What can you detect?", []),
    ("How many plastic bottles and tires?", ["bottle_plastic", "tire_rubber"]),
    ("How many objects?", []),
])
def test_mentioned_classes(q, expected):
    assert mentioned_classes(q) == expected


# --- answer_from_detections --------------------------------------------------------

def test_unanswerable_guardrail_wins_even_with_detections():
    ans, insufficient = answer_from_detections("What is the weight of the can?", SCENE)
    assert insufficient is True
    assert ans.startswith("Insufficient information")


def test_no_detections_is_insufficient():
    ans, insufficient = answer_from_detections("How many cans?", [])
    assert insufficient is True
    assert "no objects" in ans


def test_low_confidence_is_insufficient():
    ans, insufficient = answer_from_detections("How many cans?", [det("can_metal", 0.2)])
    assert insufficient is True
    assert "0.20" in ans and f"{CONF_THRESHOLD:.2f}" in ans


def test_custom_threshold_is_respected():
    _, insufficient = answer_from_detections("How many cans?", [det("can_metal", 0.5)], conf_thresh=0.6)
    assert insufficient is True
    ans, insufficient = answer_from_detections("How many cans?", [det("can_metal", 0.5)], conf_thresh=0.4)
    assert insufficient is False and "1 can_metal" in ans


def test_count_specific_class():
    ans, insufficient = answer_from_detections("How many cans are there?", SCENE)
    assert insufficient is False
    assert ans == "There are 2 can_metal(s) detected."


def test_count_absent_class_reports_zero():
    # Regression: previously fell through to the total count of all objects.
    ans, _ = answer_from_detections("How many tires are there?", SCENE)
    assert ans == "There are 0 tire_rubber(s) detected."


def test_count_disambiguates_bottle_material():
    # Regression: "plastic bottles" used to match bottle_glass via the "bottle" prefix.
    ans, _ = answer_from_detections("How many plastic bottles?", SCENE)
    assert ans == "There are 0 bottle_plastic(s) detected."


def test_count_multiple_classes():
    ans, _ = answer_from_detections("How many bottles and cans?", SCENE)
    assert "can_metal: 2" in ans and "bottle_glass: 1" in ans and "bottle_plastic: 0" in ans
    assert ans.startswith("There are 3 matching objects")


def test_count_total_ignores_low_confidence():
    scene = SCENE + [det("tire_rubber", 0.1)]
    ans, _ = answer_from_detections("How many objects?", scene)
    assert ans.startswith("There are 4 objects detected in total")
    assert "tire_rubber" not in ans


def test_presence_yes():
    ans, insufficient = answer_from_detections("Is there a glass bottle?", SCENE)
    assert insufficient is False
    assert ans == "Yes, bottle_glass is present (1 detected)."


def test_presence_no_for_undetected_class():
    # Regression: previously answered "Yes, objects are present" for any class.
    ans, _ = answer_from_detections("Is there a tire?", SCENE)
    assert ans == "No, tire_rubber was not detected with sufficient confidence."


def test_presence_generic():
    ans, _ = answer_from_detections("Are there any objects visible?", SCENE)
    assert ans.startswith("Yes, objects are present:")


def test_most_common():
    ans, _ = answer_from_detections("What is the most common debris?", SCENE)
    assert ans == "The most common object is can_metal with 2 instances."


def test_locations_filtered_by_class():
    ans, _ = answer_from_detections("Where is the plastic bag?", SCENE)
    assert ans.startswith("Detected locations: bag_plastic at [10,20,110,220]")
    assert "can_metal" not in ans


def test_locations_truncated_after_five():
    scene = [det("can_metal", 0.9)] * 7
    ans, _ = answer_from_detections("Where are the objects?", scene)
    assert ans.count("can_metal at") == 5
    assert ans.endswith("and 2 more.")


def test_locations_for_absent_class():
    ans, insufficient = answer_from_detections("Where is the net?", SCENE)
    assert insufficient is False
    assert "net_plastic" in ans and "cannot be located" in ans


def test_what_objects():
    ans, _ = answer_from_detections("What objects are in the image?", SCENE)
    assert "can_metal (2)" in ans


def test_fallback_summary():
    ans, insufficient = answer_from_detections("Describe the seabed debris", SCENE)
    assert insufficient is False
    assert ans.startswith("Detected 4 objects:")


@pytest.mark.parametrize("q", ["hi", "Hello!", "hey there"])
def test_is_greeting_whole_words(q):
    from app.reasoning import is_greeting
    assert is_greeting(q)


@pytest.mark.parametrize("q", ["I like sushi", "this one", "they said", "chill"])
def test_is_greeting_ignores_substrings(q):
    from app.reasoning import is_greeting
    assert not is_greeting(q)


def test_is_capability_question():
    from app.reasoning import is_capability_question
    assert is_capability_question("Who are you?")
    assert is_capability_question("so what can you do")
    assert not is_capability_question("how many cans")

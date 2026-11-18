"""Gesture vocabulary shared by training, evaluation and the app."""

# HaGRID class name -> text shown to the student
GESTURE_TEXT = {
    "like": "Thumbs up 👍",
    "peace": "Peace sign ✌️",
    "palm": "Open palm ✋",
    "fist": "Fist ✊",
    "ok": "OK sign 👌",
    "one": "One finger ☝️",
    # extra HaGRID classes, in case you change the vocabulary
    "dislike": "Thumbs down 👎",
    "call": "Call me 🤙",
    "rock": "Rock 🤘",
    "three": "Three fingers",
    "four": "Four fingers",
    "stop": "Stop (palm forward)",
    "mute": "Finger on lips 🤫",
}

DEFAULT_GESTURES = ["like", "peace", "palm", "fist", "ok", "one"]
FALLBACK_GESTURES = ["like", "peace", "palm", "fist"]

NONE_LABEL = "none"  # prediction when no gesture is detected


def gesture_text(name: str) -> str:
    return GESTURE_TEXT.get(name, name)

"""Pretrained MediaPipe Gesture Recognizer as a gesture "detector" (no training needed).

MediaPipe finds each hand, classifies its gesture with Google's canned classifier, and returns
21 hand landmarks; the box is the landmark extent. Canned classes are mapped to our HaGRID names:

    Thumb_Up -> like      Victory -> peace      Open_Palm -> palm
    Closed_Fist -> fist   Pointing_Up -> one    Thumb_Down -> dislike   ILoveYou -> rock
There is no "OK" class, so the default challenge set is [like, peace, palm, fist, one].
"""
from __future__ import annotations

import atexit
import threading

import cv2
import numpy as np

from .downloads import CACHE_DIR, _download
from .types import Box, Detection

GESTURE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                     "gesture_recognizer/float16/1/gesture_recognizer.task")

MP_TO_OURS = {"Thumb_Up": "like", "Victory": "peace", "Open_Palm": "palm", "Closed_Fist": "fist",
              "Pointing_Up": "one", "Thumb_Down": "dislike", "ILoveYou": "rock"}
SUPPORTED = sorted(set(MP_TO_OURS.values()))


def gesture_model(path=None):
    if path:
        return path
    dst = CACHE_DIR / "gesture_recognizer.task"
    return dst if dst.exists() else _download(GESTURE_MODEL_URL, dst)


class MediaPipeGestureDetector:
    def __init__(self, model_path=None, min_conf: float = 0.5, num_hands: int = 4,
                 min_hand_conf: float = 0.4, pad: float = 0.15):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(gesture_model(model_path))),
            num_hands=num_hands, min_hand_detection_confidence=min_hand_conf,
            min_hand_presence_confidence=min_hand_conf, min_tracking_confidence=min_hand_conf)
        self._rec = vision.GestureRecognizer.create_from_options(opts)
        self.min_conf, self.pad = min_conf, pad
        try:
            threading._register_atexit(self.close)
        except Exception:
            atexit.register(self.close)

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        res = self._rec.recognize(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out = []
        for gest, lms in zip(res.gestures, res.hand_landmarks):
            if not gest:
                continue
            top = gest[0]
            label = MP_TO_OURS.get(top.category_name)
            if label is None or top.score < self.min_conf:  # "None" = hand without a known gesture
                continue
            xs = np.array([p.x for p in lms]) * w
            ys = np.array([p.y for p in lms]) * h
            bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            box = Box(xs.min() - self.pad * bw, ys.min() - self.pad * bh,
                      xs.max() + self.pad * bw, ys.max() + self.pad * bh).clip(w, h)
            out.append(Detection(box, label, float(top.score)))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

    def close(self):
        if getattr(self, "_rec", None) is not None:
            try:
                self._rec.close()
            except Exception:
                pass
            self._rec = None

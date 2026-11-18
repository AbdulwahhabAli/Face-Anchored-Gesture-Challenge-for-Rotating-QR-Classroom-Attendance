"""Baseline (full-frame) vs proposed (face-anchored ROI cascade) gesture pipelines.

Both pipelines use the SAME gesture detector weights; only the inference input differs.
The predicted gesture of an image is the class of its highest-confidence detection
("none" if nothing is detected).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..gestures import NONE_LABEL
from .roi import crop_for_detector, face_roi
from .types import Box, Detection, Face


class GestureDetector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> list[Detection]: ...


class FaceDetector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> list[Face]: ...


@dataclass
class GestureResult:
    label: str                      # top-1 gesture or "none"
    conf: float
    detections: list[Detection]     # all detections, full-image coordinates, sorted by conf
    face_box: Box | None = None     # anchor face (ROI pipeline only)
    roi: Box | None = None
    used_fallback: bool = False     # ROI pipeline found no face and fell back to the full frame
    timings_ms: dict = field(default_factory=dict)


def _top1(dets: list[Detection]) -> tuple[str, float]:
    if not dets:
        return NONE_LABEL, 0.0
    best = max(dets, key=lambda d: d.conf)
    return best.label, best.conf


class FullFramePipeline:
    """Baseline: gesture detector on the full selfie."""
    name = "full"

    def __init__(self, gesture_detector: GestureDetector):
        self.gesture_detector = gesture_detector

    def predict(self, img: np.ndarray, faces: list[Face] | None = None) -> GestureResult:
        t0 = time.perf_counter()
        dets = self.gesture_detector.detect(img)
        t1 = time.perf_counter()
        label, conf = _top1(dets)
        return GestureResult(label, conf, dets, timings_ms={"gesture": (t1 - t0) * 1e3})


class FaceAnchoredROIPipeline:
    """Ours: face detector -> ROI = scale x face box -> gesture detector on the (up-scaled) crop."""

    def __init__(self, gesture_detector: GestureDetector, face_detector: FaceDetector, scale: float = 3.0,
                 upscale: bool = True, y_shift: float = 0.0, fallback_full_frame: bool = True, imgsz: int = 640,
                 fallback_on_no_gesture: bool = False):
        self.gesture_detector = gesture_detector
        self.face_detector = face_detector
        self.scale, self.upscale, self.y_shift = scale, upscale, y_shift
        self.fallback_full_frame = fallback_full_frame
        self.fallback_on_no_gesture = fallback_on_no_gesture  # app robustness; off for the controlled comparison
        self.imgsz = imgsz

    @property
    def name(self) -> str:
        return f"roi_s{self.scale:g}_{'up' if self.upscale else 'noup'}"

    def predict(self, img: np.ndarray, faces: list[Face] | None = None) -> GestureResult:
        """`faces` may be passed in (pre-computed) so ablations do not re-run the face detector."""
        timings = {}
        t0 = time.perf_counter()
        if faces is None:
            faces = self.face_detector.detect(img)
        timings["face"] = (time.perf_counter() - t0) * 1e3
        face = max(faces, key=lambda f: f.box.area) if faces else None

        if face is None:
            if not self.fallback_full_frame:
                return GestureResult(NONE_LABEL, 0.0, [], timings_ms=timings)
            t1 = time.perf_counter()
            dets = self.gesture_detector.detect(img)
            timings["gesture"] = (time.perf_counter() - t1) * 1e3
            label, conf = _top1(dets)
            return GestureResult(label, conf, dets, used_fallback=True, timings_ms=timings)

        h, w = img.shape[:2]
        t1 = time.perf_counter()
        roi = face_roi(face.box, w, h, self.scale, self.y_shift)
        crop, tf = crop_for_detector(img, roi, upscale=self.upscale, size=self.imgsz)
        timings["crop"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        raw = self.gesture_detector.detect(crop)
        timings["gesture"] = (time.perf_counter() - t2) * 1e3
        dets = [Detection(tf.to_full(d.box), d.label, d.conf) for d in raw]
        dets = [d for d in dets if d.box.area > 0]
        used_fallback = False
        if not dets and self.fallback_on_no_gesture:  # hand outside the ROI (e.g. held at chest level)
            t3 = time.perf_counter()
            dets = self.gesture_detector.detect(img)
            timings["gesture_fallback"] = (time.perf_counter() - t3) * 1e3
            used_fallback = True
        label, conf = _top1(dets)
        return GestureResult(label, conf, dets, face_box=face.box, roi=roi, used_fallback=used_fallback,
                             timings_ms=timings)

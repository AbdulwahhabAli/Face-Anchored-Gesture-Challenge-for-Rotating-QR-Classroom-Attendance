"""Accept/reject decision for one check-in selfie.

accept  <=>  (i) detected gesture == challenged gesture   (liveness / challenge-response)
        and (ii) the main face matches the claimed student's enrolled ArcFace embedding.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import resolve_path
from .gestures import NONE_LABEL
from .vision.face_recognizer import cosine
from .vision.pipelines import FaceAnchoredROIPipeline, FullFramePipeline, GestureResult


@dataclass
class Decision:
    accepted: bool
    reason: str                       # ok | enroll | no_gesture | wrong_gesture | no_face | face_mismatch
    gesture_pred: str
    gesture_conf: float
    face_sim: float | None
    embedding: np.ndarray | None = field(default=None, repr=False)
    timings_ms: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason, "gesture_pred": self.gesture_pred,
                "gesture_conf": round(self.gesture_conf, 3),
                "face_sim": None if self.face_sim is None else round(self.face_sim, 3),
                "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()}}


class Verifier:
    def __init__(self, gesture_pipeline, recognizer, face_threshold: float = 0.35):
        self.pipeline = gesture_pipeline
        self.recognizer = recognizer
        self.face_threshold = face_threshold

    def verify(self, img: np.ndarray, challenge: str, enrolled: np.ndarray | None) -> Decision:
        """enrolled=None means first check-in: the face embedding is returned for enrollment."""
        t0 = time.perf_counter()
        g: GestureResult = self.pipeline.predict(img)
        timings = dict(g.timings_ms)

        t1 = time.perf_counter()
        emb, _face = self.recognizer.embed_main(img, hint=g.face_box)
        timings["arcface"] = (time.perf_counter() - t1) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3

        if g.label == NONE_LABEL:
            return Decision(False, "no_gesture", g.label, g.conf, None, emb, timings)
        if g.label != challenge:
            return Decision(False, "wrong_gesture", g.label, g.conf, None, emb, timings)
        if emb is None:
            return Decision(False, "no_face", g.label, g.conf, None, None, timings)
        if enrolled is None:
            return Decision(True, "enroll", g.label, g.conf, None, emb, timings)
        sim = cosine(emb, enrolled)
        if sim < self.face_threshold:
            return Decision(False, "face_mismatch", g.label, g.conf, sim, emb, timings)
        return Decision(True, "ok", g.label, g.conf, sim, emb, timings)


def build_face_anchor(cfg: dict, recognizer=None):
    """Face detector that anchors the ROI: BlazeFace (proposal) or SCRFD (more robust to small faces)."""
    m = cfg["models"]
    kind = cfg["pipeline"].get("face_anchor", "blazeface")
    if kind == "scrfd":
        if recognizer is None:
            from .vision.face_recognizer import ArcFaceRecognizer
            recognizer = ArcFaceRecognizer(resolve_path(m.get("arcface_dir")))
        return recognizer
    from .vision.face_detector import BlazeFaceDetector
    return BlazeFaceDetector(resolve_path(m.get("blazeface_model")), m.get("blazeface_min_conf", 0.5))


def build_gesture_detector(cfg: dict, weights=None):
    """'mediapipe' = Google's pretrained recognizer (no training); 'yolo' = YOLOv8n fine-tuned on HaGRID."""
    m = cfg["models"]
    backend = "yolo" if weights else m.get("gesture_backend", "mediapipe")
    if backend == "mediapipe":
        from .vision.mediapipe_gesture import MediaPipeGestureDetector
        return MediaPipeGestureDetector(resolve_path(m.get("mediapipe_gesture_model")),
                                        min_conf=m.get("gesture_min_conf", 0.5))
    if backend == "yolo":
        from .vision.gesture_detector import YoloGestureDetector
        w = resolve_path(weights or m["yolo_weights"])
        if not w.exists():
            raise FileNotFoundError(f"YOLO weights {w} not found - train with scripts/train_yolo.py "
                                    "or set models.gesture_backend: mediapipe")
        return YoloGestureDetector(w, m.get("device", "cpu"), m.get("imgsz", 640), m.get("yolo_conf", 0.25))
    raise ValueError(f"unknown gesture_backend {backend!r}")


def build_gesture_pipeline(cfg: dict, gesture_detector=None, face_detector=None, recognizer=None):
    m, p = cfg["models"], cfg["pipeline"]
    if gesture_detector is None:
        gesture_detector = build_gesture_detector(cfg)
    if p.get("mode", "roi") == "full":
        return FullFramePipeline(gesture_detector)
    if face_detector is None:
        face_detector = build_face_anchor(cfg, recognizer)
    return FaceAnchoredROIPipeline(gesture_detector, face_detector, scale=p.get("roi_scale", 3.0),
                                   upscale=p.get("roi_upscale", True), y_shift=p.get("roi_y_shift", 0.0),
                                   fallback_full_frame=p.get("fallback_full_frame", True),
                                   fallback_on_no_gesture=p.get("fallback_on_no_gesture", False),
                                   imgsz=m.get("imgsz", 640))


def build_verifier(cfg: dict) -> Verifier:
    from .vision.face_recognizer import ArcFaceRecognizer
    recognizer = ArcFaceRecognizer(resolve_path(cfg["models"].get("arcface_dir")))
    pipeline = build_gesture_pipeline(cfg, recognizer=recognizer)
    return Verifier(pipeline, recognizer, cfg["verification"]["face_threshold"])

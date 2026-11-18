"""BlazeFace (MediaPipe Tasks) face detector used to anchor the gesture ROI."""
from __future__ import annotations

import atexit
import threading

import cv2
import numpy as np

from .downloads import blazeface_model
from .types import Box, Face


class BlazeFaceDetector:
    def __init__(self, model_path=None, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(blazeface_model(model_path))),
            min_detection_confidence=min_conf,
        )
        self._det = vision.FaceDetector.create_from_options(opts)
        # close before MediaPipe's worker threads shut down (avoids a noisy __del__ traceback at exit)
        try:
            threading._register_atexit(self.close)
        except Exception:
            atexit.register(self.close)

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        h, w = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        faces = []
        for d in res.detections:
            bb = d.bounding_box
            box = Box(bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height).clip(w, h)
            if box.area <= 0:
                continue
            score = d.categories[0].score if d.categories else 0.0
            faces.append(Face(box=box, score=float(score)))
        return faces

    def close(self):
        if self._det is not None:
            try:
                self._det.close()
            except Exception:
                pass
            self._det = None


def main_face(faces: list[Face]) -> Face | None:
    """The selfie-taker's face = the largest detected face."""
    return max(faces, key=lambda f: f.box.area) if faces else None

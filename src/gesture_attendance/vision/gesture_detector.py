"""YOLOv8 gesture detector wrapper (same fine-tuned weights for baseline and ROI pipelines)."""
from __future__ import annotations

import numpy as np

from .types import Box, Detection


class YoloGestureDetector:
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640, conf: float = 0.25,
                 class_names: list[str] | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.imgsz, self.conf = device, imgsz, conf
        names = self.model.names  # {id: name}
        self.names = [names[i] for i in sorted(names)] if class_names is None else list(class_names)

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, conf):
            out.append(Detection(Box(*map(float, b)), self.names[c], float(s)))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

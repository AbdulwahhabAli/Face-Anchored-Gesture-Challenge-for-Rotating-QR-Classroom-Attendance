from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def shift(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def scale(self, s: float) -> "Box":
        return Box(self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)

    def clip(self, w: float, h: float) -> "Box":
        return Box(min(max(self.x1, 0), w), min(max(self.y1, 0), h),
                   min(max(self.x2, 0), w), min(max(self.y2, 0), h))

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_list(self) -> list[float]:
        return [round(float(v), 1) for v in (self.x1, self.y1, self.x2, self.y2)]

    @staticmethod
    def from_list(v) -> "Box":
        return Box(*map(float, v))


@dataclass
class Detection:
    """A gesture detection."""
    box: Box
    label: str
    conf: float

    def to_dict(self) -> dict:
        return {"box": self.box.as_list(), "label": self.label, "conf": round(float(self.conf), 4)}

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(Box.from_list(d["box"]), d["label"], float(d["conf"]))


@dataclass
class Face:
    """A detected face; kps = 5 landmarks (eyes, nose, mouth corners) when available."""
    box: Box
    score: float
    kps: np.ndarray | None = field(default=None, repr=False)

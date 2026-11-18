"""Face-anchored region of interest (the contribution of the project).

The ROI is a box of `scale` x the face box, centred on the face (optionally shifted down),
clipped to the image. It is then either handed to the detector as-is (the detector letterboxes
it up to its input size -> "up-scaling"), or pasted at native resolution on a neutral canvas of
the detector input size ("no up-scaling" ablation).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Box


def face_roi(face: Box, img_w: int, img_h: int, scale: float = 3.0, y_shift: float = 0.0) -> Box:
    """ROI = face box scaled by `scale` around its centre, shifted down by y_shift*face_h, clipped."""
    cx = face.cx
    cy = face.cy + y_shift * face.h
    half_w = face.w * scale / 2
    half_h = face.h * scale / 2
    roi = Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h).clip(img_w, img_h)
    return Box(int(np.floor(roi.x1)), int(np.floor(roi.y1)), int(np.ceil(roi.x2)), int(np.ceil(roi.y2)))


@dataclass
class CropTransform:
    """Maps detector-input coordinates back to full-image coordinates: full = det / s + offset."""
    offset_x: float
    offset_y: float
    s: float = 1.0
    valid: Box | None = None  # region of the detector input that holds real pixels

    def to_full(self, b: Box) -> Box:
        if self.valid is not None:
            b = Box(max(b.x1, self.valid.x1), max(b.y1, self.valid.y1),
                    min(b.x2, self.valid.x2), min(b.y2, self.valid.y2))
        return b.scale(1.0 / self.s).shift(self.offset_x, self.offset_y)


def crop_for_detector(img: np.ndarray, roi: Box, upscale: bool = True, size: int = 640,
                      pad_value: int = 114) -> tuple[np.ndarray, CropTransform]:
    """Cut the ROI out of `img` and prepare it for a detector with input size `size`.

    upscale=True : return the raw crop; Ultralytics letterboxes it to `size` (small crops get enlarged).
    upscale=False: paste the crop at 1:1 pixel scale on a size x size canvas so it is NOT enlarged
                   (crops larger than `size` are shrunk to fit, as the detector would do anyway).
    """
    x1, y1, x2, y2 = int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"empty ROI {roi}")
    if upscale:
        return np.ascontiguousarray(crop), CropTransform(x1, y1, 1.0)

    h, w = crop.shape[:2]
    s = min(1.0, size / max(h, w))
    if s < 1.0:
        crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
    canvas = np.full((size, size, 3), pad_value, dtype=img.dtype)
    canvas[:h, :w] = crop
    return canvas, CropTransform(x1, y1, s, valid=Box(0, 0, w, h))

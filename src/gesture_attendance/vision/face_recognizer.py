"""ArcFace face verification, running InsightFace's ONNX models directly with onnxruntime.

* SCRFD-500MF (det_500m.onnx) finds faces + 5 landmarks (needed for ArcFace alignment)
* ArcFace MobileFaceNet (w600k_mbf.onnx) gives a 512-d L2-normalised embedding

This re-implements the small part of the `insightface` package we need, so the project
installs on Windows without a C++ compiler. Decoding follows insightface/model_zoo/scrfd.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .downloads import arcface_dir
from .types import Box, Face

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _session(path: Path):
    import onnxruntime as ort
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), providers=providers)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    def __init__(self, model_path: Path, input_size=(640, 640), det_thresh=0.5, nms_thresh=0.4):
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.batched = len(self.session.get_outputs()[0].shape) == 3
        n = len(self.output_names)
        if n == 9:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, True
        elif n == 6:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, False
        elif n == 15:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, True
        else:
            raise ValueError(f"unsupported SCRFD model with {n} outputs")
        self.input_size = input_size
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh
        self._centers: dict = {}

    def _forward(self, img: np.ndarray):
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        ih, iw = blob.shape[2], blob.shape[3]
        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            take = (lambda a: a[0]) if self.batched else (lambda a: a)
            scores = take(outs[idx]).reshape(-1)
            bbox = take(outs[idx + self.fmc]).reshape(-1, 4) * stride
            fh, fw = ih // stride, iw // stride
            key = (fh, fw, stride)
            centers = self._centers.get(key)
            if centers is None:
                centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                if self.num_anchors > 1:
                    centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
                self._centers[key] = centers
            pos = np.where(scores >= self.det_thresh)[0]
            boxes = np.stack([centers[:, 0] - bbox[:, 0], centers[:, 1] - bbox[:, 1],
                              centers[:, 0] + bbox[:, 2], centers[:, 1] + bbox[:, 3]], axis=-1)
            scores_l.append(scores[pos])
            boxes_l.append(boxes[pos])
            if self.use_kps:
                kp = take(outs[idx + self.fmc * 2]).reshape(-1, 10) * stride
                pts = np.empty_like(kp)
                pts[:, 0::2] = centers[:, :1] + kp[:, 0::2]
                pts[:, 1::2] = centers[:, 1:2] + kp[:, 1::2]
                kps_l.append(pts[pos].reshape(-1, 5, 2))
        return scores_l, boxes_l, kps_l

    def detect(self, img: np.ndarray) -> list[Face]:
        iw, ih = self.input_size
        im_ratio = img.shape[0] / img.shape[1]
        if im_ratio > ih / iw:
            nh, nw = ih, int(ih / im_ratio)
        else:
            nw, nh = iw, int(iw * im_ratio)
        det_scale = nh / img.shape[0]
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        scores_l, boxes_l, kps_l = self._forward(canvas)
        scores = np.concatenate(scores_l)
        if scores.size == 0:
            return []
        boxes = np.concatenate(boxes_l) / det_scale
        kpss = np.concatenate(kps_l) / det_scale if self.use_kps else None
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_thresh)
        h, w = img.shape[:2]
        return [Face(Box(*map(float, dets[i, :4])).clip(w, h), float(dets[i, 4]), None if kpss is None else kpss[i])
                for i in keep]


class ArcFaceRecognizer:
    """detect() faces with landmarks; embed() an aligned face -> 512-d unit vector."""

    def __init__(self, model_dir=None, det_thresh: float = 0.5):
        d = arcface_dir(model_dir)
        self.detector = SCRFD(d / "det_500m.onnx", det_thresh=det_thresh)
        self.rec = _session(d / "w600k_mbf.onnx")
        self.rec_input = self.rec.get_inputs()[0].name

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        return self.detector.detect(img_bgr)

    @staticmethod
    def align(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST * (size / 112.0), method=cv2.LMEDS)
        return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        if face.kps is None:
            raise ValueError("face has no landmarks; use a face from ArcFaceRecognizer.detect()")
        chip = self.align(img_bgr, face.kps)
        blob = cv2.dnn.blobFromImage(chip, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.rec.run(None, {self.rec_input: blob})[0][0].astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)

    def embed_main(self, img_bgr: np.ndarray, hint: Box | None = None) -> tuple[np.ndarray | None, Face | None]:
        """Embedding of the main face: the one overlapping `hint` best (e.g. the BlazeFace box), else the largest."""
        faces = self.detect(img_bgr)
        if not faces:
            return None, None
        if hint is not None:
            best = max(faces, key=lambda f: f.box.iou(hint))
            if best.box.iou(hint) < 0.1:
                best = max(faces, key=lambda f: f.box.area)
        else:
            best = max(faces, key=lambda f: f.box.area)
        return self.embed(img_bgr, best), best


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

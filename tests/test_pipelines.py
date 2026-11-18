import numpy as np

from gesture_attendance.verify import Verifier
from gesture_attendance.vision.pipelines import FaceAnchoredROIPipeline, FullFramePipeline, GestureResult
from gesture_attendance.vision.types import Box, Detection, Face


class FakeYolo:
    """Returns one detection at fixed coordinates of whatever image it is given; records input shapes."""
    def __init__(self, label="like", box=Box(10, 10, 50, 50), conf=0.9):
        self.label, self.box, self.conf, self.shapes = label, box, conf, []

    def detect(self, img):
        self.shapes.append(img.shape)
        return [Detection(self.box, self.label, self.conf), Detection(self.box, "fist", 0.3)]


class FakeFaces:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, img):
        return self.faces


IMG = np.zeros((1000, 800, 3), np.uint8)


def test_full_frame_top1():
    r = FullFramePipeline(FakeYolo()).predict(IMG)
    assert r.label == "like" and r.conf == 0.9 and r.face_box is None


def test_roi_uses_largest_face_and_maps_back():
    yolo = FakeYolo()
    faces = [Face(Box(0, 0, 10, 10), 0.9), Face(Box(300, 300, 400, 400), 0.8)]
    r = FaceAnchoredROIPipeline(yolo, FakeFaces(faces), scale=3.0).predict(IMG)
    assert r.face_box == Box(300, 300, 400, 400)
    assert yolo.shapes[-1][:2] == (300, 300)          # detector saw only the 3x ROI
    assert r.detections[0].box.as_list() == [210, 210, 250, 250]
    assert r.label == "like" and not r.used_fallback


def test_roi_fallback_and_no_fallback():
    r = FaceAnchoredROIPipeline(FakeYolo(), FakeFaces([]), fallback_full_frame=True).predict(IMG)
    assert r.used_fallback and r.label == "like"
    r = FaceAnchoredROIPipeline(FakeYolo(), FakeFaces([]), fallback_full_frame=False).predict(IMG)
    assert r.label == "none"


class FakePipe:
    def __init__(self, label):
        self.label = label

    def predict(self, img):
        return GestureResult(self.label, 0.9 if self.label != "none" else 0.0, [], face_box=Box(0, 0, 10, 10))


class FakeRec:
    def __init__(self, emb):
        self.emb = emb

    def embed_main(self, img, hint=None):
        return (None, None) if self.emb is None else (self.emb, Face(Box(0, 0, 10, 10), 1.0))


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_verifier_decisions():
    me, other = unit([1, 0, 0]), unit([0, 1, 0])
    assert Verifier(FakePipe("like"), FakeRec(me), 0.35).verify(IMG, "like", me).reason == "ok"
    assert Verifier(FakePipe("peace"), FakeRec(me), 0.35).verify(IMG, "like", me).reason == "wrong_gesture"
    assert Verifier(FakePipe("none"), FakeRec(me), 0.35).verify(IMG, "like", me).reason == "no_gesture"
    assert Verifier(FakePipe("like"), FakeRec(other), 0.35).verify(IMG, "like", me).reason == "face_mismatch"
    assert Verifier(FakePipe("like"), FakeRec(None), 0.35).verify(IMG, "like", me).reason == "no_face"
    d = Verifier(FakePipe("like"), FakeRec(me), 0.35).verify(IMG, "like", None)
    assert d.accepted and d.reason == "enroll" and d.embedding is not None


class NoHandInCrop(FakeYolo):
    """Finds the gesture only on the full image (hand outside the ROI)."""
    def detect(self, img):
        self.shapes.append(img.shape)
        return [Detection(self.box, "like", 0.8)] if img.shape[:2] == IMG.shape[:2] else []


def test_roi_retries_full_frame_when_crop_has_no_gesture():
    faces = FakeFaces([Face(Box(300, 300, 400, 400), 0.9)])
    strict = FaceAnchoredROIPipeline(NoHandInCrop(), faces).predict(IMG)
    assert strict.label == "none"                       # controlled comparison: no retry
    app = FaceAnchoredROIPipeline(NoHandInCrop(), faces, fallback_on_no_gesture=True).predict(IMG)
    assert app.label == "like" and app.used_fallback


def test_mediapipe_label_mapping_covers_default_gestures():
    from gesture_attendance.config import load_config
    from gesture_attendance.vision.mediapipe_gesture import SUPPORTED
    assert set(load_config()["gestures"]) <= set(SUPPORTED)

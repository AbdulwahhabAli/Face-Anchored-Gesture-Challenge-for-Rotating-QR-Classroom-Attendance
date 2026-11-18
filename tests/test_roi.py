import numpy as np

from gesture_attendance.vision.roi import crop_for_detector, face_roi
from gesture_attendance.vision.types import Box


def test_face_roi_centred_and_scaled():
    roi = face_roi(Box(400, 300, 500, 400), 1000, 1000, scale=3.0)
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (300, 200, 600, 500)


def test_face_roi_clipped_to_image():
    roi = face_roi(Box(0, 0, 100, 100), 200, 150, scale=4.0)
    assert roi.x1 == 0 and roi.y1 == 0 and roi.x2 == 200 and roi.y2 == 150


def test_face_roi_y_shift():
    roi = face_roi(Box(400, 300, 500, 400), 1000, 1000, scale=2.0, y_shift=0.5)
    assert (roi.y1, roi.y2) == (300, 500)


def test_crop_upscale_maps_back():
    img = np.zeros((600, 800, 3), np.uint8)
    crop, tf = crop_for_detector(img, Box(100, 50, 300, 250), upscale=True)
    assert crop.shape == (200, 200, 3)
    b = tf.to_full(Box(10, 20, 30, 40))
    assert b.as_list() == [110, 70, 130, 90]


def test_crop_no_upscale_pastes_at_native_scale():
    img = np.full((600, 800, 3), 7, np.uint8)
    crop, tf = crop_for_detector(img, Box(100, 50, 300, 250), upscale=False, size=640)
    assert crop.shape == (640, 640, 3)
    assert (crop[:200, :200] == 7).all() and (crop[300:, 300:] == 114).all()
    # a box running into the padding is clipped to the real pixels
    assert tf.to_full(Box(150, 150, 400, 400)).as_list() == [250, 200, 300, 250]


def test_crop_no_upscale_shrinks_large_crop():
    img = np.zeros((2000, 2000, 3), np.uint8)
    crop, tf = crop_for_detector(img, Box(0, 0, 1280, 1280), upscale=False, size=640)
    assert crop.shape == (640, 640, 3) and tf.s == 0.5
    assert tf.to_full(Box(10, 10, 20, 20)).as_list() == [20, 20, 40, 40]

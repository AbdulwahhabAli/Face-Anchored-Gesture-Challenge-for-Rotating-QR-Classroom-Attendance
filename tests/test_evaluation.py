import numpy as np
import pytest

from gesture_attendance.evaluation import (average_precision, best_accuracy_threshold, choose_enrollment, eer,
                                           map_at_iou, mcnemar_exact, parse_config_name, wrong_gesture_far)
from gesture_attendance.vision.types import Box


def test_parse_config_name():
    assert parse_config_name("full") == {"mode": "full"}
    p = parse_config_name("roi_s2.5_noup_scrfd")
    assert p == {"mode": "roi", "roi_scale": 2.5, "roi_upscale": False, "face_anchor": "scrfd"}
    with pytest.raises(ValueError):
        parse_config_name("roi_3")


def test_map_perfect_and_miss():
    g = [[("like", Box(0, 0, 10, 10))], [("fist", Box(0, 0, 10, 10))]]
    p = [[("like", 0.9, Box(0, 0, 10, 10))], [("fist", 0.8, Box(1, 1, 10, 10))]]
    assert map_at_iou(p, g, ["like", "fist"])["mAP"] == pytest.approx(1.0)
    p_bad = [[("like", 0.9, Box(50, 50, 60, 60))], [("fist", 0.8, Box(0, 0, 10, 10))]]
    assert map_at_iou(p_bad, g, ["like", "fist"])["per_class"]["like"] == 0.0


def test_average_precision_half():
    # one TP then one FP, 2 GT -> recall 0.5 at precision 1
    assert average_precision(np.array([0.5, 0.5]), np.array([1.0, 0.5])) == pytest.approx(0.5)


def test_mcnemar_counts():
    a = np.array([1, 1, 0, 0, 0, 1], bool)
    b = np.array([1, 0, 1, 1, 1, 1], bool)
    r = mcnemar_exact(a, b)
    assert r["n01_b_fixes"] == 3 and r["n10_b_breaks"] == 1 and 0 < r["p_value"] <= 1


def test_wrong_gesture_far():
    cls = ["a", "b", "c"]
    # img1 true a pred b -> accepted for challenge b (1 of 2); img2 true b pred b -> 0 of 2
    assert wrong_gesture_far(["a", "b"], ["b", "b"], cls) == pytest.approx(0.25)


def test_threshold_and_eer():
    sims = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    same = np.array([1, 1, 1, 0, 0, 0], bool)
    t, acc = best_accuracy_threshold(sims, same)
    assert acc == 1.0 and 0.2 < t <= 0.7
    assert eer(sims[same], sims[~same])[0] == 0.0


def test_choose_enrollment_prefers_normal():
    files = ["s1_b_low.jpg", "s1_a_normal.jpg", "s2_x_far.jpg"]
    e = choose_enrollment(files, ["s1", "s1", "s2"], ["low_light", "normal", "far"], np.array([1, 1, 1], bool))
    assert e == {"s1": 1, "s2": 2}

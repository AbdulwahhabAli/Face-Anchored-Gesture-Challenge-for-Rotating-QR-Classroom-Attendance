"""Metric helpers shared by the evaluation scripts (all pure numpy, unit-tested)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .gestures import NONE_LABEL
from .vision.types import Box

# ----------------------------------------------------------------------------- pipeline config names


def parse_config_name(name: str) -> dict:
    """'full' | 'roi_s3_up' | 'roi_s2.5_noup' | 'roi_s3_up_scrfd' -> pipeline config dict."""
    if name == "full":
        return {"mode": "full"}
    parts = name.split("_")
    if parts[0] != "roi" or len(parts) < 3 or not parts[1].startswith("s") or parts[2] not in ("up", "noup"):
        raise ValueError(f"bad config name '{name}' (expected full or roi_s<scale>_<up|noup>[_scrfd|_blazeface])")
    out = {"mode": "roi", "roi_scale": float(parts[1][1:]), "roi_upscale": parts[2] == "up",
           "face_anchor": "blazeface"}
    if len(parts) > 3:
        out["face_anchor"] = parts[3]
    return out


DEFAULT_CONFIGS = ["full", "roi_s2_up", "roi_s3_up", "roi_s4_up", "roi_s3_noup"]

# ----------------------------------------------------------------------------- classification


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the mean of a 0/1 vector."""
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> tuple[float, float, float]:
    """mean(b) - mean(a) with a paired percentile CI (a, b: 0/1 correctness on the same images)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def mcnemar_exact(a_correct, b_correct) -> dict:
    """Exact McNemar test on paired 0/1 outcomes. n01 = a wrong & b right, n10 = a right & b wrong."""
    from scipy.stats import binomtest

    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return {"n01_b_fixes": n01, "n10_b_breaks": n10, "p_value": p}


def per_class_pr(y_true, y_pred, classes: list[str]) -> list[dict]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        rows.append({"class": c, "support": int((y_true == c).sum()),
                     "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "recall": tp / (tp + fn) if tp + fn else float("nan")})
    return rows


def confusion(y_true, y_pred, classes: list[str]) -> np.ndarray:
    labels = list(classes) + [NONE_LABEL]
    ix = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(classes), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in ix and ix[t] < len(classes):
            m[ix[t], ix.get(p, len(labels) - 1)] += 1
    return m

# ----------------------------------------------------------------------------- detection mAP


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP (VOC2010+/COCO style without 101-point sampling)."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def map_at_iou(preds: list[list[tuple[str, float, Box]]], gts: list[list[tuple[str, Box]]],
               classes: list[str], iou_thr: float = 0.5) -> dict:
    """preds[i] = [(label, conf, box)], gts[i] = [(label, box)] for image i. Returns per-class AP and mAP."""
    aps = {}
    for c in classes:
        dets = [(conf, i, box) for i, p in enumerate(preds) for (lab, conf, box) in p if lab == c]
        n_gt = sum(1 for g in gts for (lab, _) in g if lab == c)
        if n_gt == 0:
            continue
        dets.sort(key=lambda d: -d[0])
        used = defaultdict(set)
        tp = np.zeros(len(dets))
        for k, (_, i, box) in enumerate(dets):
            best, best_j = 0.0, -1
            for j, (lab, g) in enumerate(gts[i]):
                if lab != c or j in used[i]:
                    continue
                iou = box.iou(g)
                if iou > best:
                    best, best_j = iou, j
            if best >= iou_thr:
                tp[k] = 1
                used[i].add(best_j)
        ctp = np.cumsum(tp)
        recall = ctp / n_gt
        precision = ctp / np.maximum(np.arange(1, len(dets) + 1), 1)
        aps[c] = average_precision(recall, precision) if len(dets) else 0.0
    return {"per_class": aps, "mAP": float(np.mean(list(aps.values()))) if aps else float("nan")}

# ----------------------------------------------------------------------------- challenge FAR (gesture side)


def wrong_gesture_far(y_true, y_pred, classes: list[str]) -> float:
    """Each image is attacked with every challenge != its true gesture; accepted iff pred == challenge."""
    acc = tot = 0
    for t, p in zip(y_true, y_pred):
        if t not in classes:
            continue
        for c in classes:
            if c == t:
                continue
            tot += 1
            acc += int(p == c)
    return acc / tot if tot else float("nan")

# ----------------------------------------------------------------------------- face verification


def roc_points(genuine: np.ndarray, impostor: np.ndarray):
    """Returns thresholds, FAR, FRR (accept iff sim >= threshold)."""
    thr = np.unique(np.concatenate([genuine, impostor]))
    thr = np.concatenate([[-1.0], thr, [1.0 + 1e-6]])
    far = np.array([(impostor >= t).mean() for t in thr])
    frr = np.array([(genuine < t).mean() for t in thr])
    return thr, far, frr


def eer(genuine, impostor) -> tuple[float, float]:
    thr, far, frr = roc_points(np.asarray(genuine), np.asarray(impostor))
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def best_accuracy_threshold(sims: np.ndarray, same: np.ndarray) -> tuple[float, float]:
    sims, same = np.asarray(sims), np.asarray(same, bool)
    cands = np.unique(sims)
    best_t, best_acc = 0.0, -1.0
    for t in cands:
        acc = ((sims >= t) == same).mean()
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    # put the threshold half-way to the next lower similarity for a margin
    lower = cands[cands < best_t]
    if len(lower):
        best_t = float((best_t + lower.max()) / 2)
    return best_t, best_acc


def choose_enrollment(files: list[str], subjects: list[str], conditions: list[str], has_face: np.ndarray) -> dict:
    """Per subject: enrollment image = first (sorted) image with a face whose condition contains 'normal'
    (falls back to any image with a face). Returns {subject: index}."""
    by_subj = defaultdict(list)
    for i, s in enumerate(subjects):
        if has_face[i]:
            by_subj[s].append(i)
    out = {}
    for s, idxs in by_subj.items():
        idxs = sorted(idxs, key=lambda i: files[i])
        normal = [i for i in idxs if "normal" in str(conditions[i]).lower()]
        out[s] = (normal or idxs)[0]
    return out


def load_threshold(path_or_value) -> float:
    try:
        return float(path_or_value)
    except (TypeError, ValueError):
        return float(json.loads(Path(path_or_value).read_text())["threshold"])

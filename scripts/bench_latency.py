"""Latency per image (median / p95) of the baseline vs ROI pipelines, end to end, plus model sizes.

    python scripts/bench_latency.py --images-dir data/classroom/images --n 100 --device cpu
    python scripts/bench_latency.py --images-dir data/hagrid_yolo/images/test --n 100 --device 0   # Colab T4

Each pipeline is timed from the decoded image to the gesture decision (face detection included for
ROI); ArcFace (needed by both pipelines) is timed separately. Images are resized so the longer side
is --max-side (phone selfies are large; set 0 to keep native size).
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np
import pandas as pd

from gesture_attendance.config import load_config, merged, resolve_path
from gesture_attendance.evaluation import parse_config_name
from gesture_attendance.verify import build_gesture_pipeline
from gesture_attendance.vision.downloads import CACHE_DIR


def stats(x):
    x = np.asarray(x)
    return {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)), "mean_ms": float(x.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--configs", nargs="+", default=["full", "roi_s3_up"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/latency"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["models"]["device"] = args.device

    paths = sorted(p for p in args.images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.n]
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")
    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
        if args.max_side and max(im.shape[:2]) > args.max_side:
            k = args.max_side / max(im.shape[:2])
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        imgs.append(im)
    print(f"{len(imgs)} images, device={cfg['models']['device']}")

    from gesture_attendance.vision.face_recognizer import ArcFaceRecognizer
    from gesture_attendance.verify import build_gesture_detector
    m = cfg["models"]
    yolo = build_gesture_detector(cfg, args.weights)
    rec = ArcFaceRecognizer(resolve_path(m.get("arcface_dir")))

    rows = []
    for name in args.configs:
        pc = parse_config_name(name)
        # controlled comparison: no "retry on full frame" fallback inside the ROI timing
        pipe = build_gesture_pipeline(merged(cfg, pipeline={**pc, "fallback_on_no_gesture": False}),
                                      gesture_detector=yolo, recognizer=rec)
        for im in imgs[: args.warmup]:
            pipe.predict(im)
        tot, stages = [], {}
        for im in imgs:
            t0 = time.perf_counter()
            r = pipe.predict(im)
            tot.append((time.perf_counter() - t0) * 1e3)
            for k, v in r.timings_ms.items():
                stages.setdefault(k, []).append(v)
        row = {"config": name, **stats(tot)}
        for k, v in stages.items():
            row[f"{k}_median_ms"] = float(np.median(v))
        rows.append(row)
        print(f"{name:>14}: median {row['median_ms']:.1f} ms, p95 {row['p95_ms']:.1f} ms")

    t_arc = []
    for im in imgs:
        t0 = time.perf_counter()
        rec.embed_main(im)
        t_arc.append((time.perf_counter() - t0) * 1e3)
    rows.append({"config": "arcface (both pipelines)", **stats(t_arc)})

    R = pd.DataFrame(rows)
    if "full" in args.configs:
        base = R.set_index("config").loc["full", "median_ms"]
        R["extra_vs_full_median_ms"] = R.median_ms - base
        R.loc[R.config.str.startswith("arcface"), "extra_vs_full_median_ms"] = float("nan")
    R.insert(0, "device", f"{m['device']} ({platform.processor() or platform.machine()})")
    R.to_csv(args.out / f"latency_{str(m['device']).replace(':', '')}.csv", index=False)

    sizes = {"yolo_weights": resolve_path(args.weights or m["yolo_weights"]),
             "mediapipe_gesture": CACHE_DIR / "gesture_recognizer.task",
             "blazeface": CACHE_DIR / "blaze_face_short_range.tflite",
             "scrfd_det_500m": CACHE_DIR / "buffalo_s" / "det_500m.onnx",
             "arcface_w600k_mbf": CACHE_DIR / "buffalo_s" / "w600k_mbf.onnx"}
    S = pd.DataFrame([{"model": k, "size_MB": round(p.stat().st_size / 2**20, 2) if p.exists() else None}
                      for k, p in sizes.items()])
    S.to_csv(args.out / "model_sizes.csv", index=False)
    try:
        print(R.to_markdown(index=False, floatfmt=".1f"))
        print(S.to_markdown(index=False))
    except ImportError:
        print(R, S, sep="\n")


if __name__ == "__main__":
    main()

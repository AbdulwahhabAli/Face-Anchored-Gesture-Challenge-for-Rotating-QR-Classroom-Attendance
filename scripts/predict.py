"""Run every pipeline configuration over a test set and cache per-image predictions.

All configurations share the same YOLO weights and the same images; face detection is run once
per image and re-used by every ROI configuration (so ablations only change the ROI geometry).

    python scripts/predict.py --dataset hagrid    --root data/hagrid_yolo --out runs/preds/hagrid
    python scripts/predict.py --dataset classroom --root data/classroom   --out runs/preds/classroom

Writes <out>/<config>.csv for each config and, with --embeddings (default for classroom),
<out>/embeddings.npz with one ArcFace embedding per image (for eval_face.py / eval_e2e.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np
import pandas as pd

from gesture_attendance.config import load_config, parse_overrides, resolve_path
from gesture_attendance.datasets import load_dataset
from gesture_attendance.evaluation import DEFAULT_CONFIGS, parse_config_name
from gesture_attendance.vision.pipelines import FaceAnchoredROIPipeline, FullFramePipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["hagrid", "classroom"], required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--weights", default=None, help="YOLO weights (forces the yolo backend)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. models.device=0")
    ap.add_argument("--embeddings", action=argparse.BooleanOptionalAction, default=None,
                    help="also save ArcFace embeddings (default: on for classroom)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (debugging)")
    ap.add_argument("--append", action="store_true",
                    help="add --configs to an existing --out run (same images); keeps its CSVs and embeddings")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    m = cfg["models"]
    samples = load_dataset(args.dataset, args.root, args.split)
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} images, configs: {args.configs}")
    args.out.mkdir(parents=True, exist_ok=True)
    prev_configs = []
    if args.append and (args.out / "run_info.json").exists():
        prev_configs = json.loads((args.out / "run_info.json").read_text())["configs"]
        args.configs = [c for c in args.configs if c not in prev_configs]
        args.embeddings = False
        print(f"appending {args.configs} to {prev_configs}")

    from gesture_attendance.verify import build_gesture_detector
    yolo = build_gesture_detector(cfg, args.weights)  # gesture detector (MediaPipe or YOLO, per config)

    parsed = {c: parse_config_name(c) for c in args.configs}
    anchors_needed = {p["face_anchor"] for p in parsed.values() if p["mode"] == "roi"}
    want_emb = args.embeddings if args.embeddings is not None else args.dataset == "classroom"

    recognizer = None
    if want_emb or "scrfd" in anchors_needed:
        from gesture_attendance.vision.face_recognizer import ArcFaceRecognizer
        recognizer = ArcFaceRecognizer(resolve_path(m.get("arcface_dir")))
    anchors = {}
    if "blazeface" in anchors_needed:
        from gesture_attendance.vision.face_detector import BlazeFaceDetector
        anchors["blazeface"] = BlazeFaceDetector(resolve_path(m.get("blazeface_model")), m.get("blazeface_min_conf", 0.5))
    if "scrfd" in anchors_needed:
        anchors["scrfd"] = recognizer

    pipelines = {}
    for name, p in parsed.items():
        if p["mode"] == "full":
            pipelines[name] = FullFramePipeline(yolo)
        else:
            pipelines[name] = FaceAnchoredROIPipeline(
                yolo, anchors[p["face_anchor"]], scale=p["roi_scale"], upscale=p["roi_upscale"],
                y_shift=p.get("roi_y_shift", cfg["pipeline"].get("roi_y_shift", 0.0)),
                fallback_full_frame=cfg["pipeline"].get("fallback_full_frame", True), imgsz=m.get("imgsz", 640))

    rows = {n: [] for n in args.configs}
    emb_list, has_face = [], []
    t_start = time.time()
    for k, s in enumerate(samples):
        img = cv2.imread(str(s.path))
        if img is None:
            print(f"  cannot read {s.path}, skipped")
            continue
        base = {"file": str(s.path.relative_to(args.root)) if s.path.is_relative_to(args.root) else str(s.path),
                "subject": s.subject, "gesture": s.gesture, "condition": s.condition,
                "background_gesture": s.background_gesture or "",
                "gt": json.dumps([[lab, b.as_list()] for lab, b in s.gt]),
                "width": img.shape[1], "height": img.shape[0]}
        faces_cache = {}
        for a, det in anchors.items():
            t0 = time.perf_counter()
            faces_cache[a] = (det.detect(img), (time.perf_counter() - t0) * 1e3)
        for name, pipe in pipelines.items():
            p = parsed[name]
            faces, t_face = faces_cache.get(p.get("face_anchor"), (None, 0.0)) if p["mode"] == "roi" else (None, 0.0)
            r = pipe.predict(img, faces=faces)
            rows[name].append({**base, "pred": r.label, "conf": round(r.conf, 4),
                               "face_found": r.face_box is not None if p["mode"] == "roi" else "",
                               "fallback": r.used_fallback,
                               "face_box": json.dumps(r.face_box.as_list() if r.face_box else None),
                               "roi": json.dumps(r.roi.as_list() if r.roi else None),
                               "dets": json.dumps([d.to_dict() for d in r.detections[:10]]),
                               "t_face_ms": round(t_face, 2),
                               "t_gesture_ms": round(r.timings_ms.get("gesture", 0.0), 2)})
        if want_emb:
            e, _ = recognizer.embed_main(img)
            has_face.append(e is not None)
            emb_list.append(e if e is not None else np.zeros(512, np.float32))
        if (k + 1) % 25 == 0 or k + 1 == len(samples):
            el = time.time() - t_start
            print(f"  {k + 1}/{len(samples)}  ({el / (k + 1):.2f} s/img)")

    for name, r in rows.items():
        df = pd.DataFrame(r)
        df.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name:>16}: top-1 acc {(df.pred == df.gesture).mean():.3f}")
    if want_emb:
        ref = pd.DataFrame(rows[args.configs[0]])
        np.savez(args.out / "embeddings.npz", files=ref.file.values, subjects=ref.subject.values,
                 gestures=ref.gesture.values, conditions=ref.condition.values,
                 emb=np.stack(emb_list), has_face=np.array(has_face))
        print(f"embeddings -> {args.out / 'embeddings.npz'}")
    (args.out / "run_info.json").write_text(json.dumps({
        "dataset": args.dataset, "root": str(args.root), "configs": prev_configs + args.configs,
        "gesture_backend": "yolo" if args.weights else m.get("gesture_backend", "mediapipe"),
        "weights": str(args.weights or m["yolo_weights"]), "classes": cfg["gestures"],
        "n_images": len(samples)}, indent=2))


if __name__ == "__main__":
    main()

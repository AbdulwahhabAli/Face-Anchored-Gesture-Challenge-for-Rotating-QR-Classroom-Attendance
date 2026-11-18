"""Demo: baseline (full frame) vs ours (face-anchored ROI) side by side, on images or a live webcam.

Both pipelines use the same gesture detector; the same ArcFace check is applied to both.
A check-in is ACCEPTED iff the predicted gesture equals the challenge AND the face matches the enrolled one.

    # held-out images (challenge = the image's true gesture from meta.csv, or --challenge)
    python scripts/demo_compare.py --images data/hagrid_eval/images/test --meta data/hagrid_eval/meta.csv \
        --pick demo --out runs/demo --video runs/demo/demo_backup.mp4
    # live webcam: SPACE = take photo, N = new random challenge, E = enroll current face, Q = quit
    python scripts/demo_compare.py --webcam 0
    # YOLO backend instead of MediaPipe
    python scripts/demo_compare.py --webcam 0 --weights models/gesture_yolov8n.pt
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np

from gesture_attendance.config import load_config, merged, parse_overrides, resolve_path
from gesture_attendance.gestures import NONE_LABEL
from gesture_attendance.verify import build_gesture_detector, build_gesture_pipeline
from gesture_attendance.vision.face_recognizer import ArcFaceRecognizer, cosine

GREEN, RED, CYAN, WHITE = (60, 200, 60), (40, 40, 230), (255, 200, 0), (255, 255, 255)


def run_both(img, pipes, rec, challenge, enrolled, thr):
    out = {}
    emb, _ = rec.embed_main(img)
    sim = cosine(emb, enrolled) if (emb is not None and enrolled is not None) else None
    face_ok = enrolled is None or (sim is not None and sim >= thr)
    for name, p in pipes.items():
        t0 = time.perf_counter()
        r = p.predict(img)
        ms = (time.perf_counter() - t0) * 1e3
        ok = r.label == challenge and face_ok
        reason = ("ACCEPT" if ok else ("no gesture" if r.label == NONE_LABEL else
                                       ("wrong gesture" if r.label != challenge else "face mismatch")))
        out[name] = (r, ms, ok, reason)
    return out, sim, emb


def render(img, res, challenge, sim, title_extra="", width=1600, attack=False):
    panels = []
    for name, label in (("full", "BASELINE: full frame"), ("roi", "OURS: face-anchored ROI")):
        r, ms, ok, reason = res[name]
        im = img.copy()
        t = max(2, im.shape[1] // 350)
        if r.face_box is not None:
            b = r.face_box
            cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), CYAN, t)
        if r.roi is not None:
            b = r.roi
            cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), WHITE, t)
        for d in r.detections[:4]:
            b = d.box
            col = GREEN if d.label == challenge else RED
            cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, t)
            cv2.putText(im, f"{d.label} {d.conf:.2f}", (int(b.x1), max(20, int(b.y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, col, t)
        pw = width // 2
        k = pw / im.shape[1]
        im = cv2.resize(im, (pw, int(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
        bar = np.full((110, pw, 3), 30, np.uint8)
        cv2.putText(bar, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(bar, f"pred: {r.label}   ({ms:.0f} ms{', no face -> full frame' if r.used_fallback else ''})",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        if attack:  # the challenge is the BACKGROUND person's gesture: accepting is a security failure
            reason, good = ("ACCEPTED ATTACKER" if ok else "attack rejected"), not ok
        else:
            good = ok
        cv2.putText(bar, reason, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN if good else RED, 2)
        panels.append(np.vstack([bar, im]))
    h = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((h - p.shape[0], p.shape[1], 3), np.uint8)]) for p in panels]
    top = np.full((50, width, 3), 15, np.uint8)
    s = f"Challenge: {challenge}" + (f"   face sim {sim:.2f}" if sim is not None else "") + title_extra
    cv2.putText(top, s, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    return np.vstack([top, np.hstack(panels)])


def pick_demo(meta_rows, n_per_cond=3, seed=7):
    """A fixed, reproducible demo playlist: a few held-out images per condition."""
    rng = random.Random(seed)
    out = []
    for cond in ("normal", "low_light", "far", "crowded"):
        rows = [r for r in meta_rows if r["condition"] == cond]
        out += rng.sample(rows, min(n_per_cond, len(rows)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="image file or folder")
    ap.add_argument("--meta", type=Path, help="meta.csv giving the true gesture (used as the challenge)")
    ap.add_argument("--pick", default="all", help="'all', 'demo' (3 per condition) or a comma list of files")
    ap.add_argument("--files", nargs="*", default=None, help="explicit file names inside --images")
    ap.add_argument("--webcam", type=int, default=None)
    ap.add_argument("--challenge", default=None)
    ap.add_argument("--weights", default=None, help="YOLO weights (default: MediaPipe backend)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides")
    ap.add_argument("--out", type=Path, default=Path("runs/demo"))
    ap.add_argument("--video", type=Path, default=None, help="also write the panels as an MP4 (backup recording)")
    ap.add_argument("--seconds-per-image", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_config(None, parse_overrides(args.set))
    classes = cfg["gestures"] if not args.weights else ["like", "peace", "palm", "fist", "ok", "one"]
    det = build_gesture_detector(cfg, args.weights)
    rec = ArcFaceRecognizer(resolve_path(cfg["models"].get("arcface_dir")))
    base_roi = {"mode": "roi", "fallback_on_no_gesture": False}
    pipes = {"full": build_gesture_pipeline(merged(cfg, pipeline={"mode": "full"}), gesture_detector=det),
             "roi": build_gesture_pipeline(merged(cfg, pipeline=base_roi), gesture_detector=det, recognizer=rec)}
    thr = cfg["verification"]["face_threshold"]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        challenge, enrolled, last = random.choice(classes), None, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            cv2.putText(view, f"Challenge: {challenge}  [SPACE photo, N new, E enroll, Q quit]"
                        + ("  enrolled" if enrolled is not None else ""), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow("camera", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n"):
                challenge = random.choice(classes)
            if k == ord("e"):
                enrolled, _ = rec.embed_main(frame)
            if k == ord(" "):
                res, sim, _ = run_both(frame, pipes, rec, challenge, enrolled, thr)
                last = render(frame, res, challenge, sim)
                cv2.imwrite(str(args.out / f"webcam_{int(time.time())}.jpg"), last)
                cv2.imshow("baseline vs ours", last)
        cap.release()
        cv2.destroyAllWindows()
        return

    meta = {}
    if args.meta:
        with open(args.meta, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        meta = {r["file"]: r for r in rows}
    if args.images.is_file():
        items = [(args.images, meta.get(args.images.name))]
    elif args.files:
        items = [(args.images / f, meta.get(f)) for f in args.files]
    elif args.pick == "demo" and meta:
        items = [(args.images / r["file"], r) for r in pick_demo(list(meta.values()))]
    else:
        items = [(p, meta.get(p.name)) for p in sorted(args.images.iterdir())
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    writer, frames = None, []
    for path, row in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        challenge = args.challenge or (row["gesture"] if row else random.choice(classes))
        attack = bool(row and row.get("condition") == "crowded" and row.get("background_gesture")
                      and args.challenge is None and row["background_gesture"] in classes)
        if attack:  # the claimant shows one gesture; the person next to them shows the challenged one
            challenge = row["background_gesture"]
        res, sim, _ = run_both(img, pipes, rec, challenge, None, thr)
        extra = (f"   [{row['condition']}, claimant shows {row['gesture']}"
                 + (", ATTACK: background person shows the challenge]" if attack else "]")) if row else ""
        panel = render(img, res, challenge, sim, extra, attack=attack)
        dst = args.out / f"compare_{path.stem}.jpg"
        cv2.imwrite(str(dst), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{path.name}: challenge={challenge}  full={res['full'][0].label}({res['full'][3]})  "
              f"roi={res['roi'][0].label}({res['roi'][3]})")
        frames.append(panel)

    if args.video and frames:
        H = max(f.shape[0] for f in frames)
        W = frames[0].shape[1]
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
        for f in frames:
            f = np.vstack([f, np.zeros((H - f.shape[0], W, 3), np.uint8)])
            for _ in range(int(args.seconds_per_image * 10)):
                writer.write(f)
        writer.release()
        print(f"backup video -> {args.video}")


if __name__ == "__main__":
    main()

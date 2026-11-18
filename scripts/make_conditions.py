"""Deterministically build the evaluation set with four capture conditions from the HaGRID test split.

Every condition is derived from the SAME 600 native-resolution test images (100 held-out subjects x
6 gestures), so baseline and ROI pipelines are compared on identical inputs. Each output image is a pure
function of (source image, condition, --seed); re-running the script recreates the set bit-for-bit.

  normal     the original HaGRID test image
  low_light  brightness x0.35, gamma 1.4, Gaussian sensor noise (sigma 6), JPEG q85
  far        zoom-out x2: the image is pasted at native resolution in the centre of a canvas twice as wide
             and twice as tall (e.g. 1920x1080 -> 3840x2160, a 8 MP phone frame), so the subject covers a
             quarter of the frame; the padding is the image's mean border colour plus mild noise
  crowded    background-hand attack: a DIFFERENT held-out subject showing a DIFFERENT gesture is pasted
             at 0.6x scale to the right of the claimant. `background_gesture` holds that gesture; a pipeline
             that outputs it would accept an attacker who copies the gesture of a person behind them.

    python scripts/make_conditions.py --src data/hagrid_yolo --out data/hagrid_eval --seed 441

Output: <out>/images/test/<condition>__<file>.jpg, YOLO labels (claimant's boxes only), meta.csv with
condition / background_gesture / background_user columns, data.yaml.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np
import yaml

CONDITIONS = ("normal", "low_light", "far", "crowded")


def rng_for(name: str, cond: str, seed: int) -> np.random.Generator:
    h = int(hashlib.sha256(f"{seed}/{cond}/{name}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def border_colour(img: np.ndarray) -> np.ndarray:
    b = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3), img[:, :8].reshape(-1, 3),
                        img[:, -8:].reshape(-1, 3)])
    return b.mean(0)


def canvas(h: int, w: int, colour: np.ndarray, rng) -> np.ndarray:
    c = np.empty((h, w, 3), np.float32)
    c[:] = colour
    c += rng.normal(0, 4, size=(h, w, 1)).astype(np.float32)
    return np.clip(c, 0, 255).astype(np.uint8)


def low_light(img, rng):
    x = img.astype(np.float32) / 255.0
    x = 0.35 * np.power(x, 1.4) * 255.0
    x += rng.normal(0, 6, size=x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def far(img, rng):
    h, w = img.shape[:2]
    out = canvas(2 * h, 2 * w, border_colour(img), rng)
    ox, oy = w // 2, h // 2
    out[oy:oy + h, ox:ox + w] = img
    return out, (ox, oy, 1.0, 2 * w, 2 * h)


def crowded(img, other, rng, s=0.6):
    h, w = img.shape[:2]
    o = cv2.resize(other, None, fx=s * h / other.shape[0], fy=s * h / other.shape[0], interpolation=cv2.INTER_AREA)
    oh, ow = o.shape[:2]
    out = canvas(h, w + ow, border_colour(img), rng)
    out[:, :w] = img
    oy = (h - oh) // 2
    out[oy:oy + oh, w:w + ow] = o
    return out, (0, 0, 1.0, w + ow, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/hagrid_yolo", type=Path)
    ap.add_argument("--out", default="data/hagrid_eval", type=Path)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=441)
    ap.add_argument("--split", default="test", help="source split (test, or val for model selection)")
    args = ap.parse_args()
    sp = args.split

    with open(args.src / "meta.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == sp]
    rows.sort(key=lambda r: r["file"])
    names = yaml.safe_load(open(args.src / "data.yaml", encoding="utf-8"))["names"]
    (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    out_rows = []
    for k, r in enumerate(rows):
        src = args.src / "images" / sp / r["file"]
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        lbl = (args.src / "labels" / sp / (Path(r["file"]).stem + ".txt")).read_text().split("\n")
        boxes = [list(map(float, ln.split())) for ln in lbl if ln.strip()]
        for cond in args.conditions:
            rng = rng_for(r["file"], cond, args.seed)
            bg_g, bg_u = "", ""
            if cond == "normal":
                out, tf = img, (0, 0, 1.0, w, h)
            elif cond == "low_light":
                out, tf = low_light(img, rng), (0, 0, 1.0, w, h)
            elif cond == "far":
                out, tf = far(img, rng)
            elif cond == "crowded":
                cands = [o for o in rows if o["user_id"] != r["user_id"] and o["gesture"] != r["gesture"]]
                o = cands[int(rng.integers(len(cands)))]
                other = cv2.imread(str(args.src / "images" / sp / o["file"]))
                out, tf = crowded(img, other, rng)
                bg_g, bg_u = o["gesture"], o["user_id"]
            else:
                raise ValueError(cond)
            ox, oy, s, W, H = tf
            name = f"{cond}__{r['file']}"
            cv2.imwrite(str(args.out / "images" / sp / name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for c, cx, cy, bw, bh in boxes:  # claimant's boxes in the new canvas, normalised
                lines.append(f"{int(c)} {(ox + cx * w * s) / W:.6f} {(oy + cy * h * s) / H:.6f} "
                             f"{bw * w * s / W:.6f} {bh * h * s / H:.6f}")
            (args.out / "labels" / sp / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
            out_rows.append({**r, "file": name, "width": W, "height": H, "condition": cond,
                             "background_gesture": bg_g, "background_user": bg_u, "source_file": r["file"]})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)}")

    with open(args.out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    yaml.safe_dump({"path": str(args.out.resolve()), "train": f"images/{sp}", "val": f"images/{sp}",
                    "test": f"images/{sp}", "names": names}, open(args.out / "data.yaml", "w", encoding="utf-8"),
                   sort_keys=False)
    print(f"{len(out_rows)} images ({len(rows)} sources x {len(args.conditions)} conditions) -> {args.out}")


if __name__ == "__main__":
    main()

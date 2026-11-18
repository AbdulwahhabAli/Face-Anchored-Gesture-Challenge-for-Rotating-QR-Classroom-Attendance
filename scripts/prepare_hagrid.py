"""Build the 6-class HaGRID subset in YOLO format, split BY SUBJECT (user_id) into train/val/test.

Works with HaGRID v1 and v2 annotation JSONs ({class}.json, one dict per image:
{"bboxes": [[x, y, w, h] normalised, top-left], "labels": [...], "user_id": ...}).

Example
    python scripts/prepare_hagrid.py --ann-dir D:/hagrid/annotations --images-root D:/hagrid/images \
        --out data/hagrid_yolo --per-class 400 --seed 441

Output
    <out>/images/{train,val,test}/<class>_<image_id>.jpg
    <out>/labels/{train,val,test}/<class>_<image_id>.txt   (YOLO: cls cx cy w h, normalised)
    <out>/meta.csv      split, file, gesture, user_id, width, height, n_boxes
    <out>/data.yaml     Ultralytics dataset file
    <out>/split_users.json  which subject ids went into which split (reproducibility)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
import cv2
import yaml

from gesture_attendance.config import load_config

SPLITS = ("train", "val", "test")


def find_annotations(ann_dir: Path, cls: str) -> dict:
    merged = {}
    for p in sorted(ann_dir.rglob(f"{cls}.json")):
        with open(p, encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def find_image_dirs(images_root: Path, cls: str) -> list[Path]:
    """Class folders: <root>/<cls>, <root>/*/<cls>, <root>/train_val_<cls>, <root>/*/test_<cls> ..."""
    dirs = []
    for depth_glob in ("*", "*/*", "*/*/*"):
        for d in images_root.glob(depth_glob):
            if d.is_dir() and (d.name == cls or d.name.endswith("_" + cls)):
                dirs.append(d)
    return dirs


def index_images(dirs: list[Path]) -> dict[str, Path]:
    idx = {}
    for d in dirs:
        for p in d.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                idx[p.stem] = p
    return idx


def split_users(users: list[str], fractions: tuple[float, float, float], seed: int) -> dict[str, str]:
    users = sorted(users)
    random.Random(seed).shuffle(users)
    n = len(users)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    assign = {}
    for i, u in enumerate(users):
        assign[u] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out", default="data/hagrid_yolo", type=Path)
    ap.add_argument("--classes", nargs="+", default=None, help="default: gestures from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--fractions", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=441)
    ap.add_argument("--max-side", type=int, default=0,
                    help="resize longer side to this when copying (0 = keep original resolution; "
                         "keep it 0 for the ROI experiments - the ROI gains come from the extra pixels)")
    args = ap.parse_args()

    classes = args.classes or load_config(args.config)["gestures"]
    cls_id = {c: i for i, c in enumerate(classes)}

    # 1) collect candidate records that have an image on disk
    records = defaultdict(list)  # cls -> [(image_id, ann, path)]
    for c in classes:
        anns = find_annotations(args.ann_dir, c)
        imgs = index_images(find_image_dirs(args.images_root, c))
        for image_id, a in anns.items():
            if image_id in imgs and c in a.get("labels", []):
                records[c].append((image_id, a, imgs[image_id]))
        print(f"{c:>8}: {len(anns):6d} annotations, {len(records[c]):6d} with images")
        if not records[c]:
            raise SystemExit(f"no images found for class '{c}' - check --ann-dir / --images-root")

    # 2) split subjects (user_id) globally -> no person appears in two splits
    users = sorted({a["user_id"] for recs in records.values() for _, a, _ in recs})
    assign = split_users(users, tuple(args.fractions), args.seed)
    print(f"{len(users)} subjects -> " + ", ".join(f"{s}: {sum(v == s for v in assign.values())}" for s in SPLITS))

    # 3) sample per class per split
    rng = random.Random(args.seed)
    chosen = []  # (split, cls, image_id, ann, path)
    for c in classes:
        by_split = defaultdict(list)
        for rec in sorted(records[c], key=lambda r: r[0]):
            by_split[assign[rec[1]["user_id"]]].append(rec)
        for s, frac in zip(SPLITS, args.fractions):
            want = round(args.per_class * frac)
            pool = by_split[s]
            pick = rng.sample(pool, min(want, len(pool)))
            if len(pick) < want:
                print(f"  warning: {c}/{s} only {len(pick)}/{want} images available")
            chosen += [(s, c, *r) for r in pick]

    # 4) write YOLO dataset
    out = args.out
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (s, c, image_id, a, src) in enumerate(chosen):
        name = f"{c}_{image_id}{src.suffix.lower()}"
        dst = out / "images" / s / name
        img = cv2.imread(str(src))
        if img is None:
            print(f"  skip unreadable {src}")
            continue
        h, w = img.shape[:2]
        if args.max_side and max(h, w) > args.max_side:
            k = args.max_side / max(h, w)
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            shutil.copy2(src, dst)
        lines = []
        for (bx, by, bw, bh), lab in zip(a["bboxes"], a["labels"]):
            if lab not in cls_id:  # "no_gesture" hands and classes outside our vocabulary = background
                continue
            lines.append(f"{cls_id[lab]} {bx + bw / 2:.6f} {by + bh / 2:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / s / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        rows.append({"split": s, "file": name, "gesture": c, "user_id": a["user_id"],
                     "width": w, "height": h, "n_boxes": len(lines)})
        if (i + 1) % 200 == 0:
            print(f"  wrote {i + 1}/{len(chosen)}")

    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {i: c for i, c in enumerate(classes)}}, f, sort_keys=False)
    json.dump({s: sorted(u for u, v in assign.items() if v == s) for s in SPLITS},
              open(out / "split_users.json", "w"), indent=1)
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print(f"done: {dict(counts)} images -> {out}")


if __name__ == "__main__":
    main()

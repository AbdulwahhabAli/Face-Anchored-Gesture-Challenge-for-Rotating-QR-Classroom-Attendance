"""Download a small, reproducible HaGRIDv2 subset WITHOUT downloading the 40 GB per-class archives.

The official archives are plain ZIP files on a server that supports HTTP range requests, so we read
each archive's central directory and fetch only the selected images (one range request per image).

Split = HaGRID's OFFICIAL subject-disjoint split (train / val / test by `user_id`):
  * train/val: --train-per-class / --val-per-class images per class, one image per subject where possible,
               resized so the longer side is --train-max-side (YOLO trains at 640 anyway)
  * test     : --test-subjects subjects that appear in the test split of ALL classes,
               one image per (subject, class), kept at NATIVE resolution (the ROI needs the pixels)

    python scripts/fetch_hagrid.py --out data/hagrid_yolo --seed 441

Output (same layout as prepare_hagrid.py, so train_yolo.py / predict.py work unchanged):
    <out>/images/{train,val,test}/<class>_<image_id>.jpg
    <out>/labels/{train,val,test}/<class>_<image_id>.txt   (YOLO: cls cx cy w h, normalised)
    <out>/meta.csv  split, file, gesture, user_id, width, height, n_boxes, age, gender, race
    <out>/data.yaml, <out>/split_users.json, <out>/selection.json (image ids -> exact reproduction)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import struct
import threading
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np
import requests
import yaml

BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets"
ANN_URL = f"{BASE}/hagrid_v2/annotations_with_landmarks/annotations.zip"
V1 = f"{BASE}/hagrid/hagrid_dataset_new_554800/hagrid_dataset"
V2 = f"{BASE}/hagrid_v2/hagrid_v2_zip"
CLASS_URL = {"like": f"{V1}/like.zip", "peace": f"{V1}/peace.zip", "palm": f"{V1}/palm.zip",
             "fist": f"{V1}/fist.zip", "ok": f"{V1}/ok.zip", "one": f"{V2}/one.zip"}
SPLITS = ("train", "val", "test")
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def get_range(url: str, start: int, end: int, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            r = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise IOError(f"range {start}-{end} of {url} failed")


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to read the central directory)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(requests.head(url, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        d = get_range(self.url, self.pos, end)
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=4 << 20))


def read_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Fetch one member with a single range request (local header + data)."""
    slack = 30 + len(info.filename.encode()) + 1024
    blob = get_range(url, info.header_offset, info.header_offset + slack + info.compress_size - 1)
    if blob[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    n, m = struct.unpack("<HH", blob[26:30])
    data = blob[30 + n + m: 30 + n + m + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return data
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise IOError(f"unsupported compression {info.compress_type}")


def load_annotations(cache: Path, classes: list[str]) -> dict:
    """{split: {cls: {image_id: ann}}}; the needed JSONs are cached under <cache>."""
    cache.mkdir(parents=True, exist_ok=True)
    z = None
    out = {s: {} for s in SPLITS}
    for s in SPLITS:
        for c in classes:
            p = cache / s / f"{c}.json"
            if not p.exists():
                z = z or open_remote_zip(ANN_URL)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(read_member(ANN_URL, z.getinfo(f"annotations/{s}/{c}.json")))
                print(f"  annotations/{s}/{c}.json  {p.stat().st_size / 2**20:.1f} MB")
            out[s][c] = json.loads(p.read_text(encoding="utf-8"))
    return out


def pick_one_per_user(anns: dict, cls: str, n: int, rng: random.Random) -> list[str]:
    by_user = defaultdict(list)
    for iid, a in sorted(anns.items()):
        if cls in a.get("labels", []):
            by_user[a["user_id"]].append(iid)
    users = sorted(by_user)
    rng.shuffle(users)
    picked = [rng.choice(by_user[u]) for u in users[:n]]
    if len(picked) < n:  # fewer subjects than wanted: allow a second image per subject
        rest = sorted(set(i for u in users for i in by_user[u]) - set(picked))
        picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/hagrid_yolo", type=Path)
    ap.add_argument("--ann-cache", default="data/hagrid_ann", type=Path)
    ap.add_argument("--classes", nargs="+", default=["like", "peace", "palm", "fist", "ok", "one"])
    ap.add_argument("--train-per-class", type=int, default=300)
    ap.add_argument("--val-per-class", type=int, default=50)
    ap.add_argument("--test-subjects", type=int, default=100)
    ap.add_argument("--train-max-side", type=int, default=960)
    ap.add_argument("--seed", type=int, default=441)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only-splits", nargs="+", default=None,
                    help="download only these splits of the (identical) selection, e.g. --only-splits val")
    ap.add_argument("--keep-native", action="store_true", help="keep native resolution for every split")
    args = ap.parse_args()
    classes, cls_id = args.classes, {c: i for i, c in enumerate(args.classes)}
    rng = random.Random(args.seed)

    print("annotations ...")
    A = load_annotations(args.ann_cache, classes)

    # --- selection -----------------------------------------------------------------------------
    sel = []  # (split, cls, image_id)
    for c in classes:
        sel += [("train", c, i) for i in pick_one_per_user(A["train"][c], c, args.train_per_class, rng)]
        sel += [("val", c, i) for i in pick_one_per_user(A["val"][c], c, args.val_per_class, rng)]
    users_per_cls = {c: {a["user_id"] for a in A["test"][c].values() if c in a.get("labels", [])} for c in classes}
    common = sorted(set.intersection(*users_per_cls.values()))
    rng.shuffle(common)
    test_users = sorted(common[: args.test_subjects])
    print(f"test: {len(common)} subjects show all {len(classes)} gestures; using {len(test_users)}")
    tu = set(test_users)
    for c in classes:
        by_user = defaultdict(list)
        for iid, a in sorted(A["test"][c].items()):
            if a["user_id"] in tu and c in a.get("labels", []):
                by_user[a["user_id"]].append(iid)
        sel += [("test", c, rng.choice(by_user[u])) for u in test_users]
    counts = defaultdict(int)
    for s, _, _ in sel:
        counts[s] += 1
    print("selected:", dict(counts))
    full_sel = list(sel)
    if args.only_splits:
        sel = [x for x in sel if x[0] in args.only_splits]
        print(f"downloading only {args.only_splits}: {len(sel)} images")

    # --- download --------------------------------------------------------------------------------
    out = args.out
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    zips = {}
    for c in classes:
        print(f"central directory of {c}.zip ...")
        z = open_remote_zip(CLASS_URL[c])
        zips[c] = {Path(i.filename).stem: i for i in z.infolist() if i.filename.lower().endswith(".jpg")}
        print(f"  {len(zips[c])} images")

    rows, lock, done = [], threading.Lock(), [0]
    t0 = time.time()

    def job(item):
        s, c, iid = item
        a = A[s][c][iid]
        name = f"{c}_{iid}.jpg"
        dst = out / "images" / s / name
        if dst.exists():
            img = cv2.imread(str(dst))
        else:
            info = zips[c].get(iid)
            if info is None:
                return None
            raw = read_member(CLASS_URL[c], info)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            if s != "test" and not args.keep_native and args.train_max_side and max(img.shape[:2]) > args.train_max_side:
                k = args.train_max_side / max(img.shape[:2])
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                dst.write_bytes(raw)  # test: original bytes, native resolution
        h, w = img.shape[:2]
        lines = [f"{cls_id[lab]} {bx + bw / 2:.6f} {by + bh / 2:.6f} {bw:.6f} {bh:.6f}"
                 for (bx, by, bw, bh), lab in zip(a["bboxes"], a["labels"]) if lab in cls_id]
        (out / "labels" / s / f"{c}_{iid}.txt").write_text("\n".join(lines) + "\n")
        meta = a.get("meta", {}) or {}
        row = {"split": s, "file": name, "gesture": c, "user_id": a["user_id"], "width": w, "height": h,
               "n_boxes": len(lines), "age": _first(meta.get("age")), "gender": _first(meta.get("gender")),
               "race": _first(meta.get("race"))}
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(sel)}  {time.time() - t0:.0f} s")
        return row

    with ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(job, sel):
            if r is None:
                print("  warning: an image could not be fetched")
            else:
                rows.append(r)

    order = {s: i for i, s in enumerate(SPLITS)}
    rows.sort(key=lambda r: (order[r["split"]], r["file"]))
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {i: c for i, c in enumerate(classes)}}, f, sort_keys=False)
    json.dump({s: sorted({r["user_id"] for r in rows if r["split"] == s}) for s in SPLITS},
              open(out / "split_users.json", "w"), indent=1)
    json.dump({"seed": args.seed, "source": "HaGRIDv2 official split", "classes": classes,
               "images": [list(x) for x in full_sel]}, open(out / "selection.json", "w"), indent=0)
    print(f"done in {time.time() - t0:.0f} s: {len(rows)} images -> {out}")


def _first(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return "" if v is None else v


if __name__ == "__main__":
    main()

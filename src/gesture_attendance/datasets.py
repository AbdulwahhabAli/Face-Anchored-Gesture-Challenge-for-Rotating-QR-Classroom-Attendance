"""Loading the evaluation datasets into a common `Sample` list.

* HaGRID test split, as produced by scripts/prepare_hagrid.py (YOLO layout + meta.csv)
* Team-collected classroom set: <root>/labels.csv + <root>/images/...

classroom labels.csv columns
    file                path relative to <root> (e.g. images/s07_like_low_2.jpg)
    subject_id          volunteer id (identity label)
    gesture             gesture the volunteer shows (HaGRID name) or "none"
    condition           normal | low_light | far_crowded  (free text, used for grouping)
    background_gesture  optional: gesture shown by SOMEONE ELSE in the background (attack images)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .vision.types import Box


@dataclass
class Sample:
    path: Path
    gesture: str
    subject: str
    condition: str = "all"
    background_gesture: str | None = None
    gt: list[tuple[str, Box]] = field(default_factory=list)  # gesture boxes (HaGRID only)

    @property
    def key(self) -> str:
        return self.path.name


def load_hagrid_split(root: str | Path, split: str = "test", class_names: list[str] | None = None) -> list[Sample]:
    root = Path(root)
    meta = root / "meta.csv"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found - run scripts/prepare_hagrid.py first")
    if class_names is None:
        import yaml
        class_names = yaml.safe_load(open(root / "data.yaml", encoding="utf-8"))["names"]
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names)]
    out = []
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            img = root / "images" / split / row["file"]
            w, h = float(row["width"]), float(row["height"])
            gt = []
            lbl = root / "labels" / split / (Path(row["file"]).stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    if not line.strip():
                        continue
                    c, cx, cy, bw, bh = line.split()
                    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                    gt.append((class_names[int(c)], Box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
            out.append(Sample(img, row["gesture"], row["user_id"], "hagrid", None, gt))
    return out


def load_classroom(root: str | Path) -> list[Sample]:
    root = Path(root)
    out = []
    with open(root / "labels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bg = (row.get("background_gesture") or "").strip() or None
            out.append(Sample(root / row["file"], row["gesture"].strip(), row["subject_id"].strip(),
                              (row.get("condition") or "all").strip(), bg))
    return out


def load_dataset(kind: str, root: str | Path, split: str = "test") -> list[Sample]:
    if kind == "hagrid":
        return load_hagrid_split(root, split)
    if kind == "classroom":
        return load_classroom(root)
    raise ValueError(kind)

"""Download pretrained model files into a local cache on first use."""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

CACHE_DIR = Path(os.environ.get("GA_CACHE_DIR", Path.home() / ".cache" / "gesture_attendance"))

BLAZEFACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
# InsightFace "buffalo_s": SCRFD-500MF face detector + ArcFace MobileFaceNet (w600k_mbf). Research use only.
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
BUFFALO_FILES = ("det_500m.onnx", "w600k_mbf.onnx")


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)
    return dst


def blazeface_model(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    dst = CACHE_DIR / "blaze_face_short_range.tflite"
    return dst if dst.exists() else _download(BLAZEFACE_URL, dst)


def arcface_dir(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    d = CACHE_DIR / "buffalo_s"
    if all((d / f).exists() for f in BUFFALO_FILES):
        return d
    z = _download(BUFFALO_S_URL, CACHE_DIR / "buffalo_s.zip")
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if Path(name).name in BUFFALO_FILES:
                with zf.open(name) as src, open(d / Path(name).name, "wb") as out:
                    shutil.copyfileobj(src, out)
    z.unlink(missing_ok=True)  # the zip also holds a 140 MB landmark model we do not need
    return d

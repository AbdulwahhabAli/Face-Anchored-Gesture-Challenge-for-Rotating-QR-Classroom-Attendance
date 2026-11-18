"""Fine-tune YOLOv8n (COCO-pretrained) on the HaGRID subset. The SAME weights serve both pipelines.

    python scripts/train_yolo.py --data data/hagrid_yolo --epochs 30 --device 0      # Colab T4
    python scripts/train_yolo.py --data data/hagrid_yolo --epochs 1 --fraction 0.1  # quick smoke test

Copies the best checkpoint to models/gesture_yolov8n.pt and writes test-split metrics
(full-frame mAP@0.5 etc.) to runs/train/<name>/test_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _common  # noqa: F401
import yaml

from gesture_attendance.config import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hagrid_yolo", type=Path, help="folder made by prepare_hagrid.py")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained COCO weights to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu'")
    ap.add_argument("--seed", type=int, default=441)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (smoke tests)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="gesture_yolov8n")
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "train"))
    args = ap.parse_args()

    from ultralytics import YOLO

    # re-write data.yaml with this machine's absolute path (dataset may have been prepared elsewhere)
    data_dir = args.data.resolve()
    cfg = yaml.safe_load(open(data_dir / "data.yaml", encoding="utf-8"))
    cfg["path"] = str(data_dir)
    local_yaml = data_dir / "data_local.yaml"
    yaml.safe_dump(cfg, open(local_yaml, "w", encoding="utf-8"), sort_keys=False)

    model = YOLO(args.model)
    model.train(data=str(local_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, seed=args.seed, deterministic=True, fraction=args.fraction,
                workers=args.workers, project=args.project, name=args.name, exist_ok=True,
                # left-right flips are fine for these gestures; keep the rest at Ultralytics defaults
                fliplr=0.5)

    run_dir = Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    dst = REPO_ROOT / "models" / "gesture_yolov8n.pt"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dst)
    print(f"best weights -> {dst}")

    # full-frame detection metrics on the held-out (subject-disjoint) HaGRID test split
    m = YOLO(str(dst)).val(data=str(local_yaml), split="test", imgsz=args.imgsz, device=args.device,
                           project=args.project, name=args.name + "_test", exist_ok=True)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    metrics = {
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class_mAP50": {names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

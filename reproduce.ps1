# Reproduces every number, table and figure in the report, from download to results/.
# Windows PowerShell, from the repository root, after the setup in README.md (Setup for the experiments).
# Total ≈ 2.5 h on an 8-core CPU (Ryzen 7 7800X3D), most of it YOLO training; no GPU needed.
# Each step skips nothing silently: re-running a step overwrites its outputs.
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# 1. data (≈ 3 min, ≈ 1.1 GB): HaGRIDv2 subset with the official subject-disjoint split, then LFW
& $py scripts/fetch_hagrid.py --out data/hagrid_yolo --seed 441
& $py scripts/make_conditions.py --src data/hagrid_yolo --out data/hagrid_eval --seed 441
if (-not (Test-Path data/lfw/lfw)) {
    New-Item -ItemType Directory -Force data/lfw | Out-Null
    curl.exe -L -o data/lfw/pairsDevTrain.txt https://ndownloader.figshare.com/files/5976012
    curl.exe -L -o data/lfw/pairsDevTest.txt  https://ndownloader.figshare.com/files/5976009
    curl.exe -L -o data/lfw/lfw.tgz           https://ndownloader.figshare.com/files/5976018
    tar -xzf data/lfw/lfw.tgz -C data/lfw
}

# 2. face threshold on LFW DevTrain, checked on DevTest (≈ 3 min)
& $py scripts/eval_face.py --lfw-root data/lfw/lfw --lfw-pairs data/lfw/pairsDevTrain.txt `
      --lfw-test-pairs data/lfw/pairsDevTest.txt --out runs/face

# 3. fine-tune YOLOv8n (≈ 60-70 min on CPU; add --device 0 on a GPU)
& $py scripts/train_yolo.py --data data/hagrid_yolo --epochs 15 --batch 16 --device cpu --workers 4 --cache ram

# 4. predictions of every config on the 2,400 test images, for both gesture detectors (≈ 15 min each)
$configs = "full", "roi_s2_up", "roi_s3_up", "roi_s4_up", "roi_s3_noup", "roi_s3_up_scrfd", "roi_s3_up_y0.5"
& $py scripts/predict.py --dataset hagrid --root data/hagrid_eval --out runs/preds/yolo --embeddings `
      --weights models/gesture_yolov8n.pt --set pipeline.roi_y_shift=0 --configs $configs
& $py scripts/predict.py --dataset hagrid --root data/hagrid_eval --out runs/preds/mediapipe --embeddings `
      --set pipeline.roi_y_shift=0 --configs $configs
& $py scripts/evaluate.py --preds runs/preds/yolo --out runs/eval/yolo
& $py scripts/evaluate.py --preds runs/preds/mediapipe --out runs/eval/mediapipe

# 5. latency on an idle machine (≈ 5 min)
& $py scripts/bench_latency.py --images-dir data/hagrid_yolo/images/test --n 100 --max-side 0 --device cpu `
      --weights models/gesture_yolov8n.pt --configs full roi_s3_up --set pipeline.roi_y_shift=0 --out runs/latency/yolo
& $py scripts/bench_latency.py --images-dir data/hagrid_yolo/images/test --n 100 --max-side 0 --device cpu `
      --configs full roi_s3_up --set pipeline.roi_y_shift=0 --out runs/latency/mediapipe

# 6. tables + figures used in the report and slides
& $py scripts/analyze.py --preds yolo=runs/preds/yolo mediapipe=runs/preds/mediapipe --out results
& $py scripts/report_figures.py --results results --eval-root data/hagrid_eval --out results/figures

# 7. demo backup recording (held-out images, baseline vs ours side by side)
& $py scripts/demo_compare.py --images data/hagrid_eval/images/test --meta data/hagrid_eval/meta.csv `
      --pick demo --weights models/gesture_yolov8n.pt --set pipeline.roi_y_shift=0 `
      --out results/demo --video results/demo/demo_backup.mp4

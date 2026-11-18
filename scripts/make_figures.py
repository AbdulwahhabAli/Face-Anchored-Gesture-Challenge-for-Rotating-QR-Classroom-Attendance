"""Figures for the report from the evaluation outputs.

    python scripts/make_figures.py --preds runs/preds/classroom --eval runs/eval/classroom \
        --images-root data/classroom --latency runs/latency/latency_cpu.csv --out runs/figures/classroom

Produces
    acc_by_condition.png   top-1 accuracy per condition and config (with 95% CI)
    roi_scale_ablation.png accuracy vs ROI scale, with / without up-scaling
    latency.png            median and p95 latency per config (if --latency)
    qualitative_*.png      ROI fixes / ROI breaks / both wrong examples with face box, ROI and detections
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _common  # noqa: F401
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def acc_by_condition(eval_dir: Path, out: Path):
    pc = pd.read_csv(eval_dir / "per_condition.csv")
    S = pd.read_csv(eval_dir / "summary.csv")
    allrows = S[["config", "n", "top1_acc", "acc_ci_lo", "acc_ci_hi"]].assign(condition="ALL")
    pc = pd.concat([pc, allrows], ignore_index=True)
    conds = list(dict.fromkeys(pc.condition))
    configs = list(S.config)
    w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(conds), 3.6))
    for k, c in enumerate(configs):
        sub = pc[pc.config == c].set_index("condition").reindex(conds)
        x = [i + (k - (len(configs) - 1) / 2) * w for i in range(len(conds))]
        err = [sub.top1_acc - sub.acc_ci_lo, sub.acc_ci_hi - sub.top1_acc]
        ax.bar(x, sub.top1_acc, w, yerr=err, capsize=2, label=c)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel("Top-1 gesture accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=min(3, len(configs)))
    fig.tight_layout()
    fig.savefig(out / "acc_by_condition.png", dpi=200)
    plt.close(fig)


def scale_ablation(eval_dir: Path, out: Path):
    S = pd.read_csv(eval_dir / "summary.csv")
    pat = re.compile(r"roi_s([\d.]+)_(up|noup)$")
    rows = [(float(m.group(1)), m.group(2), r.top1_acc, r.acc_ci_lo, r.acc_ci_hi)
            for r in S.itertuples() if (m := pat.match(r.config))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for mode in ("up", "noup"):
        pts = sorted(r for r in rows if r[1] == mode)
        if pts:
            ax.errorbar([p[0] for p in pts], [p[2] for p in pts],
                        yerr=[[p[2] - p[3] for p in pts], [p[4] - p[2] for p in pts]],
                        marker="o", capsize=3, label=f"ROI {'with' if mode == 'up' else 'without'} up-scaling")
    if (S.config == "full").any():
        ax.axhline(S.set_index("config").loc["full", "top1_acc"], color="gray", ls="--", label="full frame")
    ax.set_xlabel("ROI scale (x face box)")
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roi_scale_ablation.png", dpi=200)
    plt.close(fig)


def latency(csv: Path, out: Path):
    L = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(L.config, L.median_ms, label="median")
    ax.scatter(L.config, L.p95_ms, color="k", marker="_", s=300, label="p95", zorder=3)
    ax.set_ylabel("ms / image")
    ax.set_title(str(L.device.iloc[0]), fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{csv.stem}.png", dpi=200)
    plt.close(fig)


def draw(img, row, title):
    im = img.copy()
    t = max(2, im.shape[1] // 400)
    fb = json.loads(row.face_box) if isinstance(row.face_box, str) and row.face_box else None
    roi = json.loads(row.roi) if isinstance(row.roi, str) and row.roi else None
    if fb:
        cv2.rectangle(im, tuple(map(int, fb[:2])), tuple(map(int, fb[2:])), (255, 200, 0), t)
    if roi:
        cv2.rectangle(im, tuple(map(int, roi[:2])), tuple(map(int, roi[2:])), (255, 255, 255), t)
    for d in json.loads(row.dets)[:3]:
        b = list(map(int, d["box"]))
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), t)
        cv2.putText(im, f"{d['label']} {d['conf']:.2f}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, (0, 0, 255), t)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), title


def qualitative(pred_dir: Path, images_root: Path, out: Path, ours: str, base: str = "full", k: int = 4):
    A = pd.read_csv(pred_dir / f"{base}.csv", keep_default_na=False).set_index("file")
    B = pd.read_csv(pred_dir / f"{ours}.csv", keep_default_na=False).set_index("file")
    a_ok, b_ok = A.pred == A.gesture, B.pred == B.gesture
    groups = {"roi_fixes": (~a_ok) & b_ok, "roi_breaks": a_ok & (~b_ok), "both_wrong": (~a_ok) & (~b_ok)}
    for gname, mask in groups.items():
        files = list(mask[mask].index[:k])
        if not files:
            continue
        fig, axes = plt.subplots(len(files), 2, figsize=(7, 3.2 * len(files)), squeeze=False)
        for r, f in enumerate(files):
            img = cv2.imread(str(images_root / f))
            if img is None:
                continue
            for c, (name, D) in enumerate(((base, A), (ours, B))):
                row = D.loc[f]
                im, title = draw(img, row, f"{name}: pred={row.pred} (true={row.gesture}) [{row.condition}]")
                axes[r, c].imshow(im)
                axes[r, c].set_title(title, fontsize=7)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"qualitative_{gname}_{ours}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--eval", type=Path)
    ap.add_argument("--images-root", type=Path, help="dataset root the 'file' column is relative to")
    ap.add_argument("--ours", default="roi_s3_up")
    ap.add_argument("--latency", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.eval:
        acc_by_condition(args.eval, args.out)
        scale_ablation(args.eval, args.out)
    for csv in args.latency:
        latency(csv, args.out)
    if args.preds and args.images_root:
        qualitative(args.preds, args.images_root, args.out, args.ours)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()

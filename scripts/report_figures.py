"""Figures for the final report and slides (reads results/ written by scripts/analyze.py).

    python scripts/report_figures.py --results results --eval-root data/hagrid_eval --out results/figures

  pipeline.pdf/png        flowchart of the baseline and the proposed cascade + verification
  conditions.png          one held-out subject under the four test conditions
  acc_by_condition.png    top-1 accuracy, baseline vs ours, per condition, both gesture detectors
  ablation.png            ROI scale / up-scaling / y-shift / anchor ablation (YOLO + MediaPipe)
  qualitative.png         ROI fixes, ROI breaks (side by side, boxes drawn)
  failures.png            representative failures of the proposed pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BASE_C, OURS_C, ALT_C, ALT2_C = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
OURS_CFG = "roi_s3_up"
COND_ORDER = ["normal", "low_light", "far", "crowded"]
COND_TXT = {"normal": "normal", "low_light": "low light", "far": "far (2x zoom-out)", "crowded": "crowded",
            "all": "all"}


# ------------------------------------------------------------------------------------ flowchart
def _box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, bold=False, fs=5.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def _arrow(ax, p, q, col=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1, color=col,
                                 shrinkA=0, shrinkB=0))


def flowchart(out: Path):
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(-0.05, 15.3)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    blue_fc, grey_fc, green_fc = "#e8f1fc", "#f1f0ed", "#e6f6ef"
    # input
    _box(ax, 0.0, 2.05, 1.6, 1.0, "selfie\n(phone,\n≤12 MP)", bold=True)
    # baseline lane (top)
    ax.text(2.05, 4.85, "BASELINE  (full frame)", fontsize=7, color=INK2, fontweight="bold")
    _box(ax, 4.3, 3.9, 2.8, 0.8, "downscale whole frame\nto detector input", fc=grey_fc)
    _box(ax, 7.7, 3.9, 2.6, 0.8, "gesture detector\n(same pretrained model)", fc=grey_fc)
    _arrow(ax, (1.6, 2.85), (4.3, 4.3))
    _arrow(ax, (7.1, 4.3), (7.7, 4.3))
    # ours lane (middle)
    ax.text(2.05, 3.12, "OURS  (face-anchored ROI cascade)", fontsize=7, color=OURS_C, fontweight="bold")
    _box(ax, 2.0, 2.05, 2.0, 0.85, "face detector\n(SCRFD or BlazeFace)", fc=blue_fc, ec=OURS_C)
    _box(ax, 4.35, 2.05, 2.25, 0.85, "ROI = s × face box\n(s*=6 chosen on val)", fc=blue_fc, ec=OURS_C)
    _box(ax, 6.95, 2.05, 1.4, 0.85, "crop at\nnative res.", fc=blue_fc, ec=OURS_C)
    _box(ax, 8.7, 2.05, 1.95, 0.85, "same detector\non up-scaled crop", fc=blue_fc, ec=OURS_C)
    _arrow(ax, (1.6, 2.47), (2.0, 2.47))
    _arrow(ax, (4.0, 2.47), (4.35, 2.47))
    _arrow(ax, (6.6, 2.47), (6.95, 2.47))
    _arrow(ax, (8.35, 2.47), (8.7, 2.47))
    ax.text(3.0, 1.78, "no face → full frame", fontsize=6, color=MUTED, ha="center")
    # top-1
    _box(ax, 10.95, 2.95, 1.75, 0.95, "top-1 gesture\n(max conf.;\n'none' if empty)")
    _arrow(ax, (10.3, 4.3), (11.4, 3.9))
    _arrow(ax, (10.65, 2.47), (11.4, 2.95))
    # face verification lane (bottom)
    ax.text(2.05, 1.35, "IDENTITY  (both pipelines)", fontsize=7, color=INK2, fontweight="bold")
    _box(ax, 4.35, 0.2, 2.6, 0.85, "SCRFD-500MF face\n+ 5-point alignment", fc=green_fc, ec=ALT2_C)
    _box(ax, 7.35, 0.2, 3.3, 0.85, "ArcFace MobileFaceNet\ncos(emb, enrolled) ≥ τ", fc=green_fc, ec=ALT2_C)
    _arrow(ax, (0.8, 2.05), (0.8, 0.62))
    _arrow(ax, (0.8, 0.62), (4.35, 0.62))
    _arrow(ax, (6.95, 0.62), (7.35, 0.62))
    # decision
    _box(ax, 13.0, 1.5, 2.25, 1.45, "ACCEPT iff\ngesture =\nchallenge AND\nface matches", bold=True, fs=5.6)
    _arrow(ax, (12.7, 3.4), (13.6, 2.95))
    _arrow(ax, (10.65, 0.62), (14.1, 0.62))
    _arrow(ax, (14.1, 0.62), (14.1, 1.5))
    ax.text(14.15, 3.55, "random challenge\n(rotating-QR ticket)", fontsize=6, color=MUTED, ha="center")
    _arrow(ax, (14.15, 3.33), (14.15, 2.95), col=MUTED)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pipeline.{ext}", dpi=250)
    plt.close(fig)


# ------------------------------------------------------------------------------------ data figures
def conditions_strip(eval_root: Path, out: Path, source: str):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [0.75, 0.75, 0.75, 1.25]})
    for ax, cond in zip(axes, COND_ORDER):
        img = cv2.imread(str(eval_root / "images" / "test" / f"{cond}__{source}"))
        k = 700 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(COND_TXT[cond], fontsize=7.5, color=INK)
        ax.axis("off")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(out / "conditions.png", dpi=220)
    plt.close(fig)


def acc_by_condition(res: Path, out: Path):
    T = pd.read_csv(res / "main_comparison.csv")
    backends = list(dict.fromkeys(T.backend))
    variants = list(dict.fromkeys(T.variant))
    fig, axes = plt.subplots(1, len(backends), figsize=(7.0, 2.35), sharey=True, squeeze=False)
    cols = [OURS_C, ALT2_C]
    nb = 1 + len(variants)
    w = 0.84 / nb
    for ax, be in zip(axes[0], backends):
        x = np.arange(len(COND_ORDER) + 1)
        for k in range(nb):
            if k == 0:
                sub = T[(T.backend == be) & (T.variant == variants[0])].set_index("condition").reindex(COND_ORDER + ["all"])
                acc, lo, hi, c = sub.full_acc, sub.full_lo, sub.full_hi, BASE_C
                lab = "baseline (full frame)"
            else:
                v = variants[k - 1]
                sub = T[(T.backend == be) & (T.variant == v)].set_index("condition").reindex(COND_ORDER + ["all"])
                acc, lo, hi, c = sub.ours_acc, sub.ours_lo, sub.ours_hi, cols[k - 1]
                cfg = sub.ours_config.dropna().iloc[0]
                lab = ("ours, s=3, BlazeFace (proposal)" if v.startswith("s3") else
                       f"ours, s*={cfg.split('_')[1][1:]}, {'SCRFD' if cfg.endswith('scrfd') else 'BlazeFace'} (val-selected)")
            xs = x + (k - (nb - 1) / 2) * w
            ax.bar(xs, acc, w - 0.02, color=c, label=lab, zorder=2)
            ax.errorbar(xs, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor=INK2, lw=0.7, capsize=1.2, zorder=3)
            if k > 0:
                for xi, (d, sig) in enumerate(zip(sub.delta_pp, sub.p_value < 0.05)):
                    ax.text(xs[xi], hi.iloc[xi] + 0.015, f"{d:+.0f}{'*' if sig else ''}", ha="center",
                            fontsize=5.2, color=INK, rotation=90, va="bottom")
        ax.set_xticks(x, [COND_TXT[c].replace(" (2x zoom-out)", "") for c in COND_ORDER + ["all"]], fontsize=7)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_title(be, fontsize=8, color=INK)
    axes[0, 0].set_ylabel("top-1 gesture accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(out / "acc_by_condition.png", dpi=250)
    fig.savefig(out / "acc_by_condition.pdf")
    plt.close(fig)


def ablation(res: Path, out: Path):
    A = pd.read_csv(res / "ablation.csv")
    backends = list(dict.fromkeys(A.backend))
    fig, axes = plt.subplots(1, len(backends), figsize=(7.0, 2.2), sharey=True, squeeze=False)
    for ax, be in zip(axes[0], backends):
        sub = A[A.backend == be].copy()
        sub["s"] = sub.config.str.extract(r"roi_s([\d.]+)_")[0].astype(float)
        up = sub[sub.config.str.fullmatch(r"roi_s[\d.]+_up")].sort_values("s")
        sc = sub[sub.config.str.fullmatch(r"roi_s[\d.]+_up_scrfd")].sort_values("s")
        base = sub[sub.config == "full"].iloc[0]
        for cond, ls, mk in (("all", "-", "o"), ("far", "--", "s")):
            ax.plot(up.s, up[f"acc_{cond}"], ls=ls, marker=mk, ms=3.5, lw=1.5, color=OURS_C,
                    label=f"ROI, BlazeFace anchor [{cond}]")
            if len(sc):
                ax.plot(sc.s, sc[f"acc_{cond}"], ls=ls, marker=mk, ms=3.5, lw=1.5, color=ALT2_C,
                        label=f"ROI, SCRFD anchor [{cond}]")
            ax.axhline(base[f"acc_{cond}"], color=BASE_C, lw=1.1, ls=ls, label=f"full frame [{cond}]")
        for cfg, mk, lab in (("roi_s3_noup", "x", "s=3, no up-scaling [all]"), ("roi_s3_up_y0.5", "^",
                                                                              "s=3, ROI shifted down 0.5 face [all]")):
            r = sub[sub.config == cfg]
            if len(r):
                ax.scatter([3.0], r.acc_all, marker=mk, s=22, color=ALT_C if mk == "x" else INK2, zorder=4, label=lab)
        ax.set_xticks(sorted(up.s.unique()))
        ax.set_xlabel("ROI side / face-box side (s)")
        ax.set_ylim(0, 1.0)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(be, fontsize=8, color=INK)
    axes[0, 0].set_ylabel("top-1 accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.0, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=250)
    fig.savefig(out / "ablation.pdf")
    plt.close(fig)


def _draw(img, row, thick):
    im = img.copy()
    fb = json.loads(row["face_box"]) if row["face_box"] not in ("", "null") else None
    roi = json.loads(row["roi"]) if row["roi"] not in ("", "null") else None
    if fb:
        cv2.rectangle(im, tuple(map(int, fb[:2])), tuple(map(int, fb[2:])), (214, 120, 42), thick)
    if roi:
        cv2.rectangle(im, tuple(map(int, roi[:2])), tuple(map(int, roi[2:])), (255, 255, 255), thick)
        cv2.rectangle(im, tuple(map(int, roi[:2])), tuple(map(int, roi[2:])), (0, 0, 0), max(1, thick // 3))
    for d in json.loads(row["dets"])[:3]:
        b = list(map(int, d["box"]))
        col = (60, 170, 30) if d["label"] == row["gesture"] else (40, 40, 220)
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, thick)
    return im


def grid(pred_dir: Path, eval_root: Path, items: list, out_path: Path, ours=OURS_CFG,
         ncols=4, width=7.0, row_h=2.75, label_ours=None):
    """items: (file, caption[, config]) - each cell shows baseline above ours (that config) for one image."""
    cache = {}

    def _load(name):
        if name not in cache:
            D = pd.read_csv(pred_dir / f"{name}.csv", keep_default_na=False)
            D["key"] = D.file.map(lambda f: Path(f.replace(chr(92), "/")).name)
            cache[name] = D.set_index("key")
        return cache[name]
    A = _load("full")
    items = [(Path(it[0].replace(chr(92), "/")).name, it[1], it[2] if len(it) > 2 else ours) for it in items]
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(width, row_h * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (f, cap, cfg) in enumerate(items):
        r, c = divmod(k, ncols)
        img = cv2.imread(str(eval_root / "images" / "test" / f))
        thick = max(3, img.shape[1] // 250)
        B = _load(cfg)
        lab = (label_ours or {}).get(cfg, "ours")
        for j, (name, D) in enumerate((("baseline", A), (lab, B))):
            row = D.loc[f]
            im = _draw(img, row, thick)
            kk = 520 / max(im.shape[:2])
            im = cv2.resize(im, None, fx=kk, fy=kk, interpolation=cv2.INTER_AREA)
            ax = axes[2 * r + j, c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ok = row["pred"] == row["gesture"]
            ax.set_title(f"{name}: {row['pred']} {'✓' if ok else '✗'}", fontsize=6.5,
                         color=("#0a7a35" if ok else "#c0261f"), pad=2)
            if j == 0:
                ax.text(0.0, 1.17, cap, transform=ax.transAxes, fontsize=6.3, color=INK, ha="left")
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _bgr(hexcol):
    h = hexcol.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def tiles(pred_dir: Path, eval_root: Path, rows: list, out_path: Path, labels: dict, width=7.0, row_titles=None):
    """One zoomed tile per example: ROI (white), face (cyan), baseline top detection (orange), ours (blue)."""
    cache = {}

    def load(name):
        if name not in cache:
            D = pd.read_csv(pred_dir / f"{name}.csv", keep_default_na=False)
            D["key"] = D.file.map(lambda f: Path(f.replace(chr(92), "/")).name)
            cache[name] = D.set_index("key")
        return cache[name]

    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(width, 2.05 * len(rows) + 0.15), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ri, items in enumerate(rows):
        for ci, (f, cap, cfg) in enumerate(items):
            key = Path(f.replace(chr(92), "/")).name
            a, b = load("full").loc[key], load(cfg).loc[key]
            img = cv2.imread(str(eval_root / "images" / "test" / key))
            H_, W_ = img.shape[:2]
            t = max(3, W_ // 300)
            boxes = []
            roi = json.loads(b["roi"]) if b["roi"] not in ("", "null") else None
            fb = json.loads(b["face_box"]) if b["face_box"] not in ("", "null") else None
            da, db = json.loads(a["dets"])[:1], json.loads(b["dets"])[:1]
            if roi:
                cv2.rectangle(img, tuple(map(int, roi[:2])), tuple(map(int, roi[2:])), (0, 0, 0), t + 4)
                cv2.rectangle(img, tuple(map(int, roi[:2])), tuple(map(int, roi[2:])), (255, 255, 255), t)
                boxes.append(roi)
            if fb:
                cv2.rectangle(img, tuple(map(int, fb[:2])), tuple(map(int, fb[2:])), (230, 200, 0), t)
                boxes.append(fb)
            for d, col in ((da, ALT_C), (db, OURS_C)):
                for x in d:
                    bb = list(map(int, x["box"]))
                    cv2.rectangle(img, tuple(bb[:2]), tuple(bb[2:]), _bgr(col), t + 1)
                    boxes.append(bb)
            if not boxes:  # nothing detected: show the claimant's half / centre
                boxes.append([W_ * 0.25, H_ * 0.25, W_ * 0.75, H_ * 0.75])
            B_ = np.array(boxes, float)
            x1, y1, x2, y2 = B_[:, 0].min(), B_[:, 1].min(), B_[:, 2].max(), B_[:, 3].max()
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.25, (y2 - y1) * 1.25
            w, h = max(w, h * 0.8), max(h, w / 0.8 * 0.6)
            X1, Y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
            X2, Y2 = int(min(W_, cx + w / 2)), int(min(H_, cy + h / 2))
            crop = img[Y1:Y2, X1:X2]
            k = 420 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            ax = axes[ri, ci]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ok_a, ok_b = a["pred"] == a["gesture"], b["pred"] == b["gesture"]
            ax.set_title(cap, fontsize=6.4, color=INK, pad=11)
            ax.text(0.5, 1.012, f"base: {a['pred']} {'✓' if ok_a else '✗'}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.8, color=("#0a7a35" if ok_a else "#c0261f"))
            ax.text(0.52, 1.012, f"  {labels.get(cfg, 'ours').replace('ours ', '')}: {b['pred']} "
                    f"{'✓' if ok_b else '✗'}", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8,
                    color=("#0a7a35" if ok_b else "#c0261f"))
        if row_titles:
            axes[ri, 0].text(-0.06, 0.5, row_titles[ri], transform=axes[ri, 0].transAxes, rotation=90,
                             ha="right", va="center", fontsize=6.5, color=INK2)
    fig.tight_layout(h_pad=0.9, w_pad=0.4)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--eval-root", type=Path, default=Path("data/hagrid_eval"))
    ap.add_argument("--preds", type=Path, default=Path("runs/preds/yolo"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = set(args.only or ["pipeline", "conditions", "acc", "ablation", "qualitative"])
    if "pipeline" in todo:
        flowchart(args.out)
    if "conditions" in todo:
        sel = json.loads((args.results / "examples.json").read_text()) if (args.results / "examples.json").exists() \
            else {}
        conditions_strip(args.eval_root, args.out, sel.get("conditions_source") or
                         sorted(p.name.split("__", 1)[1] for p in (args.eval_root / "images" / "test").glob("normal__*"))[0])
    if "acc" in todo:
        acc_by_condition(args.results, args.out)
    if "ablation" in todo:
        ablation(args.results, args.out)
    if "qualitative" in todo:
        ex = json.loads((args.results / "examples.json").read_text())
        labels = ex.get("labels", {})
        rows = [[(e["file"], e["caption"], e["config"]) for e in ex[k]] for k in ("qualitative", "failures")]
        tiles(Path(ex["preds"]), args.eval_root, rows, args.out / "qualitative.png", labels,
              row_titles=["Successes (and the s=3 failure)", "Failures of the val-selected ROI"])
        for key, r in zip(("qualitative", "failures"), rows):  # single-row versions for the slides
            tiles(Path(ex["preds"]), args.eval_root, [r], args.out / f"{key}_slide.png", labels, width=7.0)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()

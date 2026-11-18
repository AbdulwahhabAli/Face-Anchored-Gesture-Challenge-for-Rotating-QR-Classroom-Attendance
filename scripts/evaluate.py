"""Gesture metrics + controlled comparison from the cached predictions of predict.py.

    python scripts/evaluate.py --preds runs/preds/classroom --out runs/eval/classroom
    python scripts/evaluate.py --preds runs/preds/hagrid    --out runs/eval/hagrid

Per configuration: top-1 accuracy (95% bootstrap CI), accuracy per condition, per-class
precision/recall, mAP@0.5 (when GT boxes exist, i.e. HaGRID), face-found / fallback rates,
wrong-gesture FAR and background-hand FAR (classroom attack images).
Versus the baseline: paired accuracy delta with bootstrap CI and exact McNemar test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd

from gesture_attendance.evaluation import (bootstrap_ci, confusion, map_at_iou, mcnemar_exact,
                                           paired_bootstrap_delta, per_class_pr, wrong_gesture_far)
from gesture_attendance.gestures import NONE_LABEL
from gesture_attendance.vision.types import Box


def load_preds(pred_dir: Path) -> dict[str, pd.DataFrame]:
    info = json.loads((pred_dir / "run_info.json").read_text())
    out = {}
    for c in info["configs"]:
        df = pd.read_csv(pred_dir / f"{c}.csv", keep_default_na=False)
        out[c] = df
    return info, out


def det_lists(df: pd.DataFrame):
    preds = [[(d["label"], d["conf"], Box.from_list(d["box"])) for d in json.loads(s)] for s in df.dets]
    gts = [[(lab, Box.from_list(b)) for lab, b in json.loads(s)] for s in df["gt"]]
    return preds, gts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--baseline", default="full")
    ap.add_argument("--h1-pp", type=float, default=5.0, help="H1 accuracy gain threshold (percentage points)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info, preds = load_preds(args.preds)
    classes = info["classes"]
    summary, per_cond, per_cls, comp = [], [], [], []
    base = preds.get(args.baseline)

    for name, df in preds.items():
        # gesture-labelled images only for accuracy (attack-only images with gesture "none" excluded)
        ev = df[df.gesture.isin(classes)]
        correct = (ev.pred == ev.gesture).values
        lo, hi = bootstrap_ci(correct)
        row = {"config": name, "n": len(ev), "top1_acc": correct.mean(), "acc_ci_lo": lo, "acc_ci_hi": hi,
               "no_detection_rate": (ev.pred == NONE_LABEL).mean(),
               "wrong_gesture_FAR": wrong_gesture_far(ev.gesture, ev.pred, classes)}
        if name != "full":
            row["face_found_rate"] = (df.face_found.astype(str) == "True").mean()
            row["fallback_rate"] = (df.fallback.astype(str) == "True").mean()
        bg = df[(df.background_gesture != "") & (df.background_gesture != df.gesture)]
        if len(bg):
            row["n_background_attacks"] = len(bg)
            row["background_hand_FAR"] = (bg.pred == bg.background_gesture).mean()
        p, g = det_lists(ev)
        if any(g):
            mp = map_at_iou(p, g, classes, 0.5)
            row["mAP50"] = mp["mAP"]
            for c, v in mp["per_class"].items():
                per_cls.append({"config": name, "class": c, "metric": "AP50", "value": v})
        row["median_gesture_ms"] = df.t_gesture_ms.median()
        summary.append(row)

        for cond, sub in ev.groupby("condition"):
            c_ok = (sub.pred == sub.gesture).values
            lo, hi = bootstrap_ci(c_ok)
            per_cond.append({"config": name, "condition": cond, "n": len(sub), "top1_acc": c_ok.mean(),
                             "acc_ci_lo": lo, "acc_ci_hi": hi})
        for r in per_class_pr(ev.gesture, ev.pred, classes):
            per_cls.append({"config": name, "class": r["class"], "metric": "precision", "value": r["precision"]})
            per_cls.append({"config": name, "class": r["class"], "metric": "recall", "value": r["recall"]})

        cm = confusion(ev.gesture, ev.pred, classes)
        pd.DataFrame(cm, index=classes, columns=classes + [NONE_LABEL]).to_csv(args.out / f"confusion_{name}.csv")

        if base is not None and name != args.baseline:
            b = base.set_index("file").loc[ev.file]
            a_ok = (b.pred.values == b.gesture.values)
            d, dlo, dhi = paired_bootstrap_delta(a_ok, correct)
            mc = mcnemar_exact(a_ok, correct)
            comp.append({"config": name, "vs": args.baseline, "condition": "all", "delta_pp": 100 * d,
                         "ci_lo_pp": 100 * dlo, "ci_hi_pp": 100 * dhi, **mc,
                         "H1_gain_met": 100 * d >= args.h1_pp})
            for cond in sorted(ev.condition.unique()):
                msk = (ev.condition == cond).values
                d, dlo, dhi = paired_bootstrap_delta(a_ok[msk], correct[msk])
                mc = mcnemar_exact(a_ok[msk], correct[msk])
                comp.append({"config": name, "vs": args.baseline, "condition": cond, "delta_pp": 100 * d,
                             "ci_lo_pp": 100 * dlo, "ci_hi_pp": 100 * dhi, **mc,
                             "H1_gain_met": 100 * d >= args.h1_pp})

    S = pd.DataFrame(summary)
    S.to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(per_cond).to_csv(args.out / "per_condition.csv", index=False)
    pd.DataFrame(per_cls).to_csv(args.out / "per_class.csv", index=False)
    C = pd.DataFrame(comp)
    C.to_csv(args.out / "comparison.csv", index=False)

    md = [f"# Gesture evaluation: {info['dataset']} ({info['n_images']} images)\n", "## Summary\n", _md(S), ""]
    if len(per_cond):
        pc = pd.DataFrame(per_cond).pivot(index="condition", columns="config", values="top1_acc")
        md += ["## Top-1 accuracy by condition\n", _md(pc.reset_index()), ""]
    if len(C):
        md += [f"## Paired comparison vs `{args.baseline}` (H1: gain >= {args.h1_pp} pp)\n", _md(C), ""]
    (args.out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"


if __name__ == "__main__":
    main()

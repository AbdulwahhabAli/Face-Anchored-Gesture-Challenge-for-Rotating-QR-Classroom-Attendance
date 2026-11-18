"""Final result tables for the report, from the cached predictions (no model is re-run).

    python scripts/analyze.py --preds yolo=runs/preds/yolo mediapipe=runs/preds/mediapipe \
        --latency runs/latency --face runs/face --out results

Writes to <out>/:
  main_comparison.csv  baseline (full) vs ours (roi_s3_up) per backend x condition: accuracy + 95% CI,
                       paired delta + 95% CI, exact McNemar p, #fixed / #broken images
  ablation.csv         every config: accuracy overall / per condition, face-found, fallback, background-hand FAR
  security.csv         background-hand FAR and wrong-gesture FAR per backend, baseline vs ours
  e2e.csv              end-to-end FRR / FAR (gesture AND ArcFace) per backend, baseline vs ours
  bias.csv             accuracy by HaGRID's (automatic) gender / race / age-group annotations
  cost.csv             latency (median / p95) and model sizes
  examples.json        deterministic picks for the qualitative and failure figures
  results.md           all of the above as markdown
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd

from gesture_attendance.evaluation import (bootstrap_ci, choose_enrollment, load_threshold, mcnemar_exact,
                                           paired_bootstrap_delta, wrong_gesture_far)
from gesture_attendance.gestures import NONE_LABEL

BACKEND_TXT = {"yolo": "YOLOv8n (fine-tuned on HaGRID)", "mediapipe": "MediaPipe Gesture Recognizer (pretrained)"}
CONDS = ["normal", "low_light", "far", "crowded"]
OURS = "roi_s3_up"
SRC_WIDTH: dict = {}


def load(pred_dir: Path):
    info = json.loads((pred_dir / "run_info.json").read_text())
    dfs = {c: pd.read_csv(pred_dir / f"{c}.csv", keep_default_na=False) for c in info["configs"]}
    return info, dfs


def attach_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.set_index("file")
    df = df.copy()
    key = df.file.str.replace("\\", "/").str.split("/").str[-1]
    for col in ("source_file", "background_user"):
        if col in m.columns:
            df[col] = key.map(m[col]).values
    if SRC_WIDTH:  # claimant's width inside a crowded canvas (the background person is pasted to its right)
        df["src_width"] = df.source_file.map(SRC_WIDTH).values
    # HaGRID's demographic labels are automatic and per image, so they can disagree between images of one
    # person: give every subject the majority gender / race and the median age over their images
    g = meta.replace("", np.nan).groupby("user_id")
    subj = pd.DataFrame({"gender": g.gender.agg(lambda x: x.mode().iloc[0] if x.notna().any() else "unknown"),
                         "race": g.race.agg(lambda x: x.mode().iloc[0] if x.notna().any() else "unknown"),
                         "age": g.age.agg(lambda x: pd.to_numeric(x, errors="coerce").median())})
    for col in ("gender", "race", "age"):
        df[col] = df.subject.map(subj[col]).values
    return df


def select_scale(val_dir: Path, classes) -> dict:
    """ROI anchor + scale chosen on the HaGRID VAL split with the same 4 conditions (never on test):
    best accuracy; ties -> BlazeFace (the proposal's anchor), then the smaller scale."""
    import re
    info, dfs = load(val_dir)
    acc = {}
    for name, df in dfs.items():
        ev = df[df.gesture.isin(classes)]
        acc[name] = float((ev.pred == ev.gesture).mean())
    roi = {n: a for n, a in acc.items() if re.fullmatch(r"roi_s[\d.]+_up(_scrfd)?", n)}
    best = sorted(roi, key=lambda n: (-round(roi[n], 6), n.endswith("_scrfd"), float(n.split("_")[1][1:])))[0]
    return {"selected": best, "val_accuracy": acc, "n_val": int(len(next(iter(dfs.values())))),
            "rule": "argmax val top-1 over anchor {BlazeFace, SCRFD} x s {2..6}; ties: BlazeFace, smaller s"}


def main_comparison(backend, dfs, classes, variants):
    rows = []
    for variant, cfg in variants.items():
        rows += _comparison(backend, dfs, classes, variant, cfg)
    return rows


def _comparison(backend, dfs, classes, variant, cfg):
    rows = []
    base, ours = dfs["full"], dfs[cfg]
    for cond in CONDS + ["all"]:
        msk = base.gesture.isin(classes) & ((base.condition == cond) if cond != "all" else True)
        a = (base.pred == base.gesture)[msk].values
        b = (ours.pred == ours.gesture)[msk].values
        lo_a, hi_a = bootstrap_ci(a)
        lo_b, hi_b = bootstrap_ci(b)
        d, dlo, dhi = paired_bootstrap_delta(a, b)
        mc = mcnemar_exact(a, b)
        rows.append({"backend": BACKEND_TXT[backend], "variant": variant, "ours_config": cfg,
                     "condition": cond, "n": int(msk.sum()),
                     "full_acc": a.mean(), "full_lo": lo_a, "full_hi": hi_a,
                     "ours_acc": b.mean(), "ours_lo": lo_b, "ours_hi": hi_b,
                     "delta_pp": 100 * d, "delta_lo_pp": 100 * dlo, "delta_hi_pp": 100 * dhi,
                     "p_value": mc["p_value"], "fixed": mc["n01_b_fixes"], "broken": mc["n10_b_breaks"]})
    return rows


def ablation_rows(backend, dfs, classes):
    rows = []
    for name, df in dfs.items():
        ev = df[df.gesture.isin(classes)]
        ok = ev.pred == ev.gesture
        r = {"backend": BACKEND_TXT[backend], "config": name, "acc_all": ok.mean()}
        for c in CONDS:
            r[f"acc_{c}"] = ok[ev.condition == c].mean()
        if name != "full":
            r["face_found_rate"] = (df.face_found.astype(str) == "True").mean()
            r["fallback_rate"] = (df.fallback.astype(str) == "True").mean()
            for c in CONDS:
                r[f"face_found_{c}"] = (df[df.condition == c].face_found.astype(str) == "True").mean()
            if "src_width" in df.columns:  # crowded: did the anchor pick the background person's face?
                cr = df[(df.condition == "crowded") & (df.face_box != "null") & (df.face_box != "")]
                cx = cr.face_box.map(lambda b: (json.loads(b)[0] + json.loads(b)[2]) / 2)
                r["wrong_anchor_crowded"] = float((cx > cr.src_width).mean()) if len(cr) else float("nan")
        bg = df[(df.condition == "crowded") & df.background_gesture.isin(classes) & df.gesture.isin(classes)]
        r["background_hand_FAR"] = (bg.pred == bg.background_gesture).mean()
        r["no_detection_rate"] = (ev.pred == NONE_LABEL).mean()
        r["median_gesture_ms"] = df.t_gesture_ms.median()
        rows.append(r)
    return rows


def security_rows(backend, dfs, classes, configs):
    rows = []
    for name in configs:
        if name not in dfs:
            continue
        df = dfs[name]
        bg = df[(df.condition == "crowded") & df.background_gesture.isin(classes) & df.gesture.isin(classes)]
        ev = df[df.gesture.isin(classes)]
        rows.append({"backend": BACKEND_TXT[backend], "config": name, "n_background_attacks": len(bg),
                     "background_hand_FAR": (bg.pred == bg.background_gesture).mean(),
                     "claimant_gesture_correct_in_crowded": (bg.pred == bg.gesture).mean(),
                     "wrong_gesture_FAR": wrong_gesture_far(ev.gesture, ev.pred, classes)})
    return rows


def e2e_rows(backend, pred_dir, dfs, classes, thr, configs, max_attempts=3):
    """Gesture AND face. Enrollment = each subject's first 'normal' image (excluded as a probe)."""
    z = np.load(pred_dir / "embeddings.npz", allow_pickle=True)
    files = [str(f) for f in z["files"]]
    emb, has = z["emb"], z["has_face"].astype(bool)
    subj, cond = [str(s) for s in z["subjects"]], [str(c) for c in z["conditions"]]
    enroll = choose_enrollment(files, subj, cond, has)
    eidx = set(enroll.values())
    E = np.stack([emb[enroll[s]] for s in sorted(enroll)])
    S = sorted(enroll)
    sims_all = emb @ E.T  # (n_images, n_subjects)
    col = {s: k for k, s in enumerate(S)}
    fix = {f: i for i, f in enumerate(files)}
    rows = []
    for name in configs:
        df = dfs[name]
        c = {"genuine": [0, 0], "impostor": [0, 0], "wrong_gesture": [0, 0], "background": [0, 0], "replay": [0, 0]}
        by_cond = {}
        for r in df.itertuples():
            i = fix[r.file]
            if i in eidx or r.subject not in enroll or r.gesture not in classes:
                continue
            own = sims_all[i, col[r.subject]] if has[i] else -1.0
            face_ok = own >= thr
            ok = (r.pred == r.gesture) and face_ok
            c["genuine"][0] += int(not ok)
            c["genuine"][1] += 1
            bc = by_cond.setdefault(r.condition, [0, 0])
            bc[0] += int(not ok)
            bc[1] += 1
            others = np.delete(sims_all[i], col[r.subject]) if has[i] else np.full(len(S) - 1, -1.0)
            c["impostor"][0] += int(r.pred == r.gesture) * int((others >= thr).sum())
            c["impostor"][1] += len(others)
            for g in classes:
                if g != r.gesture:
                    c["wrong_gesture"][0] += int(r.pred == g and face_ok)
                    c["wrong_gesture"][1] += 1
            if r.condition == "crowded" and r.background_gesture in classes:
                c["background"][0] += int(r.pred == r.background_gesture and face_ok)
                c["background"][1] += 1
            if r.condition == "normal":  # a stored photo replayed against a random challenge
                c["replay"][0] += (1.0 / len(classes)) * int(r.pred in classes and face_ok)
                c["replay"][1] += 1
        row = {"backend": BACKEND_TXT[backend], "config": name, "threshold": thr,
               "FRR": c["genuine"][0] / c["genuine"][1], "n_genuine": c["genuine"][1],
               "FAR_impostor_face": c["impostor"][0] / c["impostor"][1], "n_impostor": c["impostor"][1],
               "FAR_wrong_gesture": c["wrong_gesture"][0] / c["wrong_gesture"][1],
               "FAR_background_hand": c["background"][0] / max(1, c["background"][1]),
               "FAR_replay_per_attempt": c["replay"][0] / c["replay"][1]}
        row[f"FAR_replay_{max_attempts}_attempts"] = 1 - (1 - row["FAR_replay_per_attempt"]) ** max_attempts
        for k, (bad, n) in by_cond.items():
            row[f"FRR_{k}"] = bad / n
        rows.append(row)
    # face-only verification on this set (same threshold): genuine vs impostor similarity
    gen, imp = [], []
    for i in range(len(files)):
        if i in eidx or not has[i] or subj[i] not in enroll or cond[i] != "normal":
            continue
        gen.append(sims_all[i, col[subj[i]]])
        imp += list(np.delete(sims_all[i], col[subj[i]]))
    gen, imp = np.array(gen), np.array(imp)
    face = {"backend": "ArcFace (shared)", "n_genuine": len(gen), "n_impostor": len(imp),
            "FRR_normal": float((gen < thr).mean()), "FAR_normal": float((imp >= thr).mean()),
            "no_face_rate_by_condition": {c_: float(1 - has[[k for k in range(len(files)) if cond[k] == c_]].mean())
                                          for c_ in CONDS}}
    return rows, face


def bias_rows(backend, dfs, classes, configs):
    rows = []
    for name in configs:
        df = dfs[name]
        ev = df[df.gesture.isin(classes)].copy()
        ev["ok"] = ev.pred == ev.gesture
        ev["age_group"] = pd.cut(pd.to_numeric(ev.age, errors="coerce"), [0, 25, 40, 120],
                                 labels=["<25", "25-40", ">40"])
        for attr in ("gender", "race", "age_group"):
            for val, sub in ev.groupby(attr, observed=True):
                lo, hi = bootstrap_ci(sub.ok.values)
                rows.append({"backend": BACKEND_TXT[backend], "config": name, "attribute": attr,
                             "group": str(val), "n_images": len(sub), "n_subjects": sub.subject.nunique(),
                             "acc": sub.ok.mean(), "lo": lo, "hi": hi})
    return rows


def taxonomy_of(pred_dir: Path, classes, ours):
    """Why does `ours` fail? no face (-> full-frame fallback), no gesture in the ROI, or a wrong class."""
    A = pd.read_csv(pred_dir / "full.csv", keep_default_na=False)
    B = pd.read_csv(pred_dir / f"{ours}.csv", keep_default_na=False)
    valid = B.gesture.isin(classes)
    bad = valid & (B.pred != B.gesture)
    face = B.face_found.astype(str) == "True"
    t = {"config": ours, "errors": int(bad.sum()), "no_face_found": int((bad & ~face).sum()),
         "no_detection_in_roi": int((bad & face & (B.pred == NONE_LABEL)).sum()),
         "wrong_class": int((bad & face & (B.pred != NONE_LABEL)).sum()),
         "baseline_also_wrong": int((bad & (A.pred != A.gesture)).sum())}
    for c in CONDS:
        t[f"errors_{c}"] = int((bad & (B.condition == c)).sum())
    return t, pd.crosstab(B.gesture[bad], B.pred[bad])


def pick_examples(pred_dir: Path, classes, tuned: str, s3: str = OURS, seed=3):
    """Deterministic figure picks. Qualitative: what the val-selected ROI fixes (far, low light, crowded) and
    how the proposal's s=3 ROI breaks a correct baseline; failures: errors of the val-selected ROI."""
    rng = np.random.default_rng(seed)
    A = pd.read_csv(pred_dir / "full.csv", keep_default_na=False)
    T = pd.read_csv(pred_dir / f"{tuned}.csv", keep_default_na=False)
    S3 = pd.read_csv(pred_dir / f"{s3}.csv", keep_default_na=False)
    a_ok, t_ok, s_ok = A.pred == A.gesture, T.pred == T.gesture, S3.pred == S3.gesture
    valid = A.gesture.isin(classes)
    qual, fail = [], []
    for cond in ("far", "low_light", "crowded"):
        idx = np.flatnonzero(valid & ~a_ok & t_ok & (A.condition == cond))
        if len(idx):
            i = int(rng.choice(idx))
            qual.append({"file": A.file[i], "config": tuned, "caption": f"fixed by ROI ({cond.replace('_', ' ')})"})
    idx = np.flatnonzero(valid & a_ok & ~s_ok & (S3.pred == NONE_LABEL) & (A.condition == "normal"))
    if len(idx):
        i = int(rng.choice(idx))
        qual.append({"file": A.file[i], "config": s3, "caption": "s=3 cuts the hand off"})
    face = T.face_found.astype(str) == "True"
    bad = valid & ~t_ok
    for name, m in (("anchored on the other person", bad & (T.condition == "crowded") & (T.pred == T.background_gesture)),
                    ("missed in low light", bad & (T.condition == "low_light") & (T.pred == NONE_LABEL)),
                    ("gesture confused", bad & face & (T.pred != NONE_LABEL) & (T.condition == "normal")),
                    ("ROI misses a visible hand", bad & a_ok)):
        idx = np.flatnonzero(m)
        if len(idx):
            i = int(rng.choice(idx))
            fail.append({"file": T.file[i], "config": tuned, "caption": f"{name} (true {T.gesture[i]})"})
    return qual, fail


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True, help="backend=dir pairs")
    ap.add_argument("--meta", type=Path, default=Path("data/hagrid_eval/meta.csv"))
    ap.add_argument("--latency", type=Path, default=Path("runs/latency"))
    ap.add_argument("--face", type=Path, default=Path("runs/face"))
    ap.add_argument("--train-metrics", type=Path, default=Path("runs/train/gesture_yolov8n/test_metrics.json"))
    ap.add_argument("--examples-from", default="yolo")
    ap.add_argument("--val-preds", nargs="*", default=[], help="backend=dir pairs of predict.py --split val runs")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.meta, keep_default_na=False)
    src_meta = Path("data/hagrid_yolo/meta.csv")
    if src_meta.exists():
        sm = pd.read_csv(src_meta, keep_default_na=False)
        SRC_WIDTH.update(dict(zip(sm[sm.split == "test"].file, sm[sm.split == "test"].width)))
    thr = load_threshold(args.face / "threshold.json")

    M, A, S, E, B, faces, tax = [], [], [], [], [], [], {}
    pred_dirs = dict(p.split("=", 1) for p in args.preds)
    val_dirs = dict(p.split("=", 1) for p in args.val_preds)
    selection = {}
    for be, d in pred_dirs.items():
        d = Path(d)
        info, dfs = load(d)
        classes = info["classes"]
        dfs = {k: attach_meta(v, meta) for k, v in dfs.items()}
        variants = {"s3 (pre-registered)": OURS}
        if be in val_dirs:
            selection[be] = select_scale(Path(val_dirs[be]), classes)
            if selection[be]["selected"] not in dfs:
                raise SystemExit(f"{be}: val-selected {selection[be]['selected']} missing from the test predictions")
            variants["s* (val-selected)"] = selection[be]["selected"]
        cfgs = ["full"] + list(dict.fromkeys(variants.values()))
        M += main_comparison(be, dfs, classes, variants)
        A += ablation_rows(be, dfs, classes)
        S += security_rows(be, dfs, classes, cfgs + [c for c in ("roi_s3_up_scrfd",) if c in dfs])
        e, f = e2e_rows(be, d, dfs, classes, thr, cfgs)
        E += e
        faces.append(f)
        B += bias_rows(be, dfs, classes, cfgs)
        tax[be] = {}
        for v, cfg in variants.items():
            t, conf = taxonomy_of(d, classes, cfg)
            tax[be][v] = t
            conf.to_csv(args.out / f"errors_confusion_{be}_{cfg}.csv")
        if be == args.examples_from:
            tuned = variants.get("s* (val-selected)", OURS)
            qual, fail = pick_examples(d, classes, tuned)
            src = meta[meta.condition == "normal"].source_file.sort_values().iloc[7]
            anc = "SCRFD" if tuned.endswith("scrfd") else "BlazeFace"
            labels = {OURS: "ours s=3", tuned: f"ours s*={tuned.split('_')[1][1:]} ({anc})"}
            (args.out / "examples.json").write_text(json.dumps(
                {"preds": str(d), "backend": be, "qualitative": qual, "failures": fail, "labels": labels,
                 "conditions_source": src}, indent=1))

    tables = {"main_comparison": pd.DataFrame(M), "ablation": pd.DataFrame(A), "security": pd.DataFrame(S),
              "e2e": pd.DataFrame(E), "bias": pd.DataFrame(B)}
    cost = []
    for p in sorted(args.latency.rglob("latency_*.csv")):
        L = pd.read_csv(p)
        L.insert(0, "file", p.name)
        cost.append(L)
    if cost:
        tables["cost"] = pd.concat(cost, ignore_index=True)
    sizes = sorted(args.latency.rglob("model_sizes.csv"))
    if sizes:
        tables["model_sizes"] = pd.concat([pd.read_csv(p) for p in sizes]).drop_duplicates("model")
    for k, v in tables.items():
        v.to_csv(args.out / f"{k}.csv", index=False)

    extra = {"face_threshold": json.loads((args.face / "threshold.json").read_text()), "face_on_eval_set": faces,
             "error_taxonomy": tax, "scale_selection_on_val": selection}
    if args.train_metrics.exists():
        extra["yolo_test_metrics_fullframe"] = json.loads(args.train_metrics.read_text())
    (args.out / "summary.json").write_text(json.dumps(extra, indent=2))

    lines = ["# Results (generated by scripts/analyze.py)\n"]
    for k, v in tables.items():
        lines += [f"## {k}\n", md(v), ""]
    lines += ["## other\n", "```json", json.dumps(extra, indent=2), "```"]
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selection, indent=1))
    print(md(tables["main_comparison"][["backend", "variant", "condition", "n", "full_acc", "ours_acc", "delta_pp",
                                        "delta_lo_pp", "delta_hi_pp", "p_value", "fixed", "broken"]]))
    print(md(tables["ablation"][["backend", "config", "acc_all", "acc_normal", "acc_low_light", "acc_far",
                                 "acc_crowded", "background_hand_FAR"]]))


if __name__ == "__main__":
    main()

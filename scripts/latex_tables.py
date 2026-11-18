"""LaTeX tables + number macros for the report, generated from results/*.csv (no number is typed by hand).

    python scripts/latex_tables.py --results results --out report/generated
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BE = {"YOLOv8n (fine-tuned on HaGRID)": ("YOLOv8n (ours, fine-tuned)", "yolo"),
      "MediaPipe Gesture Recognizer (pretrained)": ("MediaPipe GR (pretrained)", "mp")}
CONDS = ["normal", "low_light", "far", "crowded", "all"]


def f1(v):
    return f"{100 * v:.1f}"


def sig(p):
    return "$^\\dagger$" if p < 0.05 else ""


def dlt(d, p):
    s = f"{d:+.1f}".replace("-", "$-$")
    return f"{{\\scriptsize({s}{sig(p)})}}"


def anchor_of(cfg):
    return "SCRFD" if str(cfg).endswith("scrfd") else "BlazeFace"


def scale_of(cfg):
    return cfg.split("_")[1][1:]


def main_table(M: pd.DataFrame) -> str:
    one = M.backend.nunique() == 1
    pre = "" if one else "& "
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{" + ("" if one else "l") + "lccccc}", r"\toprule",
         ("" if one else "Gesture detector & ") + r"Pipeline & Normal & Low light & Far (2$\times$) & Crowded & All \\",
         r"\midrule"]
    for be, (name, _) in BE.items():
        T = M[M.backend == be]
        if T.empty:
            continue
        variants = list(dict.fromkeys(T.variant))
        first = T[T.variant == variants[0]].set_index("condition")
        L.append(("" if one else f"\\multirow{{{1 + len(variants)}}}{{*}}{{{name}}} & ") + "Baseline (full frame) & "
                 + " & ".join(f1(first.loc[c, "full_acc"]) for c in CONDS) + r" \\")
        for v in variants:
            R = T[T.variant == v].set_index("condition")
            cfg = R.ours_config.iloc[0]
            lab = (f"Ours, $s{{=}}3$ (proposal)" if v.startswith("s3") else
                   f"Ours, $s^*{{=}}{scale_of(cfg)}$, {'SCRFD' if cfg.endswith('scrfd') else 'BlazeFace'} (val-sel.)")
            cells = [f"{f1(R.loc[c, 'ours_acc'])} {dlt(R.loc[c, 'delta_pp'], R.loc[c, 'p_value'])}" for c in CONDS]
            L.append(f"{pre}{lab} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\vspace{-1mm}",
          r"\caption{\textbf{Main comparison} on the 100 held-out HaGRID subjects. Top-1 gesture accuracy (\%), "
          r"with the paired change vs.\ the baseline in pp; $^\dagger$: exact McNemar $p<0.05$. Both pipelines use "
          r"the \emph{same} pretrained MediaPipe Gesture Recognizer, scored on its 5 classes (2,000 images, 500 per condition). "
          r"95\% bootstrap CIs of the overall change are given in the text.}",
          r"\label{tab:main}", r"\vspace{-2mm}", r"\end{table*}"]
    return "\n".join(L)


def ablation_table(A: pd.DataFrame, sel: dict) -> str:
    """Rows: ROI scale s; column groups: detector x anchor; cells: top-1 % on all / far images."""
    bes = [be for be in BE if be in set(A.backend)]
    sel_by_be = {be: sel.get("yolo" if BE[be][1] == "yolo" else "mediapipe", {}).get("selected") for be in bes}

    def cell(be, cfg, col):
        r = A[(A.backend == be) & (A.config == cfg)]
        if r.empty:
            return "--"
        v = f1(r.iloc[0][col])
        return f"\\textbf{{{v}}}" if sel_by_be[be] == cfg else v

    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{l" + "cc" * 2 * len(bes) + "}", r"\toprule"]
    L.append(" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{BE[be][0].split(' ')[0]}}}" for be in bes) + r" \\")
    L.append("".join(f"\\cmidrule(lr){{{2 + 4 * k}-{5 + 4 * k}}}" for k in range(len(bes))))
    L.append(" & " + " & ".join(r"\multicolumn{2}{c}{BlazeFace} & \multicolumn{2}{c}{SCRFD}" for _ in bes) + r" \\")
    L.append(" & ".join(["$s$"] + ["all", "far"] * 2 * len(bes)) + r" \\")
    L.append(r"\midrule")
    row = ["full frame"]
    for be in bes:
        row += [f"\\multicolumn{{4}}{{c}}{{{cell(be, 'full', 'acc_all')} / {cell(be, 'full', 'acc_far')}}}"]
    L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
    for sv in "23456":
        row = [sv + (" (prop.)" if sv == "3" else "")]
        for be in bes:
            for suf in ("", "_scrfd"):
                cfg = f"roi_s{sv}_up{suf}"
                row += [cell(be, cfg, "acc_all"), cell(be, cfg, "acc_far")]
        L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
    for cfg, lab in (("roi_s3_noup", "3, no up-scale"), ("roi_s3_up_y0.5", "3, shift $0.5h_f$")):
        row = [lab]
        for be in bes:
            row += [cell(be, cfg, "acc_all"), cell(be, cfg, "acc_far"), "", ""]
        L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
    row = ["face found"]
    for be in bes:
        for cfg in ("roi_s3_up", "roi_s3_up_scrfd"):
            r = A[(A.backend == be) & (A.config == cfg)]
            row += ([f1(r.iloc[0].face_found_normal), f1(r.iloc[0].face_found_far)] if len(r) else ["--", "--"])
    L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\vspace{-1mm}",
          r"\caption{\textbf{Ablation}: top-1 accuracy (\%) on all 4 conditions / far only, by ROI scale $s$ and "
          r"face anchor; every cell uses the same detector weights and images. Bold: the configuration selected "
          r"on the val split. ``face found'': \% of normal / far images in which the anchor found a face "
          r"(otherwise the pipeline falls back to the full frame).}",
          r"\label{tab:abl}", r"\vspace{-3mm}", r"\end{table}"]
    return "\n".join(L)


def cost_table(C: pd.DataFrame, sizes: pd.DataFrame | None) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Pipeline & median & p95 & extra \\", r"\midrule"]
    for key, name in (("yolo", "YOLOv8n"), ("mediapipe", "MediaPipe GR")):
        sub = C[C.gesture_backend == key]
        if not len(sub):
            continue
        for _, r in sub.iterrows():
            if str(r.config).startswith("arcface"):
                continue
            pipe = "full frame" if r.config == "full" else f"ROI $s{{=}}{scale_of(r.config)}$, {anchor_of(r.config)}"
            ex = "--" if r.config == "full" else f"{r.extra_vs_full_median_ms:+.1f}"
            L.append(f"{pipe} & {r.median_ms:.1f} & {r.p95_ms:.1f} & {ex} \\\\")
        L.append(r"\midrule")
    arc = C[C.config.astype(str).str.startswith("arcface")]
    if len(arc):
        r = arc.iloc[0]
        L.append(f"SCRFD + ArcFace (both) & {r.median_ms:.1f} & {r.p95_ms:.1f} & -- \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    sz = ""
    if sizes is not None:
        d = dict(zip(sizes.model, sizes.size_MB))
        yo = d.get("yolo_weights")
        sz = ((f" Model sizes: YOLOv8n {yo:.1f}\\,MB, " if yo is not None and yo == yo else " Model sizes: ")
              + f"MediaPipe GR "
              f"{d.get('mediapipe_gesture', float('nan')):.1f}\\,MB, BlazeFace {d.get('blazeface', float('nan')):.2f}\\,MB, "
              f"SCRFD {d.get('scrfd_det_500m', float('nan')):.1f}\\,MB, ArcFace {d.get('arcface_w600k_mbf', float('nan')):.1f}\\,MB.")
    dev = str(C.device.iloc[0]).replace("_", r"\_")
    L += [r"\vspace{-1mm}",
          f"\\caption{{\\textbf{{Cost}}: CPU latency per selfie in ms (100 native-resolution test images, "
          f"{dev.split('(')[0].strip()}, Ryzen 7 7800X3D, decode excluded). The ROI pipeline includes the anchor face detector and "
          f"the crop, and makes one face-detector call plus one gesture-detector call per image.{sz}}}",
          r"\label{tab:cost}", r"\vspace{-3mm}", r"\end{table}"]
    return "\n".join(L)


def security_table(E: pd.DataFrame, S: pd.DataFrame) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Pipeline & FRR & FAR$_\text{bg}$ & FAR$_\text{face}$ & FAR$_\text{gest}$ & FAR$_\text{rep}$ \\",
         r"\midrule"]
    for be, (name, _) in BE.items():
        sub = E[E.backend == be]
        for _, r in sub.iterrows():
            pipe = "full" if r.config == "full" else f"ROI $s{{=}}{scale_of(r.config)}$, {anchor_of(r.config)}"
            L.append(f"{pipe} & {f1(r.FRR)} & {f1(r.FAR_background_hand)} & "
                     f"{100 * r.FAR_impostor_face:.2f} & {100 * r.FAR_wrong_gesture:.2f} & "
                     f"{f1(r['FAR_replay_3_attempts'])} \\\\")
        if len(sub):
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\vspace{-1mm}",
          r"\caption{\textbf{End-to-end} accept/reject (\%; gesture \emph{and} ArcFace, $\tau{=}0.2235$). "
          r"FRR: genuine check-ins rejected (all conditions). FAR$_\text{bg}$: the challenge is the gesture of the "
          r"\emph{other} person in a crowded image. FAR$_\text{face}$: another enrolled student's identity is "
          r"claimed. FAR$_\text{gest}$: a wrong gesture is accepted. FAR$_\text{rep}$: an old photo passes a random "
          r"challenge within 3 attempts.}",
          r"\label{tab:sec}", r"\vspace{-3mm}", r"\end{table}"]
    return "\n".join(L)


def macros(M, A, S, E, C, B, summ) -> str:
    out = {}

    def put(k, v):
        out[re.sub(r"[^A-Za-z]", "", k)] = v

    for be, (_, key) in BE.items():
        T = M[M.backend == be]
        for v in dict.fromkeys(T.variant):
            vk = "S" if v.startswith("s3") else "T"
            for c in CONDS:
                r = T[(T.variant == v) & (T.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ck = {"normal": "Nor", "low_light": "Low", "far": "Far", "crowded": "Cro", "all": "All"}[c]
                put(f"{key}{vk}{ck}D", f"{r.delta_pp:+.1f}".replace("-", "$-$"))
                put(f"{key}{vk}{ck}Lo", f"{r.delta_lo_pp:+.1f}".replace("-", "$-$"))
                put(f"{key}{vk}{ck}Hi", f"{r.delta_hi_pp:+.1f}".replace("-", "$-$"))
                put(f"{key}{vk}{ck}P", f"{r.p_value:.2g}" if r.p_value >= 1e-3 else "<0.001")
                put(f"{key}{vk}{ck}Acc", f1(r.ours_acc))
                put(f"{key}Base{ck}Acc", f1(r.full_acc))
                put(f"{key}{vk}{ck}Fix", str(int(r.fixed)))
                put(f"{key}{vk}{ck}Brk", str(int(r.broken)))
            cfg = T[T.variant == v].ours_config.iloc[0]
            put(f"{key}{vk}Scale", scale_of(cfg))
            put(f"{key}{vk}Anchor", "SCRFD" if cfg.endswith("scrfd") else "BlazeFace")
        for _, r in A[A.backend == be].iterrows():
            ck = r.config.replace("roi_", "").replace("_up", "Up").replace("_noup", "Noup").replace("_scrfd", "Scrfd")
            ck = ck.replace("s", "S").replace(".", "")
            ck = "".join({"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "0": "Zero"}.get(ch, ch)
                         for ch in ck)
            if r.config != "full":
                put(f"{key}Face{ck}", f1(r.face_found_rate))
                for c in ("normal", "far", "crowded", "low_light"):
                    put(f"{key}Face{ck}{c}", f1(r[f"face_found_{c}"]))
            put(f"{key}Acc{ck}", f1(r.acc_all))
            put(f"{key}AccFar{ck}", f1(r.acc_far))
            put(f"{key}Bg{ck}", f1(r.background_hand_FAR))
            if "wrong_anchor_crowded" in r and r.wrong_anchor_crowded == r.wrong_anchor_crowded:
                put(f"{key}WrongAnchor{ck}", f1(r.wrong_anchor_crowded))
        for _, r in E[E.backend == be].iterrows():
            ck = "Full" if r.config == "full" else "Roi" + {"3": "Three", "4": "Four", "5": "Five", "6": "Six",
                                                              "2": "Two"}[scale_of(r.config)]
            put(f"{key}FRR{ck}", f1(r.FRR))
            put(f"{key}FARbg{ck}", f1(r.FAR_background_hand))
            for c in ("normal", "low_light", "far", "crowded"):
                if f"FRR_{c}" in r:
                    put(f"{key}FRR{ck}{c}", f1(r[f"FRR_{c}"]))
    if C is not None:
        for _, r in C.iterrows():
            if str(r.config).startswith("arcface"):
                put(f"latArc{r.gesture_backend}", f"{r.median_ms:.1f}")
                continue
            ck = "Full" if r.config == "full" else "Roi"
            put(f"lat{r.gesture_backend}{ck}", f"{r.median_ms:.1f}")
            put(f"latP{r.gesture_backend}{ck}", f"{r.p95_ms:.1f}")
            if r.config != "full":
                put(f"latExtra{r.gesture_backend}", f"{r.extra_vs_full_median_ms:+.1f}")
    ft = summ["face_threshold"]
    put("lfwThr", f"{ft['threshold']:.4f}")
    put("lfwTestAcc", f1(ft["lfw_test_acc"]))
    put("lfwTestFAR", f1(ft["lfw_test_FAR"]))
    put("lfwTestFRR", f1(ft["lfw_test_FRR"]))
    fe = summ["face_on_eval_set"][0]
    put("faceFRRnormal", f1(fe["FRR_normal"]))
    put("faceFARnormal", f"{100 * fe['FAR_normal']:.2f}")
    if "yolo_test_metrics_fullframe" in summ:
        y = summ["yolo_test_metrics_fullframe"]
        put("yoloMapFifty", f1(y["mAP50"]))
        put("yoloMap", f1(y["mAP50_95"]))
    for be, s in summ.get("scale_selection_on_val", {}).items():
        k = "yolo" if be == "yolo" else "mp"
        for cfg, a in s["val_accuracy"].items():
            put(f"{k}Val{cfg.replace('roi_s', 'S').replace('_up', '')}".replace("2", "Two").replace("3", "Three")
                .replace("4", "Four").replace("5", "Five").replace("6", "Six"), f1(a))
    for be, per_variant in summ.get("error_taxonomy", {}).items():
        k = "yolo" if be == "yolo" else "mp"
        for v, t in per_variant.items():
            vk = "S" if v.startswith("s3") else "T"
            for kk, val in t.items():
                if kk != "config":
                    put(f"{k}{vk}Err{kk.replace('_', ' ').title().replace(' ', '')}", str(val))
    if B is not None:
        grp = {"<25": "Young", "25-40": "Mid", ">40": "Old"}
        for _, r in B.iterrows():
            key = BE[r.backend][1]
            ck = "Full" if r.config == "full" else ("Three" if r.config == "roi_s3_up" else "T")
            g = grp.get(str(r.group), str(r.group))
            put(f"{key}Bias{ck}{r.attribute}{g}", f1(r.acc))
            put(f"{key}BiasN{r.attribute}{g}", str(int(r.n_subjects)))
    lines = ["% generated by scripts/latex_tables.py from results/*.csv - do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("report/generated"))
    args = ap.parse_args()
    R = args.results
    args.out.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(R / "main_comparison.csv")
    A = pd.read_csv(R / "ablation.csv")
    S = pd.read_csv(R / "security.csv")
    E = pd.read_csv(R / "e2e.csv")
    B = pd.read_csv(R / "bias.csv")
    C = pd.read_csv(R / "cost.csv") if (R / "cost.csv").exists() else None
    sizes = pd.read_csv(R / "model_sizes.csv") if (R / "model_sizes.csv").exists() else None
    summ = json.loads((R / "summary.json").read_text())
    (args.out / "tab_main.tex").write_text(main_table(M), encoding="utf-8")
    (args.out / "tab_ablation.tex").write_text(ablation_table(A, summ.get("scale_selection_on_val", {})),
                                               encoding="utf-8")
    if C is not None:
        (args.out / "tab_cost.tex").write_text(cost_table(C, sizes), encoding="utf-8")
    (args.out / "tab_security.tex").write_text(security_table(E, S), encoding="utf-8")
    (args.out / "numbers.tex").write_text(macros(M, A, S, E, C, B, summ), encoding="utf-8")
    print(f"tables -> {args.out}")


if __name__ == "__main__":
    main()

"""Face verification: pick the ArcFace threshold on LFW, then test it on the volunteer set.

1) LFW (no training, threshold only). Needs the LFW images (<lfw-root>/<Name>/<Name>_0001.jpg)
   and the official pair lists pairsDevTrain.txt (threshold) and optionally pairsDevTest.txt.
       python scripts/eval_face.py --lfw-root data/lfw --lfw-pairs data/lfw/pairsDevTrain.txt \
           --lfw-test-pairs data/lfw/pairsDevTest.txt --out runs/face

2) Volunteer set: genuine/impostor pairs from the embeddings that predict.py saved
   (enrollment image per subject vs. all other images):
       python scripts/eval_face.py --threshold runs/face/threshold.json \
           --classroom-emb runs/preds/classroom/embeddings.npz --out runs/face

Both can be given in one call. Writes threshold.json, face_report.md and ROC plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np

from gesture_attendance.config import load_config, resolve_path
from gesture_attendance.evaluation import best_accuracy_threshold, choose_enrollment, eer, load_threshold, roc_points
from gesture_attendance.vision.types import Box


def read_pairs(path: Path) -> list[tuple[str, int, str, int]]:
    pairs = []
    for line in path.read_text().splitlines()[1:]:
        t = line.split()
        if len(t) == 3:
            pairs.append((t[0], int(t[1]), t[0], int(t[2])))
        elif len(t) == 4:
            pairs.append((t[0], int(t[1]), t[2], int(t[3])))
    return pairs


def lfw_sims(pairs, lfw_root: Path, rec, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    def emb(name, idx):
        key = (name, idx)
        if key not in cache:
            p = lfw_root / name / f"{name}_{idx:04d}.jpg"
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(p)
            h, w = img.shape[:2]
            e, _ = rec.embed_main(img, hint=Box(w * 0.3, h * 0.3, w * 0.7, h * 0.7))  # LFW faces are centred
            cache[key] = e
        return cache[key]

    sims, same = [], []
    for k, (n1, i1, n2, i2) in enumerate(pairs):
        e1, e2 = emb(n1, i1), emb(n2, i2)
        sims.append(float(np.dot(e1, e2)) if e1 is not None and e2 is not None else -1.0)  # no face -> reject
        same.append(n1 == n2)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")
    return np.array(sims), np.array(same)


def plot_roc(curves: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for label, (g, i) in curves.items():
        _, far, frr = roc_points(g, i)
        ax.plot(np.clip(far, 1e-4, 1), 1 - frr, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("True accept rate (1 - FRR)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfw-root", type=Path)
    ap.add_argument("--lfw-pairs", type=Path, help="pairsDevTrain.txt (threshold selection)")
    ap.add_argument("--lfw-test-pairs", type=Path, help="pairsDevTest.txt (report only)")
    ap.add_argument("--threshold", default=None, help="value or threshold.json (if LFW is not given)")
    ap.add_argument("--classroom-emb", type=Path, help="embeddings.npz from predict.py --dataset classroom")
    ap.add_argument("--out", type=Path, default=Path("runs/face"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    md = ["# Face verification (ArcFace MobileFaceNet, cosine similarity)\n"]
    curves = {}
    thr = None

    if args.lfw_root and args.lfw_pairs:
        from gesture_attendance.vision.face_recognizer import ArcFaceRecognizer
        cfg = load_config(args.config)
        rec = ArcFaceRecognizer(resolve_path(cfg["models"].get("arcface_dir")))
        cache = {}
        sims, same = lfw_sims(read_pairs(args.lfw_pairs), args.lfw_root, rec, cache)
        thr, acc = best_accuracy_threshold(sims, same)
        e, _ = eer(sims[same], sims[~same])
        res = {"threshold": thr, "criterion": "max accuracy on LFW DevTrain", "lfw_train_acc": acc,
               "lfw_train_eer": e, "n_pairs": int(len(sims))}
        curves["LFW DevTrain"] = (sims[same], sims[~same])
        if args.lfw_test_pairs:
            ts, tsame = lfw_sims(read_pairs(args.lfw_test_pairs), args.lfw_root, rec, cache)
            res["lfw_test_acc"] = float(((ts >= thr) == tsame).mean())
            res["lfw_test_FAR"] = float((ts[~tsame] >= thr).mean())
            res["lfw_test_FRR"] = float((ts[tsame] < thr).mean())
            curves["LFW DevTest"] = (ts[tsame], ts[~tsame])
        (args.out / "threshold.json").write_text(json.dumps(res, indent=2))
        md += ["## LFW threshold selection\n", "```json", json.dumps(res, indent=2), "```", "",
               f"Put `face_threshold: {thr:.4f}` in configs/default.yaml.\n"]
        print(json.dumps(res, indent=2))
    elif args.threshold is not None:
        thr = load_threshold(args.threshold)

    if args.classroom_emb:
        if thr is None:
            raise SystemExit("need --threshold (or LFW) to evaluate the classroom set")
        z = np.load(args.classroom_emb, allow_pickle=True)
        files, subj, cond = list(z["files"]), list(z["subjects"]), list(z["conditions"])
        emb, has = z["emb"], z["has_face"].astype(bool)
        enroll = choose_enrollment(files, subj, cond, has)
        gen, imp, gen_cond = [], [], []
        for i in range(len(files)):
            if i in enroll.values():
                continue
            for s, j in enroll.items():
                sim = float(np.dot(emb[i], emb[j])) if has[i] else -1.0
                if s == subj[i]:
                    gen.append(sim)
                    gen_cond.append(cond[i])
                else:
                    imp.append(sim)
        gen, imp = np.array(gen), np.array(imp)
        e, e_thr = eer(gen, imp)
        res = {"threshold": thr, "n_subjects": len(enroll), "n_genuine": len(gen), "n_impostor": len(imp),
               "FRR": float((gen < thr).mean()), "FAR": float((imp >= thr).mean()), "EER": e, "EER_threshold": e_thr,
               "FRR_by_condition": {c: float((gen[np.array(gen_cond) == c] < thr).mean())
                                    for c in sorted(set(gen_cond))},
               "no_face_rate": float((~has).mean())}
        curves["Volunteers"] = (gen, imp)
        (args.out / "classroom_face.json").write_text(json.dumps(res, indent=2))
        md += ["## Volunteer set (genuine vs impostor)\n", "```json", json.dumps(res, indent=2), "```", ""]
        print(json.dumps(res, indent=2))

    if curves:
        plot_roc(curves, args.out / "roc.png")
    (args.out / "face_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

"""End-to-end accept/reject rates on the volunteer set (gesture challenge AND face match).

    python scripts/eval_e2e.py --preds runs/preds/classroom --threshold runs/face/threshold.json

For each pipeline config (same cached predictions as evaluate.py):
  genuine          right student, challenge = the gesture they show       -> reject = false reject (FRR)
  impostor_face    someone else's face showing the challenged gesture     -> accept = false accept
  wrong_gesture    right face, every challenge != shown gesture           -> accept = false accept
  background_hand  challenge = gesture shown by a person in the background -> accept = false accept
  replay           an old photo of the student vs a fresh random challenge -> accept per attempt,
                   and within --max-attempts attempts: 1 - (1 - p)^k
One image per subject (first 'normal' one) is the enrollment image and is not used as a probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd

from gesture_attendance.evaluation import choose_enrollment, load_threshold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--threshold", required=True, help="cosine threshold value or threshold.json")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path("runs/eval") / args.preds.name
    out.mkdir(parents=True, exist_ok=True)

    thr = load_threshold(args.threshold)
    info = json.loads((args.preds / "run_info.json").read_text())
    classes = info["classes"]
    K = len(classes)
    z = np.load(args.preds / "embeddings.npz", allow_pickle=True)
    files = list(z["files"])
    emb, has = z["emb"], z["has_face"].astype(bool)
    subj, cond = list(z["subjects"]), list(z["conditions"])
    enroll = choose_enrollment(files, subj, cond, has)
    enroll_idx = set(enroll.values())
    file_ix = {f: i for i, f in enumerate(files)}

    rows = []
    for name in info["configs"]:
        df = pd.read_csv(args.preds / f"{name}.csv", keep_default_na=False)
        cnt = {k: [0, 0] for k in ("genuine", "impostor_face", "wrong_gesture", "background_hand", "replay")}
        genuine_by_cond = {}
        for _, r in df.iterrows():
            i = file_ix[r.file]
            if i in enroll_idx or r.subject not in enroll:
                continue
            own_sim = float(np.dot(emb[i], emb[enroll[r.subject]])) if has[i] else -1.0
            face_ok = own_sim >= thr
            shows = r.gesture in classes
            if shows:  # genuine attempt
                ok = (r.pred == r.gesture) and face_ok
                cnt["genuine"][0] += int(not ok)
                cnt["genuine"][1] += 1
                gc = genuine_by_cond.setdefault(r.condition, [0, 0])
                gc[0] += int(not ok)
                gc[1] += 1
                for s, j in enroll.items():  # claim another student's identity
                    if s == r.subject:
                        continue
                    sim = float(np.dot(emb[i], emb[j])) if has[i] else -1.0
                    cnt["impostor_face"][0] += int((r.pred == r.gesture) and sim >= thr)
                    cnt["impostor_face"][1] += 1
                for c in classes:
                    if c != r.gesture:
                        cnt["wrong_gesture"][0] += int((r.pred == c) and face_ok)
                        cnt["wrong_gesture"][1] += 1
            if r.background_gesture and r.background_gesture != r.gesture:
                cnt["background_hand"][0] += int((r.pred == r.background_gesture) and face_ok)
                cnt["background_hand"][1] += 1
            # replay: probability that a fixed old photo passes a uniformly random challenge
            cnt["replay"][0] += (1.0 / K) * int((r.pred in classes) and face_ok)
            cnt["replay"][1] += 1

        row = {"config": name, "threshold": thr}
        for k, (bad, n) in cnt.items():
            key = "FRR" if k == "genuine" else f"FAR_{k}"
            row[key] = bad / n if n else float("nan")
            row[f"n_{k}"] = n
        p = row["FAR_replay"]
        row[f"FAR_replay_within_{args.max_attempts}_attempts"] = 1 - (1 - p) ** args.max_attempts
        for c, (bad, n) in genuine_by_cond.items():
            row[f"FRR[{c}]"] = bad / n
        rows.append(row)

    R = pd.DataFrame(rows)
    R.to_csv(out / "e2e.csv", index=False)
    try:
        md = R.set_index("config").T.reset_index().rename(columns={"index": "metric"}).to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        md = R.T.to_string()
    (out / "e2e.md").write_text(f"# End-to-end FRR / FAR (threshold {thr:.4f})\n\n{md}\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

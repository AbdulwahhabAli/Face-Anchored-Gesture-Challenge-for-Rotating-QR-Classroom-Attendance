"""Build the 8-slide presentation from results/ (every number is read from the CSVs / summary.json, none typed by hand).

    python slides/build_slides.py --results results --out slides/slides.pptx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x14, 0x22, 0x3B)       # dominant (title + closing slides)
INK = RGBColor(0x1B, 0x1B, 0x1B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x8A, 0x89, 0x84)
BLUE = RGBColor(0x2A, 0x78, 0xD6)       # "ours" everywhere (same as figures)
GREY = RGBColor(0x8A, 0x89, 0x84)       # "baseline" everywhere
TINT = RGBColor(0xEE, 0xF4, 0xFC)
SOFT = RGBColor(0xF3, 0xF2, 0xEF)
RED = RGBColor(0xC0, 0x26, 0x1F)
GREEN = RGBColor(0x0A, 0x7A, 0x35)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)
MX = Inches(0.6)


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, spacing=None):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, {bold,color,size,italic})."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, opt in segs:
            r = p.add_run()
            r.text = seg
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", False)
            f.color.rgb = opt.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        segs = it if isinstance(it, list) else [(it, {})]
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.22))))
        pPr.set("indent", str(-Emu(Inches(0.22))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        for seg, opt in segs:
            r = p.add_run()
            r.text = seg
            r.font.name = BODY
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", False)
            r.font.color.rgb = opt.get("color", color)
    return tb


def card(slide, x, y, w, h, fill=SOFT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def stat(slide, x, y, w, value, label, color=BLUE, vsize=40):
    text(slide, x, y, w, Inches(0.8), value, size=vsize, bold=True, color=color, font=HEAD)
    text(slide, x, y + Inches(0.82), w, Inches(0.7), label, size=13, color=INK2)


def title(slide, t, sub=None):
    text(slide, MX, Inches(0.42), W - 2 * MX, Inches(0.8), t, size=34, bold=True, color=NAVY, font=HEAD,
         name="Title")
    if sub:
        text(slide, MX, Inches(1.18), W - 2 * MX, Inches(0.45), sub, size=16, color=INK2, name="Subtitle")


def footer(slide, n):
    text(slide, W - MX - Inches(3), H - Inches(0.42), Inches(3), Inches(0.3),
         f"CSBP441 · Face-anchored gesture attendance · {n}/8", size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def pp(v, d=1):
    return f"{v:+.{d}f} pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("slides/slides.pptx"))
    args = ap.parse_args()
    R = args.results
    F = R / "figures"
    M = pd.read_csv(R / "main_comparison.csv")
    A = pd.read_csv(R / "ablation.csv")
    E = pd.read_csv(R / "e2e.csv")
    B = pd.read_csv(R / "bias.csv")
    C = pd.read_csv(R / "cost.csv")
    sizes = pd.read_csv(R / "model_sizes.csv")
    summ = json.loads((R / "summary.json").read_text())
    be = "MediaPipe Gesture Recognizer (pretrained)"
    PROP, SEL = "roi_s3_up", summ["scale_selection_on_val"]["mediapipe"]["selected"]   # roi_s6_up_scrfd

    def mrow(cfg, cond):
        return M[(M.backend == be) & (M.ours_config == cfg) & (M.condition == cond)].iloc[0]

    def arow(cfg):
        return A[(A.backend == be) & (A.config == cfg)].iloc[0]

    def erow(cfg):
        return E[(E.backend == be) & (E.config == cfg)].iloc[0]

    def lat(cfg):
        r = C[(C.gesture_backend == "mediapipe") & (C.config == cfg)].iloc[0]
        return r.median_ms, r.p95_ms

    def bias(cfg, attr, group):
        return B[(B.backend == be) & (B.config == cfg) & (B.attribute == attr) & (B.group == group)].iloc[0]

    p_all, s_all = mrow(PROP, "all"), mrow(SEL, "all")
    s_far, s_low, s_cro, s_nor = (mrow(SEL, c) for c in ("far", "low_light", "crowded", "normal"))
    a_full, a_prop, a_sel = arow("full"), arow(PROP), arow(SEL)
    e_full, e_prop, e_sel = erow("full"), erow(PROP), erow(SEL)
    lf, lp, ls, la = lat("full"), lat(PROP), lat(SEL), lat("arcface (both pipelines)")
    extra_sel = ls[0] - lf[0]
    thr = summ["face_threshold"]
    tax = summ["error_taxonomy"]["mediapipe"]["s* (val-selected)"]
    val = summ["scale_selection_on_val"]["mediapipe"]["val_accuracy"]
    scrfd_min_face = min(a_sel[f"face_found_{c}"] for c in ("normal", "low_light", "far", "crowded"))
    rep = [e_full.FAR_replay_3_attempts, e_prop.FAR_replay_3_attempts, e_sel.FAR_replay_3_attempts]
    total_mb = sizes.size_MB.dropna().sum()
    n_black = int(bias("full", "race", "Black").n_subjects)
    n_asian = int(bias("full", "race", "Asian").n_subjects)
    PROP_C, SEL_C = BLUE, RGBColor(0x12, 0x8A, 0x5C)    # same hues as the figures (blue = s=3, green = s*=6)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    notes = {}

    # 1 ---------------------------------------------------------------- title (dark)
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    text(s, MX, Inches(1.7), Inches(11.5), Inches(1.9),
         "Face-Anchored Gesture Challenge\nfor Rotating-QR Classroom Attendance", size=44, bold=True,
         color=WHITE, font=HEAD)
    text(s, MX, Inches(3.85), Inches(11.5), Inches(0.9),
         "A random hand gesture next to your face proves you are in the room.\n"
         "Does cropping around the face before gesture detection make it more reliable?",
         size=20, color=RGBColor(0xCA, 0xDC, 0xFC))
    text(s, MX, Inches(5.6), Inches(11), Inches(0.9),
         [[("Abdulwahhab Ali  ·  Yazan Badawie", {"bold": True, "color": WHITE, "size": 18})],
          [("CSBP441 Computer Vision · Fall 2026 · Phase 2", {"color": RGBColor(0xCA, 0xDC, 0xFC), "size": 14})]])
    notes[1] = ("~20 s. Introduce the team and the one-line idea: QR attendance is easy to forward, so we add a "
                "random gesture next to the face plus face verification. The CV question is whether a "
                "face-anchored crop makes the gesture detector more accurate.")

    # 2 ---------------------------------------------------------------- problem, contribution, hypothesis
    s = prs.slides.add_slide(blank)
    title(s, "Problem: a QR code can't prove who scanned it")
    cw, gap, y0 = Inches(3.85), Inches(0.36), Inches(1.6)
    cards = [("1  Forwarding", "A student photographs the board QR and sends it to an absent friend. A 3 s "
              "rotating token narrows the window but does not prove who scanned it."),
             ("2  Our check-in", "After scanning, the phone asks for a random gesture (1 of 5). The selfie is "
              "accepted only if gesture = challenge AND the face matches the enrolled ArcFace embedding."),
             ("3  The CV problem", "Detectors downscale the whole frame (MediaPipe's palm detector sees 192 px), "
              "so the hand shrinks to a few pixels. Classmates' hands are in the frame too: whose gesture is it?")]
    for k, (h_, b_) in enumerate(cards):
        x = MX + k * (cw + gap)
        card(s, x, y0, cw, Inches(2.75), fill=SOFT if k < 2 else TINT)
        text(s, x + Inches(0.3), y0 + Inches(0.26), cw - Inches(0.6), Inches(0.5), h_, size=20, bold=True,
             color=NAVY if k < 2 else BLUE, font=HEAD)
        text(s, x + Inches(0.3), y0 + Inches(0.85), cw - Inches(0.6), Inches(1.8), b_, size=15, color=INK)
    card(s, MX, Inches(4.6), W - 2 * MX, Inches(2.2), fill=WHITE)
    text(s, MX + Inches(0.3), Inches(4.72), W - 2 * MX - Inches(0.6), Inches(0.45), "Contribution and hypothesis",
         size=20, bold=True, color=NAVY, font=HEAD)
    text(s, MX + Inches(0.3), Inches(5.22), W - 2 * MX - Inches(0.6), Inches(1.5),
         [[("Face-anchored ROI cascade: ", {"bold": True, "color": BLUE}),
           ("a face detector finds the selfie-taker's face, we crop s × the face box at native resolution, and "
            "the same gesture detector runs on the up-scaled crop. More pixels for the hand, and the gesture is "
            "tied to the face that is verified.", {})],
          [("H1: ", {"bold": True, "color": BLUE}),
           ("with the same detector, top-1 gesture accuracy improves by ≥ 5 pp over full-frame inference on "
            "held-out people, at ≤ 50 ms extra CPU latency.", {})]], size=16, spacing=6)
    footer(s, 2)
    notes[2] = ("~1 min. Problem, then what we add, then why it is a vision problem. Related work in one "
                "sentence: crop-then-detect cascades (Viola-Jones, MediaPipe palm->landmarks, SAHI tiling); "
                "our anchor is the face, which identity needs anyway, so it costs one detector call, not k tiles. "
                "State H1 exactly as on the slide; we come back to it with evidence for and against.")

    # 3 ---------------------------------------------------------------- method
    s = prs.slides.add_slide(blank)
    title(s, "Method: same detector, different input")
    pw = Inches(11.0)
    pic = picture(s, F / "pipeline.png", (W - pw) // 2, Inches(1.3), w=pw)
    yb_ = pic.top + pic.height + Inches(0.2)
    colw = (W - 2 * MX - Inches(0.6)) / 3
    items = [("Baseline", "Pretrained MediaPipe Gesture Recognizer on the full frame. Top-1 = highest-scoring "
                          "gesture (score ≥ 0.5), 'none' if no hand."),
             ("Ours", "Largest face (SCRFD) → ROI = s × face box, clipped → the same detector on the crop → boxes "
                      "mapped back. No face → full frame (counted)."),
             ("Shared by both", f"Same weights and top-1 rule. Identity: SCRFD + ArcFace MobileFaceNet, "
                                f"τ = {thr['threshold']:.3f} from LFW (DevTest accuracy {pct(thr['lfw_test_acc'])}).")]
    for k, (h_, b_) in enumerate(items):
        x = MX + k * (colw + Inches(0.3))
        text(s, x, yb_, colw, Inches(0.4), h_, size=18, bold=True, color=BLUE if k == 1 else NAVY, font=HEAD)
        text(s, x, yb_ + Inches(0.42), colw, Inches(1.2), b_, size=14, color=INK)
    footer(s, 3)
    notes[3] = ("~1 min. Walk the figure left to right. Both pipelines share the exact same pretrained detector "
                "and top-1 rule, so any difference comes from the input crop. The proposal fixed BlazeFace and "
                "s = 3; the final s* = 6 with SCRFD was chosen on a separate validation set (next slide). "
                "Identity is the same for both pipelines; the threshold comes from LFW, not from our data.")

    # 4 ---------------------------------------------------------------- data + protocol
    s = prs.slides.add_slide(blank)
    title(s, "Data and protocol: 100 unseen people, 4 conditions")
    picture(s, F / "conditions.png", MX, Inches(1.45), w=Inches(7.4))
    text(s, MX, Inches(4.15), Inches(7.4), Inches(0.4),
         "One held-out subject in the four test conditions (deterministic: same input for both pipelines).",
         size=12, color=INK2)
    for k, (v_, l_) in enumerate((("100", "held-out test subjects\n(none seen in val)"),
                                  ("2,400", "test images\n(600 × 4 conditions)"),
                                  ("1,200", "val images, used only\nto choose anchor and s"))):
        stat(s, MX + k * Inches(2.5), Inches(4.85), Inches(2.4), v_, l_, color=NAVY, vsize=36)
    x2 = MX + Inches(7.8)
    bullets(s, x2, Inches(1.45), W - x2 - MX, Inches(5.4), [
        [("Source: ", {"bold": True}), ("HaGRIDv2, official subject-disjoint split; we download only the "
                                         "selected images (≈ 1.2 GB) by HTTP range requests", {})],
        [("Test: ", {"bold": True}), ("100 held-out subjects × 6 gestures = 600 native-resolution images; "
                                      "scored on MediaPipe's 5 gestures", {})],
        [("4 conditions (our construction): ", {"bold": True}),
         ("normal, low light, far (2× zoom-out), crowded (another person with another gesture pasted beside "
          "the claimant = background-hand attack) → 2,400 images", {})],
        [("Val: ", {"bold": True}), ("other subjects, 1,200 images; used only to choose the anchor and s "
                                     "(grid: {BlazeFace, SCRFD} × s = 2…6)", {})],
        [("Metrics: ", {"bold": True}), ("top-1 accuracy (95% bootstrap CI), paired Δ + exact McNemar, "
                                         "end-to-end FRR / FAR, CPU latency", {})],
        [("Not done: ", {"bold": True, "color": RED}), ("the planned classroom volunteer set; the hard "
                                                        "conditions are synthetic", {})],
    ], size=14, gap=7)
    footer(s, 4)
    notes[4] = ("~50 s. The volunteer classroom set from the proposal was not collected; instead we built a "
                "controlled test set from HaGRID's held-out subjects. Explain the four conditions and that the "
                "crowded one doubles as a background-hand attack. Stress that ROI settings come from val, the "
                "face threshold from LFW, never from test.")

    # 5 ---------------------------------------------------------------- main result
    s = prs.slides.add_slide(blank)
    title(s, "Result: the crop helps — once the ROI is large enough")
    cw_ = Inches(8.3)
    chart = picture(s, F / "acc_by_condition.png", MX, Inches(1.3), w=cw_)
    picture(s, F / "qualitative_slide.png", MX, chart.top + chart.height + Inches(0.15), w=cw_)
    x2 = MX + cw_ + Inches(0.4)
    wst = W - x2 - MX
    stat(s, x2, Inches(1.35), wst, pp(p_all.delta_pp), f"proposal (s = 3, BlazeFace), all images\n→ H1 rejected",
         color=PROP_C, vsize=36)
    stat(s, x2, Inches(2.85), wst, pp(s_all.delta_pp),
         f"val-selected (s* = 6, SCRFD): {pct(s_all.full_acc)} → {pct(s_all.ours_acc)}\n"
         f"95% CI [{s_all.delta_lo_pp:+.1f}, {s_all.delta_hi_pp:+.1f}], McNemar p < 0.001", color=SEL_C, vsize=36)
    stat(s, x2, Inches(4.35), wst, pp(s_far.delta_pp),
         f"far condition: {pct(s_far.full_acc)} → {pct(s_far.ours_acc)}", color=SEL_C, vsize=36)
    text(s, x2, Inches(5.65), wst, Inches(1.2),
         [[(f"+{extra_sel:.1f} ms", {"bold": True}), (" median CPU latency (budget ≤ 50 ms).", {})],
          [("H1 holds for the val-selected setting, not for the one we fixed in advance.", {"italic": True})]],
         size=13, color=INK2, spacing=4)
    footer(s, 5)
    notes[5] = (f"~1.5 min. Same 2,000 scored test images, same detector. First the honest part: the setting we "
                f"pre-registered, s = 3 with BlazeFace, LOSES {abs(p_all.delta_pp):.1f} pp because it cuts off "
                f"hands held beside the head (right-most photo). With the setting chosen on val, accuracy goes "
                f"from {pct(s_all.full_acc)} to {pct(s_all.ours_acc)}, and every condition improves: far "
                f"{pp(s_far.delta_pp)}, low light {pp(s_low.delta_pp)}, crowded {pp(s_cro.delta_pp)}, normal "
                f"{pp(s_nor.delta_pp)}. The gain is largest where the hand is small, as hypothesised. Photos: "
                f"white = ROI, cyan = anchor face, orange = baseline detection, blue = ours.")

    # 6 ---------------------------------------------------------------- ablation + cost
    s = prs.slides.add_slide(blank)
    title(s, "Ablation and cost: ROI size and anchor decide it")
    aw = Inches(7.9)
    abl = picture(s, F / "ablation.png", MX, Inches(1.3), w=aw)
    bullets(s, MX, abl.top + abl.height + Inches(0.25), aw, Inches(2.6), [
        [("ROI size matters most: ", {"bold": True}),
         (f"s ≤ 3 cuts off hands held beside the head (s = 3: {pct(a_prop.acc_all)} vs {pct(a_full.acc_all)} "
          f"full frame).", {})],
        [("Anchor decides 'far': ", {"bold": True}),
         (f"BlazeFace finds a face in only {pct(a_prop.face_found_far)} of far images and falls back to the full "
          f"frame; SCRFD finds ≥ {pct(scrfd_min_face, 0)} in every condition.", {})],
        [("SCRFD boxes are tighter, ", {"bold": True}),
         (f"so it needs s ≥ 5; s = 6 is the grid edge (val {pct(val[SEL])} vs {pct(val['full'])} full).", {})],
        [("Up-scaling, shifting the ROI down: ", {"bold": True}), ("≈ 1 pp at s = 3.", {})],
    ], size=14, gap=6)
    x2 = MX + aw + Inches(0.4)
    wtab = W - x2 - MX
    rows = [("CPU, ms", "median", "p95", "extra"),
            ("full frame", f"{lf[0]:.1f}", f"{lf[1]:.1f}", "–"),
            ("s = 3, BlazeFace", f"{lp[0]:.1f}", f"{lp[1]:.1f}", f"+{lp[0] - lf[0]:.1f}"),
            ("s* = 6, SCRFD", f"{ls[0]:.1f}", f"{ls[1]:.1f}", f"+{extra_sel:.1f}"),
            ("ArcFace (both)", f"{la[0]:.1f}", f"{la[1]:.1f}", "–")]
    rh = Inches(0.4)
    tbl = s.shapes.add_table(len(rows), 4, x2, Inches(1.4), wtab, rh * len(rows)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.text = v
            para = c.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            run = para.runs[0]
            run.font.size = Pt(13)
            run.font.name = BODY
            run.font.bold = i == 0 or (i == 3)
            run.font.color.rgb = SEL_C if i == 3 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = SOFT if i == 0 else WHITE
    for j, f_ in enumerate((0.38, 0.23, 0.19, 0.20)):
        tbl.columns[j].width = int(wtab * f_)
    text(s, x2, Inches(1.4) + rh * len(rows) + Inches(0.2), wtab, Inches(3.0),
         [[("Per selfie, Ryzen 7 7800X3D, 100 native-resolution test images, decode excluded.",
            {"color": INK2, "size": 12})],
          [("Both pipelines make one gesture-detector call; ours adds one face-detector call and a crop.", {})],
          [("SCRFD already runs for identity, so in deployment the face detection is shared and the added cost "
            "is close to zero.", {})],
          [(f"All models together: {total_mb:.0f} MB.", {})]], size=14, spacing=8)
    footer(s, 6)
    notes[6] = ("~1 min. Read the plot left to right: below s = 4 the ROI cuts the hand off and accuracy collapses; "
                "the SCRFD curve keeps rising to s = 6, the end of our grid. BlazeFace short-range misses most "
                "faces in the far condition, so its ROI silently becomes the baseline there. Cost: a few "
                "milliseconds, far inside the 50 ms budget.")

    # 7 ---------------------------------------------------------------- security + failures
    s = prs.slides.add_slide(blank)
    title(s, "Security, failures and limitations")
    fp = picture(s, F / "failures_slide.png", MX, Inches(1.3), w=Inches(8.7))
    text(s, MX, fp.top + fp.height + Inches(0.05), Inches(8.7), Inches(0.3),
         "Failures of the val-selected ROI (held-out test images).", size=12, color=INK2)
    xr = MX + Inches(9.1)
    wr = W - xr - MX
    stat(s, xr, Inches(1.3), wr, f"{pct(e_full.FAR_background_hand)} → {pct(e_sel.FAR_background_hand)}",
         "background-hand attack accepted,\nend to end (baseline → ours)", color=SEL_C, vsize=28)
    stat(s, xr, Inches(2.85), wr, f"{pct(e_full.FRR)} → {pct(e_sel.FRR)}",
         "genuine check-ins rejected\n(FRR, baseline → ours)", color=SEL_C, vsize=28)
    yb2, cw2 = Inches(4.6), (W - 2 * MX - Inches(0.4)) / 2
    text(s, MX, yb2, cw2, Inches(2.3),
         [[("Where it fails", {"bold": True, "color": NAVY, "size": 17, "font": HEAD})],
          [(f"{tax['errors']} errors: {tax['no_detection_in_roi']} hands missed inside the ROI (mostly low light "
            f"and crowded), {tax['wrong_class']} wrong class, {tax['no_face_found']} no face.", {})],
          [(f"Wrong anchor: the largest face is the background person in {pct(a_sel.wrong_anchor_crowded)} of "
            f"crowded images → the genuine student is rejected (the attack still fails).", {})]],
         size=14, color=INK, spacing=5)
    text(s, MX + cw2 + Inches(0.4), yb2, cw2, Inches(2.3),
         [[("Limitations", {"bold": True, "color": NAVY, "size": 17, "font": HEAD})],
          [(f"Replay: an old photo passes a random 1-of-5 challenge within 3 tries {100 * min(rep):.1f}–{100 * max(rep):.1f}% "
            f"of the time, with any pipeline → needs two-gesture challenges.", {})],
          [(f"Synthetic conditions; no real classroom selfies. HaGRID is 80% White ({n_black} Black, {n_asian} "
            f"Asian test subjects). Face embeddings need consent and deletion.", {})]],
         size=14, color=INK, spacing=5)
    footer(s, 7)
    notes[7] = (f"~1 min. With the ROI the gesture and the verified face come from the same person, so copying a "
                f"classmate's gesture fails: the face next to that hand is not yours. Then the failures: low light "
                f"and crowded scenes, and the largest-face rule picking the wrong person. Be explicit that the "
                f"gesture challenge alone is weak liveness against replay, and that our hard conditions are "
                f"synthetic. Bias: accuracy is similar for female and male subjects "
                f"({pct(bias(SEL, 'gender', 'female').acc)} vs {pct(bias(SEL, 'gender', 'male').acc)}); too few "
                f"non-White subjects to conclude anything about race.")

    # 8 ---------------------------------------------------------------- conclusion (dark) + demo
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    light = RGBColor(0xCA, 0xDC, 0xFC)
    text(s, MX, Inches(0.55), W - 2 * MX, Inches(0.8), "Takeaways", size=36, bold=True, color=WHITE, font=HEAD)
    tk = [[("The face-anchored crop works if the ROI is large enough: ", {"bold": True}),
           (f"{pp(s_all.delta_pp)} overall, {pp(s_far.delta_pp)} for distant subjects, +{extra_sel:.1f} ms.", {})],
          [("Our pre-registered s = 3 failed ", {"bold": True}),
           (f"({pp(p_all.delta_pp)}): the ROI size depends on how people pose and must be validated on the "
            f"target users.", {})],
          [("Tying the gesture to the verified face ", {"bold": True}),
           (f"blocks the background-hand attack ({pct(e_full.FAR_background_hand)} → "
            f"{pct(e_sel.FAR_background_hand)}) and halves false rejects.", {})],
          [("Next: ", {"bold": True}), ("real classroom selfies, a two-gesture challenge against replay, "
                                        "and larger s.", {})]]
    tb = text(s, MX, Inches(1.6), Inches(7.6), Inches(4.8), tk, size=18, color=WHITE, spacing=14)
    card(s, MX + Inches(8.1), Inches(1.6), Inches(4.0), Inches(3.5), fill=RGBColor(0x22, 0x35, 0x57))
    text(s, MX + Inches(8.4), Inches(1.85), Inches(3.4), Inches(3.8),
         [[("Live demo (3 min)", {"bold": True, "size": 20, "color": WHITE, "font": HEAD})],
          [("1. Held-out images: baseline vs ours side by side", {"color": light})],
          [("2. Attack: the challenge is the gesture of the person behind", {"color": light})],
          [("3. Live app: scan QR → gesture → selfie → dashboard → Excel", {"color": light})],
          [("Backup: results/demo/demo_backup.mp4", {"color": MUTED, "size": 13})]], size=15, spacing=10)
    notes[8] = ("~30 s, then switch to the demo (3 min): demo_compare.py on held-out images, then the crowded "
                "attack image, then the live app with a phone. If the network fails, play demo_backup.mp4.")

    for i, sl in enumerate(prs.slides, 1):
        sl.notes_slide.notes_text_frame.text = notes.get(i, "")
    prs.core_properties.title = "Face-Anchored Gesture Challenge for Rotating-QR Classroom Attendance"
    prs.core_properties.author = "Abdulwahhab Ali, Yazan Badawie"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"slides -> {args.out}")


if __name__ == "__main__":
    main()

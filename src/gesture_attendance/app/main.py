"""FastAPI app: rotating QR on the board -> student phone -> random gesture challenge -> selfie -> accept/reject.

Pages
    /instructor   dashboard: roster upload, start/end session, enrollment, live count, flags, Excel report
    /board        full-screen rotating QR + live count (project this)
    /q/<token>    target of the QR; issues a short-lived ticket cookie and redirects to /checkin
    /checkin      student page: enter ID -> gesture challenge -> camera selfie
    /collect      volunteer test-set collection (consented), saves images + labels.csv
Instructor pages/APIs need the instructor key (?key=... once; it is then kept in a cookie).
"""
from __future__ import annotations

import csv
import io
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import load_config, resolve_path
from ..gestures import gesture_text
from ..vision.face_recognizer import cosine
from .report import course_report, session_report
from .roster import parse_roster
from .store import Store
from .tokens import RotatingToken, Signer

HERE = Path(__file__).parent
TICKET_COOKIE, DEVICE_COOKIE, INSTR_COOKIE = "ga_ticket", "ga_dev", "ga_instr"

MESSAGES = {
    "ok": "✅ You are checked in. Thank you!",
    "enroll": "✅ Checked in and face enrolled (the instructor will confirm your enrollment).",
    "no_gesture": "No hand gesture was detected. Hold your hand clearly next to your face and try again.",
    "wrong_gesture": "That was not the requested gesture. Try again with the new gesture.",
    "no_face": "No face was found. Make sure your face is visible and well lit.",
    "face_mismatch": "Your face does not match the enrolled student for this ID.",
    "challenge_expired": "Too slow - the gesture challenge expired. Try again.",
    "stale_photo": "That photo is not fresh. Take a new photo with the camera.",
    "bad_image": "Could not read the image. Try again.",
}


def _secret(data_dir: Path, env: str, fname: str, nbytes: int) -> str:
    import os
    if os.environ.get(env):
        return os.environ[env]
    p = data_dir / fname
    if p.exists():
        return p.read_text().strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    val = secrets.token_urlsafe(nbytes)
    p.write_text(val)
    return val


def decode_image(data: bytes, max_side: int = 1920) -> tuple[np.ndarray, datetime | None]:
    """Decode an upload (applying EXIF rotation) -> BGR array and the EXIF capture time if present."""
    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(data))
    taken = None
    try:
        exif = im.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal / DateTime
        if raw:
            taken = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except Exception:
        taken = None
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    return np.asarray(im)[:, :, ::-1].copy(), taken


def create_app(cfg: dict | None = None, verifier=None, store: Store | None = None) -> FastAPI:
    cfg = cfg or load_config()
    A = cfg["app"]
    db_path = resolve_path(A["db_path"])
    data_dir = db_path.parent
    store = store or Store(db_path)
    secret = _secret(data_dir, "GA_SECRET", "secret.key", 32).encode()
    instructor_key = _secret(data_dir, "GA_INSTRUCTOR_KEY", "instructor_key.txt", 9)
    rot = RotatingToken(secret, A["qr_period_s"], A["grace_windows"])
    signer = Signer(secret)
    gestures = list(cfg["gestures"])
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    app = FastAPI(title="Gesture attendance")
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.state.store, app.state.instructor_key, app.state.rot, app.state.signer = store, instructor_key, rot, signer
    app.state.verifier = verifier
    vlock = threading.Lock()  # one verification at a time (YOLO / onnx sessions are not shared safely)

    def get_verifier():
        if app.state.verifier is None:
            from ..verify import build_verifier
            app.state.verifier = build_verifier(cfg)
        return app.state.verifier

    # ------------------------------------------------------------------ helpers
    def is_instructor(request: Request) -> bool:
        k = (request.query_params.get("key") or request.headers.get("x-instructor-key")
             or request.cookies.get(INSTR_COOKIE))
        return bool(k) and secrets.compare_digest(k, instructor_key)

    def require_instructor(request: Request):
        if not is_instructor(request):
            raise HTTPException(401, "instructor key required (open the page with ?key=...)")

    def instructor_page(request: Request, name: str, **ctx):
        if not is_instructor(request):
            return HTMLResponse("<h3>Instructor key required: open this page as ...?key=YOUR_KEY</h3>", 401)
        resp = templates.TemplateResponse(request, name, {"gestures": gestures, **ctx})
        if request.query_params.get("key"):
            resp.set_cookie(INSTR_COOKIE, instructor_key, httponly=True, samesite="lax", max_age=30 * 86400)
        return resp

    def ticket(request: Request) -> dict | None:
        t = signer.unsign(request.cookies.get(TICKET_COOKIE))
        sess = store.active_session()
        if t is None or sess is None or t.get("sid") != sess["id"]:
            return None
        return t

    def base_url(request: Request) -> str:
        return (A.get("base_url") or str(request.base_url)).rstrip("/")

    def board_state(request: Request) -> dict:
        sess = store.active_session()
        roster = len(store.students())
        if sess is None:
            return {"active": False, "roster": roster}
        return {"active": True, "session_id": sess["id"], "title": sess["title"], "roster": roster,
                "present": len(store.attendance(sess["id"])), "enroll_open": bool(sess["enroll_open"]),
                "token": rot.token(sess["id"]), "seconds_left": round(rot.seconds_left(), 2),
                "period": rot.period}

    # ------------------------------------------------------------------ public pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(request, "message.html", {
            "title": "Gesture attendance", "msg": "Scan the QR code on the board to check in."})

    @app.get("/q/{token}")
    def scan(token: str, request: Request):
        sess = store.active_session()
        if sess is None or not rot.verify(sess["id"], token):
            return templates.TemplateResponse(request, "message.html", {
                "title": "QR code expired", "msg": "This QR code has expired. Scan the code on the board again "
                "(codes change every few seconds, so photos of the code do not work)."}, status_code=410)
        dev = request.cookies.get(DEVICE_COOKIE) or secrets.token_urlsafe(9)
        resp = RedirectResponse("/checkin", status_code=303)
        resp.set_cookie(TICKET_COOKIE, signer.sign({"sid": sess["id"], "dev": dev}, A["ticket_ttl_s"]),
                        httponly=True, samesite="lax", max_age=int(A["ticket_ttl_s"]))
        resp.set_cookie(DEVICE_COOKIE, dev, httponly=True, samesite="lax", max_age=365 * 86400)
        return resp

    @app.get("/checkin", response_class=HTMLResponse)
    def checkin_page(request: Request):
        t = ticket(request)
        if t is None:
            return templates.TemplateResponse(request, "message.html", {
                "title": "Please scan again", "msg": "Your check-in link has expired. Scan the QR code on the board."},
                status_code=410)
        sess = store.active_session()
        left = int(t["exp"] - time.time())
        return templates.TemplateResponse(request, "checkin.html", {"title": sess["title"], "seconds_left": left})

    @app.post("/api/challenge")
    def new_challenge(request: Request, student_id: str = Form(...)):
        t = ticket(request)
        if t is None:
            raise HTTPException(401, "Check-in link expired - scan the QR code again.")
        sess = store.active_session()
        sid = student_id.strip()
        st = store.student(sid)
        if st is None:
            raise HTTPException(404, "This student ID is not on the class roster.")
        if store.is_present(sess["id"], sid):
            return {"already_present": True, "name": st["name"]}
        if store.n_attempts(sess["id"], sid) >= A["max_attempts"]:
            store.flag(sess["id"], sid, "max_attempts", f"limit {A['max_attempts']} reached")
            raise HTTPException(429, "Attempt limit reached - please see the instructor.")
        enrolling = st["enroll_status"] == "none" or store.embedding(sid) is None
        if enrolling and not sess["enroll_open"]:
            raise HTTPException(403, "You are not enrolled yet. Ask the instructor to open enrollment.")
        others = store.students_on_device(sess["id"], t["dev"]) - {sid}
        if others:
            store.flag(sess["id"], sid, "shared_device", "same phone also used for " + ", ".join(sorted(others)))
        cid = secrets.token_urlsafe(12)
        g = secrets.choice(gestures)
        store.add_challenge(cid, sess["id"], sid, g, t["dev"])
        return {"challenge_id": cid, "gesture": g, "text": gesture_text(g), "name": st["name"],
                "expires_in": A["challenge_ttl_s"], "enrolling": enrolling,
                "attempts_left": A["max_attempts"] - store.n_attempts(sess["id"], sid)}

    @app.post("/api/verify")
    async def verify(request: Request, challenge_id: str = Form(...), image: UploadFile = File(...),
                     source: str = Form("camera")):
        t = ticket(request)
        if t is None:
            raise HTTPException(401, "Check-in link expired - scan the QR code again.")
        ch = store.take_challenge(challenge_id)
        if ch is None or ch["session_id"] != t["sid"] or ch["device_id"] != t["dev"]:
            raise HTTPException(400, "Unknown or already used challenge - request a new gesture.")
        data = await image.read()
        if len(data) > 12 * 2**20:
            raise HTTPException(413, "Image too large.")
        sid, sess_id = ch["student_id"], ch["session_id"]

        def finish(reason, accepted=False, pred="", conf=None, sim=None, latency=None):
            store.add_attempt(session_id=sess_id, student_id=sid, ts=time.time(), challenge=ch["gesture"],
                              pred=pred, conf=conf, face_sim=sim, accepted=int(accepted), reason=reason,
                              device_id=t["dev"], source=source[:16], latency_ms=latency)
            n = store.n_attempts(sess_id, sid)
            if not accepted and n >= A["max_attempts"]:
                store.flag(sess_id, sid, "max_attempts", f"limit {A['max_attempts']} reached")
            elif not accepted and n >= 2:
                store.flag(sess_id, sid, "repeat_attempts", f"{n} failed attempts")
            return {"accepted": accepted, "reason": reason, "message": MESSAGES.get(reason, reason),
                    "gesture_pred": pred, "attempts_left": max(0, A["max_attempts"] - n)}

        if time.time() - ch["created_at"] > A["challenge_ttl_s"]:
            return finish("challenge_expired")
        try:
            img, taken = decode_image(data)
        except Exception:
            return finish("bad_image")
        max_age = A.get("max_photo_age_s", 600)
        if taken is not None and max_age and abs((datetime.now() - taken).total_seconds()) > max_age:
            store.flag(sess_id, sid, "stale_photo", f"EXIF time {taken:%Y-%m-%d %H:%M}")
            return finish("stale_photo")

        enrolled = store.embedding(sid)
        if enrolled is None and not (store.session(sess_id) or {}).get("enroll_open"):
            raise HTTPException(403, "Enrollment is closed.")
        with vlock:
            v = get_verifier()
            d = v.verify(img, ch["gesture"], enrolled)
        del img  # the selfie is never stored

        if d.accepted and d.reason == "enroll":
            dup_thr = cfg["verification"].get("duplicate_face_threshold", 0.5)
            for other, e in store.all_embeddings().items():
                if other != sid and cosine(d.embedding, e) >= dup_thr:
                    store.flag(sess_id, sid, "duplicate_face", f"face matches enrolled student {other}")
            store.set_embedding(sid, d.embedding, "pending")
        if d.accepted:
            store.mark_present(sess_id, sid, t["dev"], d.face_sim,
                               "enrollment" if d.reason == "enroll" else "gesture+face")
        elif d.reason == "face_mismatch":
            store.flag(sess_id, sid, "face_mismatch", f"similarity {d.face_sim:.2f}")
        return finish(d.reason, d.accepted, d.gesture_pred, d.gesture_conf, d.face_sim, d.timings_ms.get("total"))

    # ------------------------------------------------------------------ instructor pages
    @app.get("/instructor", response_class=HTMLResponse)
    def instructor(request: Request):
        return instructor_page(request, "instructor.html")

    @app.get("/board", response_class=HTMLResponse)
    def board(request: Request):
        return instructor_page(request, "board.html")

    @app.get("/collect", response_class=HTMLResponse)
    def collect_page(request: Request):
        return instructor_page(request, "collect.html")

    # ------------------------------------------------------------------ instructor APIs
    dep = [Depends(require_instructor)]

    @app.get("/api/board/state", dependencies=dep)
    def api_board_state(request: Request):
        return board_state(request)

    @app.get("/api/board/qr.png", dependencies=dep)
    def api_qr(request: Request):
        import qrcode
        sess = store.active_session()
        if sess is None:
            raise HTTPException(404, "no active session")
        url = f"{base_url(request)}/q/{rot.token(sess['id'])}"
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=12, border=2)
        qr.add_data(url)
        buf = io.BytesIO()
        qr.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/roster", dependencies=dep)
    async def api_roster(file: UploadFile = File(...)):
        try:
            rows = parse_roster(await file.read(), file.filename or "roster.csv")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"imported": store.upsert_students(rows), "roster": len(store.students())}

    @app.post("/api/session/start", dependencies=dep)
    def api_start(title: str = Form(""), enroll_open: bool = Form(False)):
        title = title.strip() or datetime.now().strftime("Lecture %Y-%m-%d %H:%M")
        return {"session_id": store.start_session(title, enroll_open)}

    @app.post("/api/session/end", dependencies=dep)
    def api_end():
        sess = store.active_session()
        if sess:
            store.end_session(sess["id"])
        return {"ended": sess["id"] if sess else None}

    @app.post("/api/session/enroll", dependencies=dep)
    def api_enroll(open: bool = Form(...)):
        sess = store.active_session()
        if sess is None:
            raise HTTPException(404, "no active session")
        store.set_enroll_open(sess["id"], open)
        return {"enroll_open": open}

    @app.get("/api/session/stats", dependencies=dep)
    def api_stats(session_id: int | None = None):
        sess = store.session(session_id) if session_id else (store.active_session() or (store.sessions() or [None])[0])
        students = store.students()
        if sess is None:
            return {"session": None, "students": students, "present": 0, "roster": len(students), "flags": []}
        att = store.attendance(sess["id"])
        flags = store.flags(sess["id"])
        for s in students:
            a = att.get(s["student_id"])
            s["present"] = bool(a)
            s["checkin_ts"] = a["ts"] if a else None
            s["method"] = a["method"] if a else None
            s["flags"] = sorted({f["kind"] for f in flags if f["student_id"] == s["student_id"]})
        atts = store.attempts(sess["id"])
        lat = [a["latency_ms"] for a in atts if a["latency_ms"]]
        return {"session": sess, "students": students, "present": len(att), "roster": len(students),
                "flags": flags, "recent_attempts": atts[-15:][::-1],
                "median_latency_ms": float(np.median(lat)) if lat else None,
                "pending_enrollments": [s for s in students if s["enroll_status"] == "pending"]}

    @app.get("/api/sessions", dependencies=dep)
    def api_sessions():
        return store.sessions()

    @app.post("/api/enrollment/{student_id}/{action}", dependencies=dep)
    def api_enrollment(student_id: str, action: str):
        if store.student(student_id) is None:
            raise HTTPException(404, "unknown student")
        if action == "confirm":
            store.set_enroll_status(student_id, "confirmed")
        elif action in ("reject", "reset"):
            store.set_embedding(student_id, None, "none")
            sess = store.active_session()
            if sess and action == "reject":
                store.unmark(sess["id"], student_id)
        else:
            raise HTTPException(400, "action must be confirm, reject or reset")
        return {"student_id": student_id, "action": action}

    @app.post("/api/attendance/{student_id}", dependencies=dep)
    def api_manual(student_id: str, present: bool = Form(...), session_id: int | None = Form(None)):
        sess = store.session(session_id) if session_id else store.active_session()
        if sess is None or store.student(student_id) is None:
            raise HTTPException(404, "unknown session or student")
        if present:
            store.mark_present(sess["id"], student_id, method="manual (instructor)")
        else:
            store.unmark(sess["id"], student_id)
        return {"student_id": student_id, "present": present}

    @app.get("/api/report.xlsx", dependencies=dep)
    def api_report(session_id: int | None = None):
        sess = store.session(session_id) if session_id else (store.active_session() or (store.sessions() or [None])[0])
        if sess is None:
            raise HTTPException(404, "no sessions yet")
        name = f"attendance_session{sess['id']}_{datetime.fromtimestamp(sess['started_at']):%Y%m%d}.xlsx"
        return Response(session_report(store, sess["id"]), headers={"Content-Disposition": f'attachment; filename="{name}"'},
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.get("/api/course_report.xlsx", dependencies=dep)
    def api_course_report():
        return Response(course_report(store), headers={"Content-Disposition": 'attachment; filename="course_attendance.xlsx"'},
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ------------------------------------------------------------------ volunteer test-set collection
    collect_dir = resolve_path(A.get("collect_dir", "data/classroom"))

    @app.post("/api/collect", dependencies=dep)
    async def api_collect(subject_id: str = Form(...), gesture: str = Form(...), condition: str = Form(...),
                          background_gesture: str = Form(""), consent: bool = Form(False),
                          image: UploadFile = File(...)):
        if not consent:
            raise HTTPException(400, "consent is required")
        safe = lambda s: "".join(ch for ch in s.strip() if ch.isalnum() or ch in "-_") or "x"  # noqa: E731
        img, _ = decode_image(await image.read(), max_side=4096)
        from PIL import Image
        (collect_dir / "images").mkdir(parents=True, exist_ok=True)
        stem = f"{safe(subject_id)}_{safe(gesture)}_{safe(condition)}"
        k = 1
        while (collect_dir / "images" / f"{stem}_{k}.jpg").exists():
            k += 1
        rel = f"images/{stem}_{k}.jpg"
        Image.fromarray(img[:, :, ::-1]).save(collect_dir / rel, quality=95)
        labels = collect_dir / "labels.csv"
        new = not labels.exists()
        with open(labels, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "subject_id", "gesture", "condition", "background_gesture"])
            w.writerow([rel, subject_id.strip(), gesture, condition, background_gesture])
        return {"saved": rel}

    return app

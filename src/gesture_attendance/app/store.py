"""SQLite persistence. Only face EMBEDDINGS are stored - never the selfies."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS students(
    student_id TEXT PRIMARY KEY, name TEXT NOT NULL,
    embedding BLOB, enrolled_at REAL, enroll_status TEXT NOT NULL DEFAULT 'none');  -- none|pending|confirmed
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, started_at REAL, ended_at REAL,
    enroll_open INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS challenges(
    id TEXT PRIMARY KEY, session_id INTEGER, student_id TEXT, gesture TEXT, created_at REAL,
    used INTEGER NOT NULL DEFAULT 0, device_id TEXT);
CREATE TABLE IF NOT EXISTS attempts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, student_id TEXT, ts REAL, challenge TEXT,
    pred TEXT, conf REAL, face_sim REAL, accepted INTEGER, reason TEXT, device_id TEXT, source TEXT,
    latency_ms REAL);
CREATE TABLE IF NOT EXISTS attendance(
    session_id INTEGER, student_id TEXT, ts REAL, device_id TEXT, face_sim REAL, method TEXT,
    PRIMARY KEY(session_id, student_id));
CREATE TABLE IF NOT EXISTS flags(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, student_id TEXT, ts REAL, kind TEXT, detail TEXT);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _x(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------ roster
    def upsert_students(self, rows: list[tuple[str, str]]) -> int:
        with self.lock:
            for sid, name in rows:
                self.db.execute("INSERT INTO students(student_id, name) VALUES(?, ?) "
                                "ON CONFLICT(student_id) DO UPDATE SET name=excluded.name", (sid, name))
            self.db.commit()
        return len(rows)

    def students(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT student_id, name, enroll_status, enrolled_at FROM students "
                                         "ORDER BY name COLLATE NOCASE")]

    def student(self, sid: str) -> dict | None:
        r = self._q("SELECT * FROM students WHERE student_id=?", (sid,))
        return dict(r[0]) if r else None

    def embedding(self, sid: str) -> np.ndarray | None:
        r = self._q("SELECT embedding FROM students WHERE student_id=?", (sid,))
        return None if not r or r[0]["embedding"] is None else np.frombuffer(r[0]["embedding"], dtype=np.float32).copy()

    def all_embeddings(self) -> dict[str, np.ndarray]:
        return {r["student_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
                for r in self._q("SELECT student_id, embedding FROM students WHERE embedding IS NOT NULL")}

    def set_embedding(self, sid: str, emb: np.ndarray | None, status: str):
        blob = None if emb is None else np.asarray(emb, np.float32).tobytes()
        self._x("UPDATE students SET embedding=?, enrolled_at=?, enroll_status=? WHERE student_id=?",
                (blob, time.time() if emb is not None else None, status, sid))

    def set_enroll_status(self, sid: str, status: str):
        self._x("UPDATE students SET enroll_status=? WHERE student_id=?", (status, sid))

    # ------------------------------------------------------------------ sessions
    def start_session(self, title: str, enroll_open: bool = False) -> int:
        with self.lock:
            self.db.execute("UPDATE sessions SET ended_at=? WHERE ended_at IS NULL", (time.time(),))
            self.db.commit()
        return self._x("INSERT INTO sessions(title, started_at, enroll_open) VALUES(?, ?, ?)",
                       (title, time.time(), int(enroll_open)))

    def end_session(self, session_id: int):
        self._x("UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL", (time.time(), session_id))

    def active_session(self) -> dict | None:
        r = self._q("SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1")
        return dict(r[0]) if r else None

    def session(self, session_id: int) -> dict | None:
        r = self._q("SELECT * FROM sessions WHERE id=?", (session_id,))
        return dict(r[0]) if r else None

    def sessions(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM sessions ORDER BY id DESC")]

    def set_enroll_open(self, session_id: int, open_: bool):
        self._x("UPDATE sessions SET enroll_open=? WHERE id=?", (int(open_), session_id))

    # ------------------------------------------------------------------ challenges / attempts
    def add_challenge(self, cid: str, session_id: int, sid: str, gesture: str, device_id: str):
        self._x("INSERT INTO challenges(id, session_id, student_id, gesture, created_at, device_id) "
                "VALUES(?, ?, ?, ?, ?, ?)", (cid, session_id, sid, gesture, time.time(), device_id))

    def take_challenge(self, cid: str) -> dict | None:
        """Atomically mark a challenge used (one selfie per challenge) and return it."""
        with self.lock:
            r = self.db.execute("SELECT * FROM challenges WHERE id=? AND used=0", (cid,)).fetchone()
            if r is None:
                return None
            self.db.execute("UPDATE challenges SET used=1 WHERE id=?", (cid,))
            self.db.commit()
            return dict(r)

    def add_attempt(self, **kw) -> int:
        cols = ",".join(kw)
        return self._x(f"INSERT INTO attempts({cols}) VALUES({','.join('?' * len(kw))})", tuple(kw.values()))

    def n_attempts(self, session_id: int, sid: str) -> int:
        return self._q("SELECT COUNT(*) n FROM attempts WHERE session_id=? AND student_id=?", (session_id, sid))[0]["n"]

    def attempts(self, session_id: int) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM attempts WHERE session_id=? ORDER BY ts", (session_id,))]

    # ------------------------------------------------------------------ attendance
    def mark_present(self, session_id: int, sid: str, device_id: str = "", face_sim: float | None = None,
                     method: str = "gesture+face"):
        self._x("INSERT OR REPLACE INTO attendance(session_id, student_id, ts, device_id, face_sim, method) "
                "VALUES(?, ?, ?, ?, ?, ?)", (session_id, sid, time.time(), device_id, face_sim, method))

    def unmark(self, session_id: int, sid: str):
        self._x("DELETE FROM attendance WHERE session_id=? AND student_id=?", (session_id, sid))

    def is_present(self, session_id: int, sid: str) -> bool:
        return bool(self._q("SELECT 1 FROM attendance WHERE session_id=? AND student_id=?", (session_id, sid)))

    def attendance(self, session_id: int) -> dict[str, dict]:
        return {r["student_id"]: dict(r) for r in self._q("SELECT * FROM attendance WHERE session_id=?", (session_id,))}

    def students_on_device(self, session_id: int, device_id: str) -> set[str]:
        rows = self._q("SELECT DISTINCT student_id FROM challenges WHERE session_id=? AND device_id=?",
                       (session_id, device_id))
        return {r["student_id"] for r in rows}

    # ------------------------------------------------------------------ flags
    def flag(self, session_id: int, sid: str, kind: str, detail: str = ""):
        exists = self._q("SELECT 1 FROM flags WHERE session_id=? AND student_id=? AND kind=? AND detail=?",
                         (session_id, sid, kind, detail))
        if not exists:
            self._x("INSERT INTO flags(session_id, student_id, ts, kind, detail) VALUES(?, ?, ?, ?, ?)",
                    (session_id, sid, time.time(), kind, detail))

    def flags(self, session_id: int) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM flags WHERE session_id=? ORDER BY ts", (session_id,))]

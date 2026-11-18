"""Rotating QR tokens and signed short-lived tickets (HMAC-SHA256, no external dependencies).

QR token  : HMAC(secret, "<session>:<window>"), window = floor(time / period). A new token every
            `period` seconds; the server also accepts the previous `grace` windows (scan latency).
            A screenshot forwarded to an absent friend expires within period * (grace + 1) seconds.
Ticket    : after a valid scan the phone gets a signed cookie {session, device, exp}, so the student
            has `ticket_ttl` seconds to finish (pick ID, perform gesture) without racing the QR.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Callable


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class RotatingToken:
    def __init__(self, secret: bytes, period: float = 3.0, grace: int = 1, clock: Callable[[], float] = time.time):
        self.secret, self.period, self.grace, self.clock = secret, period, grace, clock

    def window(self, t: float | None = None) -> int:
        return int((self.clock() if t is None else t) // self.period)

    def token(self, session_id: int, window: int | None = None) -> str:
        w = self.window() if window is None else window
        mac = hmac.new(self.secret, f"qr:{session_id}:{w}".encode(), hashlib.sha256).digest()[:12]
        return _b64(mac)

    def verify(self, session_id: int, token: str) -> bool:
        w = self.window()
        return any(hmac.compare_digest(token, self.token(session_id, w - k)) for k in range(self.grace + 1))

    def seconds_left(self) -> float:
        t = self.clock()
        return self.period - (t % self.period)


class Signer:
    def __init__(self, secret: bytes, clock: Callable[[], float] = time.time):
        self.secret, self.clock = secret, clock

    def sign(self, data: dict, ttl: float) -> str:
        body = _b64(json.dumps({**data, "exp": self.clock() + ttl}, separators=(",", ":")).encode())
        mac = _b64(hmac.new(self.secret, b"ticket:" + body.encode(), hashlib.sha256).digest()[:16])
        return f"{body}.{mac}"

    def unsign(self, token: str | None) -> dict | None:
        if not token or "." not in token:
            return None
        body, mac = token.rsplit(".", 1)
        good = _b64(hmac.new(self.secret, b"ticket:" + body.encode(), hashlib.sha256).digest()[:16])
        if not hmac.compare_digest(mac, good):
            return None
        try:
            data = json.loads(_unb64(body))
        except Exception:
            return None
        return data if data.get("exp", 0) >= self.clock() else None

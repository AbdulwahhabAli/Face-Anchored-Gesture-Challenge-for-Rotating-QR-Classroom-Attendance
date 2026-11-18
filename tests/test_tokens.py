from gesture_attendance.app.tokens import RotatingToken, Signer


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_token_rotates_and_expires():
    c = Clock(1000.0)
    rt = RotatingToken(b"k", period=3, grace=1, clock=c)
    tok = rt.token(1)
    assert rt.verify(1, tok)
    assert not rt.verify(2, tok)            # bound to the session
    c.t += 3                                # next window: still accepted (grace)
    assert rt.verify(1, tok) and rt.token(1) != tok
    c.t += 3                                # two windows later: rejected (screenshot replay)
    assert not rt.verify(1, tok)


def test_token_secret_matters():
    a, b = RotatingToken(b"a", clock=Clock()), RotatingToken(b"b", clock=Clock())
    assert not b.verify(1, a.token(1))


def test_signer_roundtrip_tamper_expiry():
    c = Clock()
    s = Signer(b"k", clock=c)
    t = s.sign({"sid": 3, "dev": "x"}, ttl=60)
    assert s.unsign(t)["sid"] == 3
    body, mac = t.rsplit(".", 1)
    assert s.unsign(body[:-2] + "AA." + mac) is None
    assert Signer(b"other", clock=c).unsign(t) is None
    c.t += 61
    assert s.unsign(t) is None

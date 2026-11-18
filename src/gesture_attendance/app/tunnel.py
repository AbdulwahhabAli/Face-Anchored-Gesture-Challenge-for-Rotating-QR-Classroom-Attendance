"""Free HTTPS for phones via a Cloudflare quick tunnel (no account needed).

Phones need HTTPS for the live camera preview, and a public URL also avoids campus Wi-Fi
blocking phone -> laptop connections. cloudflared is downloaded on first use into tools/.
"""
from __future__ import annotations

import atexit
import platform
import re
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

from ..config import REPO_ROOT

TOOLS = REPO_ROOT / "tools"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _binary() -> Path:
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system == "windows":
        name = f"cloudflared-windows-{arch}.exe"
    elif system == "darwin":
        name = f"cloudflared-darwin-{arch}.tgz"
    else:
        name = f"cloudflared-linux-{arch}"
    dst = TOOLS / ("cloudflared.exe" if system == "windows" else "cloudflared")
    if dst.exists():
        return dst
    TOOLS.mkdir(exist_ok=True)
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{name}"
    print(f"[tunnel] downloading cloudflared ({name}) ...")
    tmp = TOOLS / (name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if name.endswith(".tgz"):
        import tarfile
        with tarfile.open(tmp) as t:
            t.extractall(TOOLS)
        tmp.unlink()
    else:
        tmp.replace(dst)
    dst.chmod(0o755)
    return dst


def start_tunnel(port: int, timeout: float = 45.0) -> str | None:
    """Start `cloudflared tunnel --url http://localhost:<port>`; return the public https URL or None."""
    try:
        exe = _binary()
    except Exception as e:
        print(f"[tunnel] could not get cloudflared: {e}")
        return None
    proc = subprocess.Popen([str(exe), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    atexit.register(proc.terminate)
    found: list[str] = []
    registered = threading.Event()

    def reader():
        for line in proc.stdout:
            if not found:
                m = URL_RE.search(line)
                if m:
                    found.append(m.group(0))
            if "Registered tunnel connection" in line:
                registered.set()

    threading.Thread(target=reader, daemon=True).start()
    # a URL alone is not enough: only use it once cloudflared has actually connected to Cloudflare,
    # otherwise the QR code would point to a dead link
    registered.wait(timeout)
    if not (found and registered.is_set()):
        print("[tunnel] tunnel did not connect (offline, or the network blocks Cloudflare tunnels)")
        proc.terminate()
        return None
    return found[0]

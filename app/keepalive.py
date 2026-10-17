"""Optional keep-warm self-ping for ephemeral free-tier hosts.

Render's free tier spins a service down after ~15 minutes without traffic;
services set RENDER_EXTERNAL_URL (or APP_URL) and a background thread pings
/healthz every five minutes. Gated on the environment — never runs locally.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request


def start_if_configured() -> None:
    base = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL") or ""
    if not base:
        return

    def _ping() -> None:  # pragma: no cover — runs only on ephemeral hosts
        while True:
            try:
                urllib.request.urlopen(f"{base}/healthz", timeout=15).read(64)
            except Exception:
                pass
            time.sleep(300)

    threading.Thread(target=_ping, daemon=True, name="keep-warm").start()

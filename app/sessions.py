"""Per-visitor case + derived-stage cache, and the bounded session store.

One visitor (identified by an httponly session cookie) works on one image at
a time. Derived stages — the Frangi preprocess stack and the ONNX probability
map — are cached per CLAHE value so scrubbing sliders is instant after the
first render.

Redis was deliberately not added: this is a single-process reading station.
The store is a thread-safe bounded LRU dict; the least recently used session
is evicted when the cap is hit. Each cached session costs ~6 MB of arrays.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

from app.config import MAX_SESSIONS, THUMB_SIZE
from app.imaging import encode_png, fit_long_edge
from app.pipeline import infer_with_timing, load_case_image, preprocess_with_timing

_COMPUTE_LOCK = threading.Lock()  # serialised globally: bounds transient memory


class CaseSession:
    """One visitor's current case image + cached derived stages."""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.case = "bench"
        self.mode = "overlay"  # plate mode persists across tune updates
        self.upload_bytes: bytes | None = None
        self.upload_name = ""
        self.thumb_bytes = b""
        self.rgb = np.zeros((1, 1, 3), np.uint8)
        self.title = ""
        self.h = 0
        self.w = 0
        self._lock = threading.Lock()
        self._clip: float | None = None
        self._pre: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._prep_ms = 0.0
        self._prob: np.ndarray | None = None
        self._infer_ms = 0.0
        self.set_case("bench")

    # ------------------------------------------------------------ state changes
    def set_case(self, which: str) -> None:
        with self._lock:
            self.case = which
            self.rgb, self.title = load_case_image(which, self.upload_bytes, self.upload_name)
            self.h, self.w = self.rgb.shape[:2]
            self._invalidate()

    def set_upload(self, rgb_bytes: bytes, name: str) -> None:
        """Store an uploaded image as the current case, with a cached thumbnail."""
        self.upload_bytes = rgb_bytes
        self.upload_name = name or "Uploaded fundus"
        self.set_case("upload")
        thumbnail = fit_long_edge(self.rgb, THUMB_SIZE)
        self.thumb_bytes = encode_png(thumbnail)

    def _invalidate(self) -> None:
        self._clip = None
        self._pre = None
        self._prob = None

    # ----------------------------------------------------------- derived stages
    def pre(self, clip: float) -> tuple:
        """(green, clahe, frangi, stacked, prep_ms, cached) for this CLAHE value."""
        if self._hit(clip)[0]:
            return self._hit(clip)[1]
        with _COMPUTE_LOCK:
            if self._hit(clip)[0]:
                return self._hit(clip)[1]
            green, clahe, frangi, stacked, ms = preprocess_with_timing(self.rgb, clip)
            with self._lock:
                self._pre = (green, clahe, frangi, stacked)
                self._clip = clip
                self._prep_ms = ms
                self._prob = None  # stacked changed -> probability map stale
            return green, clahe, frangi, stacked, ms, False

    def prob(self, clip: float) -> tuple:
        """(prob, infer_ms, cached) for this clip, computing preprocess if needed."""
        self.pre(clip)  # guarantees a fused stack for this clip exists
        cached = self._cached_prob(clip)
        if cached is not None:
            return cached.copy(), self._infer_ms, True
        with _COMPUTE_LOCK:
            with self._lock:
                if self._clip == clip and self._prob is not None:
                    return self._prob.copy(), self._infer_ms, True
                stacked = self._pre[3]
            prob, ms = infer_with_timing(stacked)
            with self._lock:
                self._prob = prob
                self._infer_ms = ms
            return prob, ms, False

    # ---------------------------------------------------------------- internals
    def _hit(self, clip: float) -> tuple[bool, tuple | None]:
        with self._lock:
            if self._clip == clip and self._pre is not None:
                return True, (*self._pre, self._prep_ms, True)
            return False, None

    def _cached_prob(self, clip: float) -> np.ndarray | None:
        with self._lock:
            if self._clip == clip and self._prob is not None:
                return self._prob
            return None


# ---------------------------------------------------------------------- store
_sessions: OrderedDict[str, CaseSession] = OrderedDict()
_store_lock = threading.Lock()


def get_session(sid: str | None) -> CaseSession:
    """Return the visitor's session, creating + registering one when missing."""
    with _store_lock:
        if sid and sid in _sessions:
            sess = _sessions[sid]
            _sessions.move_to_end(sid)  # refresh recency for LRU eviction
            return sess
        fresh = CaseSession(sid or f"c{os.urandom(6).hex()}")
        while len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)  # evict the least recently used
        _sessions[fresh.sid] = fresh
        return fresh


def session_count() -> int:
    with _store_lock:
        return len(_sessions)

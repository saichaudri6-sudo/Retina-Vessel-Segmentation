"""Serving pipeline for the RetinaVessel app.

Preprocessing is reused verbatim from utils.preprocessing (unchanged). The
trained UNet runs through ONNX Runtime (weights/vessel_unet.onnx) instead of
PyTorch at serving time — verified numerically against the torch model by
scripts/export_onnx.py, which maxes at |diff| ~1e-4.

Also provides: colored probability plates from runtime LUTs, thresholded
masks, red overlays, sample generation fallback, and per-session caching.
"""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from colormaps import MAPS
from utils.preprocessing import preprocess_fundus_image

WEIGHTS_PATH = os.path.join("weights", "vessel_unet.onnx")
WORK_MAX = 640  # uploaded images are downscaled so the long edge is at most this
VESSEL_RGB = np.array([255.0, 69.0, 58.0], dtype=np.float32)

MODES: Tuple[str, ...] = ("overlay", "prob", "mask", "orig", "green", "clahe", "frangi")

SAMPLES: Dict[str, Tuple[str, str, int]] = {
    "bench": (os.path.join("samples", "fundus_sample_1.png"), "Benchmark retinal fundus", 42),
    "dense": (os.path.join("samples", "fundus_sample_2.png"), "Dense microvascular retina", 123),
}

_ort_session: Optional[InferenceSession] = None
_ort_lock = threading.Lock()


def ort_session() -> InferenceSession:
    """Thread-safe lazily initialized ONNX inference session."""
    global _ort_session
    if _ort_session is None:
        with _ort_lock:
            if _ort_session is None:
                opts = SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.log_severity_level = 3
                # 512 MB hosts: the default CPU arena reserves hundreds of MB and
                # keeps growing across inferences. Arena-free runs in +27 MB.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                _ort_session = InferenceSession(
                    WEIGHTS_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
                )
    return _ort_session


def infer_vessels(stacked: np.ndarray) -> np.ndarray:
    """Run the ONNX UNet on a (H, W, 3) float32 stack; returns (H, W) in [0, 1]."""
    out = ort_session().run(["prob"], {"stacked": stacked[None].transpose(0, 3, 1, 2)})
    return out[0][0, 0].astype(np.float32)


def generate_sample_fundus_image(size: int = 512, seed: int = 42) -> np.ndarray:
    """Synthetic fundus fallback when a sample file is missing. Returns uint8 RGB."""
    np.random.seed(seed)
    h, w = size, size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    fov = size * 0.46
    mask = (dist <= fov).astype(np.float32)

    red = 190 + 35 * np.sin(x / 90.0) - 40 * (dist / fov) ** 2
    green = 85 + 20 * np.cos(y / 90.0) - 30 * (dist / fov) ** 2
    blue = 20 + 8 * np.sin((x + y) / 120.0)

    dx, dy = int(w * 0.32), int(h * 0.5)
    disc = np.clip(1.0 - np.sqrt((x - dx) ** 2 + (y - dy) ** 2) / (size * 0.08), 0.0, 1.0)
    red += disc * 60
    green += disc * 85
    blue += disc * 40

    fx, fy = int(w * 0.62), int(h * 0.52)
    fovea = np.clip(1.0 - np.sqrt((x - fx) ** 2 + (y - fy) ** 2) / (size * 0.12), 0.0, 1.0)
    red -= fovea * 35
    green -= fovea * 25

    vessels = np.zeros((h, w), dtype=np.float32)
    for ang in np.linspace(-np.pi * 0.88, np.pi * 0.88, 14):
        px, py = float(dx), float(dy)
        vx, vy = np.cos(ang) * 1.8, np.sin(ang) * 1.8
        thick = 4.2
        for _ in range(int(size * 0.56)):
            ix, iy = int(px), int(py)
            if 0 <= ix < w and 0 <= iy < h:
                r = max(1, int(thick))
                vessels[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = np.maximum(
                    vessels[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1], thick / 4.2
                )
            vx += np.random.uniform(-0.16, 0.16)
            vy += np.random.uniform(-0.16, 0.16)
            px += vx
            py += vy
            thick = max(1.0, thick - 0.016)
            if not (0 <= px < w and 0 <= py < h):
                break

    green -= vessels * mask * 60.0
    red -= vessels * mask * 35.0
    return np.stack([
        np.clip(red * mask, 0, 255).astype(np.uint8),
        np.clip(green * mask, 0, 255).astype(np.uint8),
        np.clip(blue * mask, 0, 255).astype(np.uint8),
    ], axis=-1)


def load_case_image(which: str, rgb_bytes: Optional[bytes] = None,
                    name: str = "") -> Tuple[np.ndarray, str]:
    """Resolve a case key to a work-sized uint8 RGB image + display title."""
    if which == "upload" and rgb_bytes is not None:
        img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        img.thumbnail((WORK_MAX, WORK_MAX), Image.LANCZOS)
        title = name or "Uploaded fundus"
        return np.asarray(img), title

    path, title, seed = SAMPLES[which]  # bench / dense
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((WORK_MAX, WORK_MAX), Image.LANCZOS)
        return np.asarray(img), title
    return generate_sample_fundus_image(512, seed), title


def precompute(rgb: np.ndarray, clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Green / CLAHE / Frangi / stacked, plus elapsed ms."""
    t0 = time.perf_counter()
    green_ch, clahe_img, frangi_img, stacked = preprocess_fundus_image(rgb, clip_limit=clip)
    return green_ch, clahe_img, frangi_img, stacked, (time.perf_counter() - t0) * 1000.0


def binary_of(prob: np.ndarray, cutoff: float) -> np.ndarray:
    return (prob >= cutoff).astype(np.uint8) * 255


def overlay_plate(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    hit = mask > 0
    out[hit] = (1.0 - alpha) * out[hit] + alpha * VESSEL_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def heat_plate(prob: np.ndarray, palette: str) -> np.ndarray:
    lut = MAPS.get(palette, MAPS["magma"])
    idx = np.clip((prob * 255.0).astype(np.uint8), 0, 255)
    return lut[idx]


def encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_jpeg(arr: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_jpg_from_prob(prob: np.ndarray) -> bytes:
    """Frangi-view convenience: normalized float, grayscale."""
    g = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    return encode_jpeg(g)


class CaseSession:
    """One visitor's current case image + derived-stage cache."""

    def __init__(self, sid: str):
        self.sid = sid
        self.case = "bench"
        self.mode = "overlay"  # eyepiece plate mode, persists across tune updates
        self.upload_bytes: Optional[bytes] = None
        self.upload_name = ""
        self.rgb: np.ndarray = np.zeros((1, 1, 3), np.uint8)
        self.title = ""
        self.h = 0
        self.w = 0
        self._lock = threading.Lock()
        self._clip: Optional[float] = None
        self._pre: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._prep_ms = 0.0
        self._clip_prob: Optional[float] = None
        self._prob: Optional[np.ndarray] = None
        self._infer_ms = 0.0
        self.set_case("bench")

    # --- state -----------------------------------------------------------------
    def set_case(self, which: str) -> None:
        with self._lock:
            self.case = which
            self.rgb, self.title = load_case_image(which, self.upload_bytes, self.upload_name)
            self.h, self.w = self.rgb.shape[:2]
            self._clip = self._clip_prob = None
            self._pre = self._prob = None

    def set_upload(self, rgb_bytes: bytes, name: str) -> None:
        self.upload_bytes = rgb_bytes
        self.upload_name = name or "Uploaded fundus"
        self.set_case("upload")

    # --- derived stages ----------------------------------------------------------
    def Pre(self, clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """(green, clahe, frangi, stacked, prep_ms, cached)"""
        with self._lock:
            if self._clip == clip and self._pre is not None:
                green, clahe, frangi, stacked = self._pre
                return green, clahe, frangi, stacked, self._prep_ms, True
            green, clahe, frangi, stacked, ms = precompute(self.rgb, clip)
            keep = (green, clahe, frangi)
            self._pre = (*keep, stacked.copy())
            self._clip = clip
            self._clip_prob = None
            self._prob = None
            self._prep_ms = ms
            return green, clahe, frangi, stacked.copy(), ms, False

    def Prob(self, clip: float) -> Tuple[np.ndarray, float, bool]:
        """(prob, infer_ms, cached) — prob for this clip (fresh Pre if needed)."""
        self.Pre(clip)  # guarantees stacked for this clip exists
        with self._lock:
            if self._clip_prob == clip and self._prob is not None:
                return self._prob.copy(), self._infer_ms, True
            stacked = self._pre[3]
            t0 = time.perf_counter()
            prob = infer_vessels(stacked)
            ms = (time.perf_counter() - t0) * 1000.0
            self._prob = prob
            self._clip_prob = clip
            self._infer_ms = ms
            return prob.copy(), ms, False

    def meta(self) -> Tuple[int, int, str]:
        return self.h, self.w, self.title


_shepherd_lock = threading.Lock()
_SESSIONS: Dict[str, CaseSession] = {}
_SESSION_LIMIT = 24


def get_session(sid: Optional[str]) -> CaseSession:
    if not sid:
        sid = f"c{os.urandom(6).hex()}"
    with _shepherd_lock:
        sess = _SESSIONS.get(sid)
        if sess is None:
            while len(_SESSIONS) >= _SESSION_LIMIT:
                _SESSIONS.pop(next(iter(_SESSIONS)))
            sess = CaseSession(sid)
            _SESSIONS[sid] = sess
    return sess


def gray_to_rgb(g: np.ndarray) -> np.ndarray:
    """Single-channel uint8 -> 3-channel uint8."""
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def tile_placeholder(size: int = 112) -> bytes:
    """Dark dotted placeholder for the empty upload tile."""
    s = size
    img = np.zeros((s, s, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:s, 0:s]
    cx = cy = s / 2
    ring = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[..., 0] = 255
    img[..., 1] = 69
    img[..., 2] = 58
    edge = (ring > s * 0.18) & (ring < s * 0.2)
    bar = (np.abs(yy - cy) < s * 0.014) | (np.abs(xx - cx) < s * 0.014)
    glyph = (edge | bar) & (ring < s * 0.2)
    img[glyph] = (255, 255, 255)
    img[~glyph & (ring < s * 0.2)] = (8, 8, 10)
    img[(ring >= s * 0.2)] = (0, 0, 0)
    dotgrid = ((xx % 12 == 6) & (yy % 12 == 6)) & ~glyph
    img[dotgrid] = (34, 34, 36)
    return encode_png(img)


def clamp(p: str, lo: float, hi: float, default: float, step: float = 0.0) -> float:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return default
    v = min(hi, max(lo, v))
    if step:
        v = round(v / step) * step
    return v


def clamp_palette(p: Optional[str]) -> str:
    return p if p in MAPS else "magma"

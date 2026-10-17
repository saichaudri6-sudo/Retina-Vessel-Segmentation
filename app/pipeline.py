"""Model-adjacent logic shared by routes: case loading, sample fallback, plates.

Preprocessing is reused verbatim from utils.preprocessing (the trainable
source of truth — green channel, CLAHE, multiscale Frangi), so the app serves
identical numbers to the training/export pipeline.
"""

from __future__ import annotations

import time

import numpy as np

from app.colormaps import MAPS
from app.config import DEFAULT_PALETTE, SAMPLES, VESSEL_RGB, WORK_MAX
from app.imaging import decode_rgb, encode_jpeg, encode_png, fit_long_edge
from app.inference import compute_lock, infer_vessels
from utils.preprocessing import preprocess_fundus_image


# --------------------------------------------------------------------- cases
def load_case_image(which: str, upload_bytes: bytes | None, upload_name: str) -> tuple[np.ndarray, str]:
    """Resolve a case key to a work-sized (<=640px long edge) RGB image + title."""
    if which == "upload" and upload_bytes is not None:
        img = decode_rgb(upload_bytes)
        if img is None:
            return generate_sample_fundus_image(512, 42), "Uploaded fundus"
        return fit_long_edge(img, WORK_MAX), (upload_name or "Uploaded fundus")

    path, title, seed = SAMPLES[which]  # bench / dense
    img = decode_rgb(path.read_bytes()) if path.exists() else None
    if img is None:
        return generate_sample_fundus_image(512, seed), title
    return fit_long_edge(img, WORK_MAX), title


def generate_sample_fundus_image(size: int = 512, seed: int = 42) -> np.ndarray:
    """Synthetic fundus fallback when a sample file is missing. Returns RGB uint8."""
    rng = np.random.default_rng(seed)
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
            vx += rng.uniform(-0.16, 0.16)
            vy += rng.uniform(-0.16, 0.16)
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


# --------------------------------------------------------------------- stages
def preprocess_with_timing(
    rgb: np.ndarray, clip: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the fused preprocessing (green / CLAHE / Frangi / stack) + elapsed ms."""
    t0 = time.perf_counter()
    green, clahe, frangi, stacked = preprocess_fundus_image(rgb, clip_limit=clip)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return green, clahe, frangi, stacked, elapsed_ms


def infer_with_timing(stacked: np.ndarray) -> tuple[np.ndarray, float]:
    """Run the ONNX UNet under the global compute lock, returning (prob, ms)."""
    with compute_lock:
        t0 = time.perf_counter()
        prob = infer_vessels(stacked)
        return prob, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------- plates
def binary_of(prob: np.ndarray, cutoff: float) -> np.ndarray:
    """Threshold a probability map into a 0/255 uint8 mask."""
    return (prob >= cutoff).astype(np.uint8) * 255


def overlay_plate(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the vessel tint over the original photograph where mask > 0."""
    out = rgb.astype(np.float32)
    hit = mask > 0
    tint = np.array(VESSEL_RGB, dtype=np.float32)
    out[hit] = (1.0 - alpha) * out[hit] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def heat_plate(prob: np.ndarray, palette: str) -> np.ndarray:
    """Apply a matplotlib-derived LUT to a probability map. Returns RGB uint8."""
    lut = MAPS.get(palette, MAPS[DEFAULT_PALETTE])
    idx = np.clip((prob * 255.0).astype(np.uint8), 0, 255)
    return lut[idx]


def tile_placeholder(size: int = 112) -> bytes:
    """Dark dotted glyph tile shown before anything is uploaded."""
    s = size
    img = np.zeros((s, s, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:s, 0:s]
    cx = cy = s / 2
    ring = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[..., 0] = VESSEL_RGB[0]
    img[..., 1] = VESSEL_RGB[1]
    img[..., 2] = VESSEL_RGB[2]
    edge = (ring > s * 0.18) & (ring < s * 0.2)
    bar = (np.abs(yy - cy) < s * 0.014) | (np.abs(xx - cx) < s * 0.014)
    glyph = (edge | bar) & (ring < s * 0.2)
    img[glyph] = (255, 255, 255)
    img[~glyph & (ring < s * 0.2)] = (8, 8, 10)
    img[(ring >= s * 0.2)] = (0, 0, 0)
    dotgrid = ((xx % 12 == 6) & (yy % 12 == 6)) & ~glyph
    img[dotgrid] = (34, 34, 36)
    return encode_png(img)


__all__ = [
    "load_case_image", "generate_sample_fundus_image", "preprocess_with_timing",
    "infer_with_timing", "binary_of", "overlay_plate", "heat_plate",
    "tile_placeholder", "encode_png", "encode_jpeg",
]

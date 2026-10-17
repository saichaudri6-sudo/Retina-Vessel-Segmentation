"""Image IO primitives — OpenCV only (Pillow was dropped in the refactor).

All internal ndarrays are RGB uint8 unless noted. BGR conversions happen
exactly here at the encode/decode boundary, so nothing else has to care.

Note: OpenCV ignores EXIF orientation on decode (Pillow honoured it). Fundus
photographs virtually never carry EXIF rotation, so this is accepted.
"""

from __future__ import annotations

import cv2
import numpy as np


def decode_rgb(data: bytes) -> np.ndarray | None:
    """Decode PNG/JPEG/TIFF bytes to RGB uint8, or None if not an image."""
    if not data:
        return None
    raw = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_long_edge(img: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the long edge is exactly `max_edge` at most (no upscale)."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    target = (max(1, round(w * scale)), max(1, round(h * scale)))  # (w, h)
    return cv2.resize(img, target, interpolation=cv2.INTER_AREA)


def gray_to_rgb(g: np.ndarray) -> np.ndarray:
    """Monochrome (single-channel) array -> (H, W, 3) RGB uint8."""
    arr = np.clip(g, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)


def encode_png(arr: np.ndarray) -> bytes:
    """Encode an RGB (or mono) uint8 array as PNG bytes."""
    return _encode(arr, ".png")


def encode_jpeg(arr: np.ndarray, quality: int = 90) -> bytes:
    """Encode an RGB (or mono) uint8 array as JPEG bytes."""
    return _encode(arr, ".jpg", quality=quality)


def _encode(arr: np.ndarray, ext: str, quality: int | None = None) -> bytes:
    flat = arr.squeeze()
    if flat.ndim == 2:
        flat = gray_to_rgb(flat)
    bgr = cv2.cvtColor(np.ascontiguousarray(flat, np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else None)
    if not ok:
        raise ValueError(f"could not encode {arr.shape} as {ext}")
    return buf.tobytes()

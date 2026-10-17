"""Unit tests: parameter parsing, plates, imaging primitives, session caching."""

import numpy as np

from app.colormaps import MAPS
from app.imaging import decode_rgb, encode_jpeg, encode_png, fit_long_edge, gray_to_rgb
from app.pipeline import binary_of, heat_plate, overlay_plate
from app.web.params import Tune, parse_mode, parse_tune


# --------------------------------------------------------------------- params
def test_parse_tune_defaults():
    assert parse_tune({}) == Tune()


def test_parse_tune_clamps_and_invalids():
    wild = parse_tune({"clip": "99", "cutoff": "-4", "opacity": "0.51", "palette": "nope"})
    assert wild.clip == 10.0
    assert wild.cutoff == 0.10
    assert wild.opacity == 0.51
    assert wild.palette == "magma"


def test_parse_mode_whitelist():
    assert parse_mode("prob") == "prob"
    assert parse_mode("nonsense") == "overlay"
    assert parse_mode(None) == "overlay"


# ---------------------------------------------------------------------- plates
def test_binary_of_thresholds():
    prob = np.array([[0.0, 0.49, 0.5, 1.0]], dtype=np.float32)
    out = binary_of(prob, 0.50)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0, 255, 255]]


def test_overlay_plate_alpha_extremes():
    rgb = np.zeros((4, 4, 3), np.uint8)
    mask = np.ones((4, 4), np.uint8) * 255
    assert overlay_plate(rgb, mask, 1.0)[0, 0].tolist() == [255, 69, 58]
    assert overlay_plate(rgb, mask, 0.0).sum() == 0
    assert overlay_plate(rgb, np.zeros((4, 4), np.uint8), 1.0).sum() == 0


def test_heat_plate_follows_lut():
    prob = np.array([[0.0, 1.0], [0.5, 0.25]], np.float32)
    out = heat_plate(prob, "magma")
    assert out[0, 0].tolist() == MAPS["magma"][0].tolist()
    assert out[0, 1].tolist() == MAPS["magma"][255].tolist()
    assert out[1, 0].tolist() == MAPS["magma"][127].tolist()


# --------------------------------------------------------------------- imaging
def _gradient(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([
        (xx * 255 // max(1, w - 1)), (yy * 255 // max(1, h - 1)), np.full((h, w), 128)
    ], axis=-1).astype(np.uint8)


def test_png_roundtrip():
    original = _gradient(64, 48)
    decoded = decode_rgb(encode_png(original))
    assert decoded.shape == original.shape
    assert decoded.mean() > 0


def test_jpeg_roundtrip():
    original = _gradient(64, 48)
    decoded = decode_rgb(encode_jpeg(original, quality=90))
    assert decoded.shape == original.shape


def test_decode_rejects_garbage():
    assert decode_rgb(b"not an image at all") is None
    assert decode_rgb(b"") is None


def test_fit_long_edge_downscales_only():
    img = np.zeros((100, 250, 3), np.uint8)
    assert fit_long_edge(img, 125).shape == (50, 125, 3)
    assert fit_long_edge(img, 500).shape == (100, 250, 3)  # never upscales


def test_gray_to_rgb():
    g = np.array([[0, 128, 255]], np.uint8)
    rgb = gray_to_rgb(g)
    assert rgb.shape == (1, 3, 3)
    assert rgb[0, 1].tolist() == [128, 128, 128]

"""Static configuration: paths, parameter contracts, tuning ranges.

The web layer reads every bound/step/default from here, so controls in
templates and validation in code can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_DIR = WEB_DIR / "static"

WEIGHTS_PATH = ROOT / "weights" / "vessel_unet.onnx"

SAMPLES: dict[str, tuple[Path, str, int]] = {
    "bench": (ROOT / "samples" / "fundus_sample_1.png", "Benchmark retinal fundus", 42),
    "dense": (ROOT / "samples" / "fundus_sample_2.png", "Dense microvascular retina", 123),
}

# Uploaded images are downscaled so the long edge is at most WORK_MAX px.
# That bounds Frangi + ONNX cost to ~640^2 per user, which is what keeps a
# 512 MB Render host alive.
WORK_MAX = 640
THUMB_SIZE = 112
UPLOAD_LIMIT_BYTES = 18 * 1024 * 1024
MAX_SESSIONS = 6  # each cached session holds ~6 MB of derived arrays

PLATE_TTL_SECONDS = 1800  # Cache-Control for generated media plates
THUMB_TTL_SECONDS = 86400

# ---------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ParamRange:
    lo: float
    hi: float
    default: float
    step: float

    def clamp(self, value: float) -> float:
        return min(self.hi, max(self.lo, value))


CLIP = ParamRange(lo=1.0, hi=10.0, default=3.0, step=0.5)
CUTOFF = ParamRange(lo=0.10, hi=0.90, default=0.50, step=0.05)
OPACITY = ParamRange(lo=0.20, hi=1.00, default=0.70, step=0.05)

# Codes applied to matplotlib-built probability plates (heatPlate).
PALETTES: tuple[str, ...] = ("magma", "inferno", "plasma", "viridis", "jet")
DEFAULT_PALETTE = "magma"

# Eyepiece plate modes. Palette applies only if the plate is a colormap view;
# overlay is fixed vessel-red and mask/orig/green/clahe are monochrome or RGB.
MODES: tuple[tuple[str, str], ...] = (
    ("overlay", "Overlay"),
    ("prob", "Probability"),
    ("mask", "Mask"),
    ("orig", "Original"),
    ("green", "Green"),
    ("clahe", "CLAHE"),
    ("frangi", "Frangi"),
)
PALETTED_MODES: tuple[str, ...] = ("prob", "frangi")

VESSEL_RGB = (255, 69, 58)  # overlay tint; matches the brand accent

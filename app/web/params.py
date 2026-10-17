"""Request parameter parsing + validation.

`parse_tune` clamps every form/query value into the declared ParamRange
bounds and never raises. HTMX does not swap 4xx responses, so a raw 422
would surface as a silent nothing for the user — invalid input is therefore
always reduced to the nearest valid value instead.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from app.colormaps import MAPS
from app.config import CLIP, CUTOFF, DEFAULT_PALETTE, MODES, OPACITY


@dataclasses.dataclass(frozen=True)
class Tune:
    clip: float = CLIP.default
    cutoff: float = CUTOFF.default
    opacity: float = OPACITY.default
    palette: str = DEFAULT_PALETTE


def _num(raw: Any, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_tune(data: Mapping[str, Any]) -> Tune:
    """Read clip/cutoff/opacity/palette from a form or query mapping."""
    return Tune(
        clip=round(CLIP.clamp(_num(data.get("clip"), CLIP.default)), 1),
        cutoff=round(CUTOFF.clamp(_num(data.get("cutoff"), CUTOFF.default)), 2),
        opacity=round(OPACITY.clamp(_num(data.get("opacity"), OPACITY.default)), 2),
        palette=_palette(data.get("palette")),
    )


def _palette(raw: Any) -> str:
    return raw if isinstance(raw, str) and raw in MAPS else DEFAULT_PALETTE


def parse_mode(raw: Any) -> str:
    """Whitelist an eyepiece plate mode key."""
    if isinstance(raw, str) and raw in dict(MODES):
        return raw
    return "overlay"

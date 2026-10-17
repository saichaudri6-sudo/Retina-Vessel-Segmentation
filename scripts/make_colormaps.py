"""Generate runtime colormap LUTs as plain NumPy tables.

Writes colormaps.py containing magma / inferno / plasma / viridis / jet as
uint8 (256, 3) arrays, so the app never imports matplotlib at runtime.

Run:  python scripts/make_colormaps.py
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_PATH = os.path.join("app", "colormaps.py")


def main() -> None:
    entries = []
    for name in ["magma", "inferno", "plasma", "viridis", "jet"]:
        try:
            cmap = matplotlib.colormaps[name]
        except AttributeError:
            cmap = matplotlib.cm.get_cmap(name)
        lut = np.asarray(cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0)
        entries.append(f'    "{name}": np.array({lut.astype(np.uint8).tolist()!r}, dtype=np.uint8),')

    lines = [
        '"""Runtime colormap LUTs (uint8, 256 entries per palette).',
        "",
        "Generated once by scripts/make_colormaps.py from matplotlib's colormap data.",
        "Applied to probability maps so the serving app never imports matplotlib.",
        '"""',
        "",
        "import numpy as np",
        "",
        "MAPS: dict[str, np.ndarray] = {",
        *entries,
        "}",
        "",
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_PATH} with 5 palettes")


if __name__ == "__main__":
    main()

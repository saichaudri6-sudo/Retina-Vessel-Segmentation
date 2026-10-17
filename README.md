# RetinaVessel

Retinal vessel segmentation reading station. Upload (or pick) a fundus
photograph, read the segmentation in a zoomable viewer, calibrate the signal
chain, and export result plates. Pure Python: FastAPI serves Jinja2 HTML
fragments (HTMX), with a single small vanilla-JS file for the viewer controls.

## Stack (8 runtime dependencies)

| Role             | Library                          | Why                                                                     |
| ---------------- | -------------------------------- | ----------------------------------------------------------------------- |
| Web framework    | `fastapi` + `uvicorn[standard]`  | async, typed, the maintained standard for Python model-serving APIs     |
| Inference        | `onnxruntime`                    | trained `VesselUNet` runs from one 17 MB `.onnx`; **no torch at serving** |
| Imaging          | `opencv-python-headless`         | decode/encode/resize/CLAHE; headless wheels need no OS packages          |
| Array math       | `numpy`                          | everything                                                              |
| Frangi filter    | `scikit-image`                   | the vesselness channel is part of the trained model's input contract    |
| Templates        | `jinja2`                         | HTML fragments                                                          |
| Upload parsing   | `python-multipart`               | multipart uploads                                                       |

Pillow is gone (OpenCV covers decode/encode/resampling), and `packages.txt`
(apt extras for the old non-headless OpenCV) is gone — headless wheels are
self-contained. Dev-only extras (`torch`, `matplotlib`, `pytest`, `httpx`,
`ruff`) live in `requirements-dev.txt`.

## Layout

```
main.py               entry shim — keeps `python -m uvicorn main:app` working
app/
  config.py           paths, parameter bounds, palettes, modes (single source of truth)
  imaging.py          OpenCV-only encode/decode boundary (internal RGB everywhere)
  inference.py        lazy, thread-safe ONNX session (arena-free: sized for 512 MB hosts)
  pipeline.py         preprocessing glue, plates, synthetic sample fallback
  sessions.py         per-visitor CaseSession + bounded LRU store (thread-safe)
  colormaps.py        generated matplotlib LUTs (numpy only — no matplotlib at runtime)
  web/                routes.py, params.py (validation), templates/, static/
utils/                trainable sources — untouched (see "pipeline is untouched")
scripts/              train/ calibrate (torch), export ONNX, regenerate colormaps
tests/                pytest: units, session caching, full HTTP smoke tests
samples/ weights/     data
```

## Running

```sh
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8501    # or: python main.py (dev, :8000)
```

Open http://localhost:8501

## Deploying (Render single web service)

- Build: `pip install -r requirements.txt`
- Start: `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
- `/healthz` is the uptime-monitor target (`{"ok": ...}`).
- Optional keep-alive self-ping is enabled only when `RENDER_EXTERNAL_URL`
  (or `APP_URL`) is set; it never runs locally.

## UI

Dark instrument theme; controls left (case tiles, CLAHE / threshold / overlay
sliders), reading surface right: mode chips (Overlay → Frangi), palette chips
in colormap modes, and a real viewer — scroll to zoom, drag to pan,
double-click to fit, number keys 1–7 switch modes. The measurement strip
reports actual numbers only: vascular density, mean confidence inside the
mask, inference/prep timing, and work resolution.

## API

`GET /` · `POST /ui/case` · `POST /ui/upload` · `POST /ui/retune` ·
`POST /ui/view` · `GET /media/view` · `GET /media/thumb` ·
`GET /export/{mask|overlay|prob}` · `GET /healthz`

Plates are ETag-cached (repeated views are a real 304) and carry only the
parameters their mode depends on — unchanged plates keep their URL while
other sliders move, so the browser skips refetching them.

## Pipeline

`utils/preprocessing.py` (green channel → CLAHE → multiscale Frangi) and
`utils/architecture.py` (U-Net) are the original trainable sources — the
serving app reuses them verbatim (the preprocessing input signature is
NumPy-only now, since Pillow was dropped; the numerics are unchanged). The
trained weights ship as
`weights/vessel_unet.onnx`, verified against the torch model by
`scripts/export_onnx.py` (max drift ~1e-4).

## Development

```sh
pip install -r requirements-dev.txt
python main.py                     # run with reload on :8000
pytest                             # tests (ONNX session + HTTP smoke)
ruff check .                       # lint
python scripts/export_onnx.py      # rebuild .onnx from .pth + parity check
python scripts/make_colormaps.py   # regenerate app/colormaps.py from matplotlib
```

Note: OpenCV ignores EXIF orientation on decode (Pillow honoured it). Fundus
photographs rarely carry EXIF rotation; this is accepted behavior.

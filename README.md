# RetinaVessel

Retinal vessel segmentation reading station. Upload (or pick) a fundus
photograph, read the segmentation through a circular eyepiece, calibrate the
signal chain, and export result plates.

## Stack

- **FastAPI + Jinja2 + HTMX** — one Python process, HTML fragments over HTTP
  (no WebSockets, no SPA framework, no client build step).
- **ONNX Runtime** — the trained `VesselUNet` is exported once to
  `weights/vessel_unet.onnx` (~17 MB) and served with onnxruntime. The serving
  process never imports PyTorch — numerically verified against the torch model
  (max drift ~1e-4) by `scripts/export_onnx.py`.
- **The segmentation pipeline is untouched** — `utils/preprocessing.py`
  (green channel, CLAHE, multiscale Frangi) and `utils/architecture.py`
  (U-Net) are the original trainable sources.

## Running

```sh
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8501
```

Open http://localhost:8501

## Training / exporting (dev only, needs requirements-dev.txt)

```sh
pip install -r requirements-dev.txt
python scripts/export_onnx.py       # rebuild .onnx from weights + parity check
python scripts/make_colormaps.py    # regenerate runtime LUTs from matplotlib
```

## Deployed

Single web service (Render etc.): build `pip install -r requirements.txt`,
start `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`.
`/healthz` is the uptime monitor target.

"""RetinaVessel serving package.

Everything under `app/` is runtime-only: ONNX Runtime inference, OpenCV
imaging, HTML fragment serving. Importing `app` never pulls in torch —
torch lives solely in `utils/` + `scripts/` for training/export.
"""

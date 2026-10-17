"""RetinaVessel entrypoint.

Production (Render etc.):
    python -m uvicorn main:app --host 0.0.0.0 --port $PORT

Local dev (reload):
    python main.py
"""

from __future__ import annotations

from app.web.app import create_app

app = create_app()


if __name__ == "__main__":  # local dev convenience
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.keepalive import start_if_configured
from app.web.routes import router


def create_app() -> FastAPI:
    web = FastAPI(title="RetinaVessel", docs_url=None, redoc_url=None, openapi_url=None)
    web.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    web.include_router(router)
    start_if_configured()
    return web

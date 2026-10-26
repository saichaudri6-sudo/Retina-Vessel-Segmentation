"""All HTTP routes: pages, HTMX fragments, media plates, exports, health.

Endpoint paths are unchanged from the previous version, so existing links and
uptime monitors keep working:

    GET  /             full page (tray + tune form + bay)
    POST /ui/case      switch to a sample case      -> bay + tray (OOB)
    POST /ui/upload    multipart image upload       -> bay + tray (OOB)
    POST /ui/retune    slider / palette change      -> bay
    POST /ui/view      plate mode change            -> viewer only
    GET  /media/view   plate pixels (png / jpeg)
    GET  /media/thumb  tray thumbnails
    GET  /export/{kind} mask / overlay / probability downloads
    GET  /favicon.ico  static
    GET  /healthz      keep-alive / monitoring target
"""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

import app.sessions as sessions
from app import pipeline as P
from app.config import (
    CLIP,
    CUTOFF,
    MODES,
    OPACITY,
    PALETTED_MODES,
    PALETTES,
    PLATE_TTL_SECONDS,
    STATIC_DIR,
    THUMB_SIZE,
    THUMB_TTL_SECONDS,
    UPLOAD_LIMIT_BYTES,
    WEB_DIR,
    WORK_MAX,
)
from app.imaging import decode_rgb, encode_jpeg, encode_png, fit_long_edge, gray_to_rgb
from app.web.params import Tune, parse_mode, parse_tune

router = APIRouter()
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
templates.env.filters["g"] = lambda v: f"{v:g}"  # 1.0 → "1", 0.10 → "0.1" for HTML attrs

SESSION_COOKIE = "rv_sid"
CASE_KEYS = ("bench", "dense")
EXPORT_KINDS = ("mask", "overlay", "prob")


# ---------------------------------------------------------------------- helpers
def _session_for(request: Request) -> sessions.CaseSession:
    return sessions.get_session(request.cookies.get(SESSION_COOKIE))


def _render(request: Request, name: str, ctx: dict[str, Any]) -> str:
    """Render a template to an HTML string (responses often combine fragments)."""
    return templates.TemplateResponse(request, name, ctx).body.decode("utf-8")


def _error(message: str) -> str:
    """Inline alert card; returned with HTTP 200 so HTMX actually swaps it."""
    return templates.get_template("err.html").render({"message": message})


def _alert_oob(message: str) -> str:
    """Alert card wrapped as a complete #upalert node (out-of-band slot)."""
    return templates.get_template("upalert.html").render({"message": message})


def _file_reject_reason(data: bytes) -> str:
    """Turn undecodable bytes into a message naming the actual culprit."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand.startswith((b"hei", b"mif", b"heif", b"hvc")):
            return ("That upload is an iPhone HEIC photo — browsers can't read it directly. "
                    "On the iPhone set Settings → Camera → Formats → Most Compatible, "
                    "or convert the photo to JPG/PNG, then upload again.")
        if brand.startswith(b"avi"):
            return ("That upload is AVIF video-codec still — not readable here. "
                    "Re-save it as JPG or PNG, then upload again.")
        return ("That container could not be opened. Re-save the image as "
                "PNG, JPG or TIFF, then upload again.")
    if data[:3] == b"GIF":
        return ("That upload is an animated GIF — mask plates need a still scan. "
                "Re-save it as PNG or JPG, then upload again.")
    if data[:5] == b"%PDF-":
        return "That upload is a PDF document — export the image itself as JPG or PNG, then try again."
    return ("That file could not be opened as a scan. Re-save it as PNG, JPG or TIFF "
            "(HEIC, AVIF and animated GIF are not readable yet) and upload again.")


def _html(request: Request, payload: str, status_code: int = 200) -> Response:
    """HTML fragment response that carries / refreshes the session cookie."""
    resp = HTMLResponse(payload, status_code=status_code)
    resp.set_cookie(
        SESSION_COOKIE, _session_for(request).sid,
        max_age=60 * 60 * 24 * 30, samesite="lax", httponly=True,
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _src_label(sess: sessions.CaseSession) -> str:
    return sess.upload_name if sess.case == "upload" else sess.title


def _tray_ctx(sess: sessions.CaseSession) -> dict[str, Any]:
    return {
        "case": sess.case,
        "src": _src_label(sess),
        "has_upload": bool(sess.upload_bytes),
    }


def _media_url(mode: str, tune: Tune) -> str:
    """Build an /media/view URL carrying only the params that plate depends on.

    Fewer params means unchanged plates keep their URL while other sliders
    move — the browser then skips refetching them.
    """
    clip = f"clip={tune.clip}"
    if mode in ("orig", "green"):
        return f"/media/view?mode={mode}"
    if mode == "clahe":
        return f"/media/view?mode=clahe&{clip}"
    if mode in ("frangi", "prob"):
        return f"/media/view?mode={mode}&{clip}&palette={tune.palette}"
    if mode == "mask":
        return f"/media/view?mode=mask&{clip}&cutoff={tune.cutoff}"
    return f"/media/view?mode=overlay&{clip}&cutoff={tune.cutoff}&opacity={tune.opacity}"


def _bay_ctx(sess: sessions.CaseSession, tune: Tune) -> dict[str, Any]:
    """Everything the bay (and viewer) templates need, computed once."""
    _green, _clahe, _frangi, _stacked, prep_ms, prep_cached = sess.pre(tune.clip)
    prob, infer_ms, prob_cached = sess.prob(tune.clip)
    binary = P.binary_of(prob, tune.cutoff)
    hit = int(np.count_nonzero(binary))
    density = hit / float(sess.w * sess.h) * 100.0
    conf = float(np.mean(prob[binary > 0])) if hit else 0.0

    plate = tune.palette if sess.mode in PALETTED_MODES else None
    q = f"clip={tune.clip}&cutoff={tune.cutoff}&opacity={tune.opacity}&palette={tune.palette}"

    return {
        "title": sess.title, "w": sess.w, "h": sess.h,
        "mode": sess.mode, "modes": list(MODES),
        "palette": plate, "palettes": list(PALETTES), "paletted": PALETTED_MODES,
        "clip": tune.clip, "cutoff": tune.cutoff, "opacity": tune.opacity,
        "density": f"{density:.2f}", "conf": f"{conf * 100:.1f}", "count": f"{hit:,}",
        "prep_ms": int(prep_ms), "infer_ms": int(infer_ms),
        "cached": bool(prep_cached and prob_cached),
        "u": {key: _media_url(key, tune) for key, _label in MODES},
        "x": {k: f"/export/{k}?{q}" for k in EXPORT_KINDS},
    }


def _bay_response(request: Request, sess: sessions.CaseSession, tune: Tune,
                  with_tray: bool = False, clear_alert: bool = False) -> Response:
    """Fragment response: bay, optionally plus a swap-out-of-band tray."""
    bay_html = _render(request, "bay.html", _bay_ctx(sess, tune))
    if with_tray:
        tray_html = _render(request, "tray.html", {**_tray_ctx(sess), "oob": True})
        bay_html += "\n" + tray_html
    if clear_alert:
        bay_html += "\n" + _alert_oob("")  # empty the upload alert slot
    return _html(request, bay_html)


def _finish(request: Request, data: bytes, media_type: str) -> Response:
    """Serve plate bytes with a stable ETag — repeated views are a real 304."""
    etag = '"' + hashlib.sha1(data).hexdigest()[:16] + '"'
    cache = f"public, max-age={PLATE_TTL_SECONDS}"
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache})
    return Response(data, media_type=media_type, headers={"Cache-Control": cache, "ETag": etag})


# ------------------------------------------------------------------------ pages
@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    sess = _session_for(request)
    tune = Tune()  # a fresh page always loads at defaults
    ctx = {
        "bay_html": _render(request, "bay.html", _bay_ctx(sess, tune)),
        "tray_html": _render(request, "tray.html", {**_tray_ctx(sess), "oob": False}),
        "clip": CLIP, "cutoff": CUTOFF, "opacity": OPACITY,
    }
    return _html(request, _render(request, "index.html", ctx))


# ------------------------------------------------------------------ fragments
@router.post("/ui/case")
async def ui_case(request: Request, case: str = Form("bench")):
    tune = parse_tune(await request.form())
    sess = _session_for(request)
    sess.set_case(case if case in CASE_KEYS else "bench")
    return _bay_response(request, sess, tune, with_tray=True)


@router.post("/ui/upload")
async def ui_upload(request: Request, file: UploadFile | None = File(None)):
    tune = parse_tune(await request.form())
    sess = _session_for(request)
    data = await file.read() if file else b""
    if not data:
        message = "Choose an image file first — tap the + tile, then pick a PNG, JPG or TIFF."
        return _html(request, _render(request, "bay.html", _bay_ctx(sess, tune)) + "\n" + _alert_oob(message))
    if len(data) > UPLOAD_LIMIT_BYTES:
        message = (f"That file is over {UPLOAD_LIMIT_BYTES // (1024 * 1024)} MB — "
                   "export a smaller copy and retry. The previous scan is still shown.")
        return _html(request, _render(request, "bay.html", _bay_ctx(sess, tune)) + "\n" + _alert_oob(message))
    if decode_rgb(data) is None:
        message = _file_reject_reason(data)
        return _html(request, _render(request, "bay.html", _bay_ctx(sess, tune)) + "\n" + _alert_oob(message))
    sess.set_upload(data, (file.filename if file else "") or "Uploaded fundus")
    return _bay_response(request, sess, tune, with_tray=True, clear_alert=True)


@router.post("/ui/retune")
async def ui_retune(request: Request):
    tune = parse_tune(await request.form())
    sess = _session_for(request)
    return _bay_response(request, sess, tune)


@router.post("/ui/view")
async def ui_view(request: Request):
    form = await request.form()
    tune = parse_tune(form)
    sess = _session_for(request)
    sess.mode = parse_mode(form.get("mode"))
    return _html(request, _render(request, "viewer.html", _bay_ctx(sess, tune)))


# ------------------------------------------------------------------ media plates
@router.get("/media/view")
def media_view(request: Request, mode: str = "overlay"):
    sess = _session_for(request)
    tune = parse_tune(request.query_params)
    if mode not in dict(MODES):
        return Response("unknown plate mode", status_code=400)

    green, clahe, frangi = sess.pre(tune.clip)[:3]
    if mode == "orig":
        return _finish(request, encode_jpeg(sess.rgb, quality=90), "image/jpeg")
    if mode == "green":
        return _finish(request, encode_jpeg(gray_to_rgb(green), quality=88), "image/jpeg")
    if mode == "clahe":
        return _finish(request, encode_jpeg(gray_to_rgb(clahe), quality=88), "image/jpeg")
    if mode == "frangi":
        return _finish(request, encode_png(P.heat_plate(frangi, tune.palette)), "image/png")

    prob, _ms, _cached = sess.prob(tune.clip)
    if mode == "prob":
        return _finish(request, encode_png(P.heat_plate(prob, tune.palette)), "image/png")
    binary = P.binary_of(prob, tune.cutoff)
    if mode == "mask":
        return _finish(request, encode_png(gray_to_rgb(binary)), "image/png")
    return _finish(request, encode_png(P.overlay_plate(sess.rgb, binary, tune.opacity)), "image/png")


@lru_cache(maxsize=4)
def _sample_thumb(which: str) -> bytes:
    img, _title = P.load_case_image(which, None, "")
    return encode_png(fit_long_edge(img, THUMB_SIZE))


@router.get("/media/thumb")
def media_thumb(request: Request, which: str = "bench"):
    sess = _session_for(request)
    if which == "upload":
        if sess.thumb_bytes:
            return Response(sess.thumb_bytes, media_type="image/png",
                            headers={"Cache-Control": "no-store"})
        return Response(P.tile_placeholder(THUMB_SIZE), media_type="image/png",
                        headers={"Cache-Control": "max-age=600"})
    if which not in CASE_KEYS:
        which = "bench"
    return Response(_sample_thumb(which), media_type="image/png",
                    headers={"Cache-Control": f"public, max-age={THUMB_TTL_SECONDS}"})


# ----------------------------------------------------------------------- export
@router.get("/export/{kind}")
def export_kind(request: Request, kind: str):
    sess = _session_for(request)
    tune = parse_tune(request.query_params)
    if kind not in EXPORT_KINDS:
        return HTMLResponse(_error("Unknown export kind."), status_code=404)

    prob, _ms, _cached = sess.prob(tune.clip)
    binary = P.binary_of(prob, tune.cutoff)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if kind == "mask":
        data = encode_png(gray_to_rgb(binary))
        name = f"retinal_vessel_mask_{stamp}.png"
    elif kind == "overlay":
        data = encode_png(P.overlay_plate(sess.rgb, binary, tune.opacity))
        name = f"retinal_vessel_overlay_{stamp}.png"
    else:
        data = encode_png(P.heat_plate(prob, tune.palette))
        name = f"retinal_vessel_probability_{stamp}.png"

    return Response(data, media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="{name}"',
        "Cache-Control": "no-store",
    })


# ------------------------------------------------------------------------ misc
@router.get("/favicon.ico")
def favicon():
    path = STATIC_DIR / "favicon.png"
    if path.exists():
        return FileResponse(path, media_type="image/png")
    return Response(status_code=404)


@router.get("/healthz")
def healthz():
    from app.inference import onnx_session  # late import keeps boot fast
    mem: dict[str, int] = {}
    try:
        with open("/proc/self/status", encoding="ascii") as fh:  # pragma: no cover — Windows dev
            for line in fh:
                if line.startswith(("VmRSS", "VmHWM")):
                    key, val = line.split(":", 1)
                    mem[key] = int(val.split()[0]) // 1024  # MB
    except OSError:
        pass  # not Linux — skip memory reporting
    ok = True
    try:
        from app.config import WEIGHTS_PATH
        ok = WEIGHTS_PATH.exists() and bool(onnx_session())
    except Exception:
        ok = False
    return JSONResponse({"ok": ok, "engine": "onnxruntime", "work_max": WORK_MAX,
                         "sessions": sessions.session_count(), "mem_mb": mem})

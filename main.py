"""RetinaVessel — FastAPI serving app (ONNX Runtime; no torch at runtime).

Endpoints
    GET  /             full instrument page
    POST /ui/case      switch to a sample case            bay + tray (OOB)
    POST /ui/upload    multipart image upload             bay + tray (OOB)
    POST /ui/retune    slider / palette / viewport change bay
    POST /ui/view      eyepiece mode change               eyepiece only
    GET  /media/view   plate pixels (png / jpeg)
    GET  /media/thumb  tray thumbnails
    GET  /export/      mask / overlay / probability downloads
    GET  /healthz      keep-alive target for UptimeRobot
"""

from __future__ import annotations

import io
import os
import threading
import time
import urllib.request
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(ROOT, "templates"))

app = FastAPI(title="RetinaVessel", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")

CLIP_LO, CLIP_HI, CLIP_DEF = 1.0, 10.0, 3.0
CUT_LO, CUT_HI, CUT_DEF = 0.10, 0.90, 0.50
OP_LO, OP_HI, OP_DEF = 0.20, 1.00, 0.70
VIEW_LO, VIEW_HI, VIEW_DEF = 280, 860, 560
PALETTES = ("magma", "inferno", "plasma", "viridis", "jet")


# --------------------------------------------------------------------- helpers
def _session_for(request: Request) -> P.CaseSession:
    return P.get_session(request.cookies.get("rv_sid"))


def _form_params(form_raw: Any) -> Dict[str, Any]:
    form = getattr(form_raw, "_data", form_raw)
    get = lambda key, default=None: (form.get(key) if hasattr(form, "get") else default)
    clip = get("clip", CLIP_DEF)
    cutoff = get("cutoff", CUT_DEF)
    opacity = get("opacity", OP_DEF)
    palette = get("palette", "magma")
    view = get("view", VIEW_DEF)
    try:
        return {
            "clip": round(P.clamp(clip, CLIP_LO, CLIP_HI, CLIP_DEF), 1),
            "cutoff": round(P.clamp(cutoff, CUT_LO, CUT_HI, CUT_DEF), 2),
            "opacity": round(P.clamp(opacity, OP_LO, OP_HI, OP_DEF), 2),
            "palette": P.clamp_palette(palette),
            "view": int(P.clamp(view, VIEW_LO, VIEW_HI, VIEW_DEF, 10)),
        }
    except Exception:
        return {"clip": CLIP_DEF, "cutoff": CUT_DEF, "opacity": OP_DEF, "palette": "magma", "view": VIEW_DEF}


def _with_cookie(request: Request, payload: str, status_code: int = 200) -> Response:
    """Build an HTML response carrying/refreshing the session cookie."""
    sess = _session_for(request)
    resp = HTMLResponse(payload, status_code=status_code)
    resp.set_cookie("rv_sid", sess.sid, max_age=60 * 60 * 24 * 30, samesite="lax", httponly=True)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _media_url(mode: str, pr: Optional[dict] = None, palette: str = "magma") -> str:
    if mode in ("orig", "green"):
        return f"/media/view?mode={mode}"
    clip = pr["clip"] if pr else 0
    if mode in ("clahe", "frangi"):
        return f"/media/view?mode={mode}&clip={clip}"
    cutoff = pr["cutoff"] if pr else 0
    opacity = pr["opacity"] if pr else 0
    return f"/media/view?mode={mode}&clip={clip}&cutoff={cutoff}&opacity={opacity}&palette={palette}"


def _bay_ctx(sess: P.CaseSession, pr: Dict[str, Any]) -> Dict[str, Any]:
    clip, cutoff, opacity, palette, view = pr["clip"], pr["cutoff"], pr["opacity"], pr["palette"], pr["view"]
    _green, _clahe, _frangi, _stacked, prep_ms, pre_cached = sess.Pre(clip)
    prob, infer_ms, prob_cached = sess.Prob(clip)
    binary = P.binary_of(prob, cutoff)
    h, w = sess.h, sess.w

    hit = int(np.count_nonzero(binary))
    density = hit / (h * w) * 100.0
    conf = float(np.mean(prob[binary > 0])) if hit else 0.0
    tpos = min(96.0, max(4.0, (cutoff - CUT_LO) / (CUT_HI - CUT_LO) * 100.0))

    modes = [
        ("overlay", "Overlay"), ("prob", "Probability"), ("mask", "Mask"), ("orig", "Original"),
        ("green", "Green"), ("clahe", "CLAHE"), ("frangi", "Frangi"),
    ]
    urls = {
        "overlay": _media_url("overlay", pr, palette),
        "prob": _media_url("prob", pr, palette),
        "mask": _media_url("mask", pr, palette),
        "orig": _media_url("orig"),
        "green": _media_url("green"),
        "clahe": _media_url("clahe", {"clip": clip}),
        "frangi": _media_url("frangi", {"clip": clip}),
    }
    exports = {
        "mask": f"/export/mask?clip={clip}&cutoff={cutoff}&opacity={opacity}&palette={palette}",
        "overlay": f"/export/overlay?clip={clip}&cutoff={cutoff}&opacity={opacity}&palette={palette}",
        "prob": f"/export/prob?clip={clip}&cutoff={cutoff}&opacity={opacity}&palette={palette}",
    }
    return {
        "title": sess.title, "h": h, "w": w,
        "mode": getattr(sess, "mode", "overlay"),
        "clip": clip, "clip_s": f"{clip:.1f}",
        "cutoff": cutoff, "cutoff_s": f"{cutoff:.2f}",
        "opacity": opacity, "opac_s": f"{int(opacity * 100)}",
        "palette": palette, "view": view,
        "density": f"{density:.2f}",
        "fill": f"{min(100.0, density * 8.0):.0f}",
        "lit": int(round(conf * 12)),
        "conf": f"{conf * 100:.1f}", "count": f"{hit:,}",
        "prep_ms": int(prep_ms), "infer_ms": int(infer_ms),
        "fresh": (not pre_cached) or (not prob_cached),
        "tpos": f"{tpos:.1f}",
        "modes": modes, "u": urls,
        "x": exports,
        "case": sess.case, "upload": sess.upload_name,
    }


def _render_bay(request: Request, sess: P.CaseSession, pr: Dict[str, Any],
                with_tray: bool = False) -> Response:
    ctx = _bay_ctx(sess, pr)
    bay_html = templates.TemplateResponse(request, "bay.html", ctx).body.decode("utf-8")
    if with_tray:
        tray_html = templates.TemplateResponse(request, "tray.html", {
            "case": sess.case, "upload": sess.upload_name,
        }).body.decode("utf-8")
        oob = tray_html.replace("<div id=\"tray-card\"", "<div id=\"tray-card\" hx-swap-oob=\"true\"", 1)
        return _with_cookie(request, bay_html + "\n" + oob)
    return _with_cookie(request, bay_html)


# ------------------------------------------------------------------ page & UI
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    sess = _session_for(request)
    pr_default = {"clip": CLIP_DEF, "cutoff": CUT_DEF, "opacity": OP_DEF,
                  "palette": "magma", "view": VIEW_DEF}
    form = None
    ctx = _bay_ctx(sess, pr_default)
    tray_html = templates.TemplateResponse(request, "tray.html", {
        "case": sess.case, "upload": sess.upload_name,
    }).body.decode("utf-8")
    bay_html = templates.TemplateResponse(request, "bay.html", ctx).body.decode("utf-8")
    html = templates.TemplateResponse(request, "index.html", {
        "tray_html": tray_html, "bay_html": bay_html,
        "clip_def": CLIP_DEF, "cutoff_def": CUT_DEF,
        "op_def": OP_DEF, "view_def": VIEW_DEF,
        "palette_def": "magma", "palettes": PALETTES,
    }).body.decode("utf-8")
    return _with_cookie(request, html)


@app.post("/ui/case")
async def ui_case(request: Request, case: str = Form("bench")):
    form = await request.form()
    pr = _form_params(form)
    sess = _session_for(request)
    sess.set_case(case if case in ("bench", "dense") else "bench")
    return _render_bay(request, sess, pr, with_tray=True)


@app.post("/ui/upload")
async def ui_upload(request: Request, file: Optional[UploadFile] = File(None)):
    form = await request.form()
    pr = _form_params(form)
    sess = _session_for(request)
    data = await (file.read() if file else _none_bytes())
    if not data:
        return _with_cookie(request, "<div class='err'>Choose an image file first.</div>", 400)
    if len(data) > 18 * 1024 * 1024:
        return _with_cookie(request,
                            "<div class='err'>That file is over 18 MB. Export a smaller copy and retry.</div>", 413)
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
    except Exception:
        return _with_cookie(request,
                            "<div class='err'>That file could not be opened. PNG, JPG or TIFF only.</div>", 415)
    sess.set_upload(data, (file.filename if file else "") or "Uploaded fundus")
    return _render_bay(request, sess, pr, with_tray=True)


async def _none_bytes() -> bytes:
    return b""


@app.post("/ui/retune")
async def ui_retune(request: Request):
    form = await request.form()
    pr = _form_params(form)
    sess = _session_for(request)
    return _render_bay(request, sess, pr)


@app.post("/ui/view")
async def ui_view(request: Request, mode: str = Form("overlay")):
    form = await request.form()
    pr = _form_params(form)
    sess = _session_for(request)
    mode = mode if mode in P.MODES else "overlay"
    sess.mode = mode
    ctx = {
        "mode": mode, "view": pr["view"], "h": sess.h, "w": sess.w,
        "palette": pr["palette"],
        "title": sess.title,
        "u": {
            "overlay": _media_url("overlay", pr, pr["palette"]),
            "prob": _media_url("prob", pr, pr["palette"]),
            "mask": _media_url("mask", pr, pr["palette"]),
            "orig": _media_url("orig"),
            "green": _media_url("green"),
            "clahe": _media_url("clahe", {"clip": pr["clip"]}),
            "frangi": _media_url("frangi", {"clip": pr["clip"]}),
        },
        "modes": [("overlay", "Overlay"), ("prob", "Probability"), ("mask", "Mask"), ("orig", "Original"),
                  ("green", "Green"), ("clahe", "CLAHE"), ("frangi", "Frangi")],
        "title": sess.title,
    }
    html = templates.TemplateResponse(request, "view.html", ctx).body.decode("utf-8")
    return _with_cookie(request, html)


# ---------------------------------------------------------------- media plates
def _ensure_session_or_default(request: Request) -> P.CaseSession:
    return _session_for(request)


@app.get("/media/view")
def media_view(request: Request, mode: str = "overlay", clip: str = "", cutoff: str = "",
               opacity: str = "", palette: str = ""):
    sess = _ensure_session_or_default(request)
    try:
        clipf = P.clamp(clip, CLIP_LO, CLIP_HI, CLIP_DEF)
        cutoff = P.clamp(cutoff, CUT_LO, CUT_HI, CUT_DEF)
        opacity = P.clamp(opacity, OP_LO, OP_HI, OP_DEF)
        palette = P.clamp_palette(palette)
    except Exception:
        clipf, cutoff, opacity, palette = CLIP_DEF, CUT_DEF, OP_DEF, "magma"

    green, clahe, frangi = sess.Pre(clipf)[:3]
    cache_head = {"Cache-Control": "public, max-age=1800"}

    if mode == "orig":
        return Response(P.encode_jpeg(sess.rgb, 90), media_type="image/jpeg", headers=cache_head)
    if mode == "green":
        return Response(P.encode_jpeg(P.gray_to_rgb(green), 88), media_type="image/jpeg", headers=cache_head)
    if mode == "clahe":
        return Response(P.encode_jpeg(P.gray_to_rgb(clahe), 88), media_type="image/jpeg", headers=cache_head)
    if mode == "frangi":
        return Response(P.encode_png(P.heat_plate(frangi, palette)), media_type="image/png", headers=cache_head)

    prob, _ms, _cached = sess.Prob(clipf)
    if mode == "prob":
        out = P.heat_plate(prob, palette)
    elif mode == "mask":
        out = P.binary_of(prob, cutoff)
        return Response(P.encode_png(P.gray_to_rgb(out)), media_type="image/png",
                        headers=cache_head)
    elif mode == "overlay":
        binary = P.binary_of(prob, cutoff)
        out = P.overlay_plate(sess.rgb, binary, opacity)
    else:
        out = P.overlay_plate(sess.rgb, P.binary_of(prob, cutoff), opacity)
    return Response(P.encode_png(out), media_type="image/png", headers=cache_head)


@app.get("/media/thumb")
def media_thumb(request: Request, which: str = "bench"):
    sess = _ensure_session_or_default(request)
    size = 112
    if which == "dense":
        img, _ = P.load_case_image("dense", None)
    elif which == "bench":
        img, _ = P.load_case_image("bench", None)
    elif which == "upload":
        if sess.upload_bytes:
            img, _ = P.load_case_image("upload", sess.upload_bytes, sess.upload_name)
        else:
            avail = P.tile_placeholder(112)
            return Response(avail, media_type="image/png", headers={"Cache-Control": "no-store"})
    else:
        img, _ = P.load_case_image("bench", None)
    im = Image.fromarray(img)
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/export/{kind}")
def export_kind(request: Request, kind: str, clip: str = "", cutoff: str = "",
                opacity: str = "", palette: str = ""):
    sess = _ensure_session_or_default(request)
    clipf = P.clamp(clip, CLIP_LO, CLIP_HI, CLIP_DEF)
    cutoff = round(P.clamp(cutoff, CUT_LO, CUT_HI, CUT_DEF), 2)
    opacity = P.clamp(opacity, OP_LO, OP_HI, OP_DEF)
    palette = P.clamp_palette(palette)

    prob, _ms, _c = sess.Prob(clipf)
    binary = P.binary_of(prob, cutoff)
    stamp = time.strftime("%H%M%S")

    if kind == "mask":
        data = P.encode_png(P.gray_to_rgb(binary))
        name = f"retinal_vessel_mask_{stamp}.png"
    elif kind == "overlay":
        data = P.encode_png(P.overlay_plate(sess.rgb, binary, opacity))
        name = f"retinal_vessel_overlay_{stamp}.png"
    elif kind == "prob":
        data = P.encode_png(P.heat_plate(prob, palette))
        name = f"retinal_vessel_probability_{stamp}.png"
    else:
        return HTMLResponse("<div class='err'>Unknown export kind.</div>", 404)
    return Response(data, media_type="image/png",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/favicon.ico")
def favicon():
    path = os.path.join(ROOT, "assets", "favicon.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return Response(status_code=404)


@app.get("/healthz")
def healthz():
    mem = {}
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(("VmRSS", "VmHWM")):
                    key, val = line.split(":", 1)
                    mem[key] = int(val.split()[0]) // 1024  # MB
    except OSError:
        pass
    ok = os.path.exists(P.WEIGHTS_PATH) and P.ort_session() is not None
    return JSONResponse({"ok": bool(ok), "engine": "onnxruntime", "work_max": P.WORK_MAX, "mem_mb": mem})


# ------------------------------------------------------------------ keep alive
def _start_keep_warm() -> None:
    base = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL") or ""
    if not base:
        return

    def _ping() -> None:
        while True:
            try:
                urllib.request.urlopen(f"{base}/healthz", timeout=15).read(64)
            except Exception:
                pass
            time.sleep(300)

    threading.Thread(target=_ping, daemon=True, name="keep-warm").start()


_start_keep_warm()

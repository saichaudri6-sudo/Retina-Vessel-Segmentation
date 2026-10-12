"""Retinal vessel segmentation workstation."""

import base64
import io
import os
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from utils.preprocessing import preprocess_fundus_image
from utils.architecture import VesselUNet, predict_vessels


st.set_page_config(
    page_title="RetinaVessel",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
html, body, .stApp {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", Inter, "Segoe UI", sans-serif;
  background: #000; color: #F5F5F7;
  background-image:
    radial-gradient(ellipse 70% 42% at 50% -60px, rgba(255,255,255,0.075), transparent 70%),
    radial-gradient(rgba(255,255,255,0.075) 1px, transparent 1.15px);
  background-size: auto, 22px 22px;
}
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }
.block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 4rem; }
div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
  background: rgba(255,255,255,0.028);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px; padding: 20px 22px;
}
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stMetricValue"] { font-size: 30px; font-variant-numeric: tabular-nums; }
.rv-nav {
  display: flex; align-items: center; justify-content: space-between;
  background: transparent; border: none;
  padding: 6px 4px; margin-bottom: 22px;
}
.rv-brand { display: flex; align-items: center; gap: 14px; }
.rv-mark { width: 48px; height: 48px; border-radius: 50%; overflow: hidden; background: #000;
  border: none; flex: none;
  box-shadow: 0 0 26px rgba(255,255,255,0.22), 0 0 4px rgba(255,255,255,0.35); }
.rv-mark img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rv-plus { aspect-ratio: 1 / 1; border: 1.5px dotted rgba(255,255,255,0.35); border-radius: 14px;
  display: flex; align-items: center; justify-content: center; font-size: 30px; font-weight: 300; color: #A1A1A6; }
.rv-dotrule { height: 12px; margin: 8px 2px 0 2px; opacity: 0.55;
  background-image: radial-gradient(rgba(255,255,255,0.4) 1.2px, transparent 1.3px);
  background-size: 12px 12px; background-repeat: repeat-x; background-position: center; }
div.stButton > button[kind="primary"] { background: #F5F5F7 !important; color: #000 !important;
  border-color: #F5F5F7 !important; font-weight: 600 !important;
  outline: 2px dotted rgba(255,255,255,0.75) !important; outline-offset: 3px; }
div.stButton > button[kind="secondary"] { background: transparent !important; color: #F5F5F7 !important;
  border: 1px solid rgba(255,255,255,0.16) !important; }
.rv-name { font-size: 17px; font-weight: 650; letter-spacing: -0.01em; }
.rv-sub { font-size: 12.5px; color: #A1A1A6; }
.rv-pill { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: #D4D4D8;
  border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; padding: 7px 14px; white-space: nowrap; }
.rv-dot { width: 7px; height: 7px; border-radius: 999px; background: #F5F5F7; flex: none; }
.rv-title { font-size: 30px; font-weight: 700; letter-spacing: -0.025em; margin: 0; }
.rv-meta { font-size: 13px; color: #A1A1A6; margin-top: 6px; font-variant-numeric: tabular-nums; }
.rv-h { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; margin: 30px 0 2px 0; }
.rv-s { font-size: 13.5px; color: #A1A1A6; margin-bottom: 12px; }
.rv-scope { background: #000; border: 1px solid rgba(255,255,255,0.10); border-radius: 20px; padding: 16px 16px 12px 16px; }
.rv-scope-top { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #E5E5EA;
  padding: 0 6px 12px 6px; font-variant-numeric: tabular-nums; }
.rv-scope-top span:last-child { color: #A1A1A6; font-weight: 500; }
.rv-ring-outer { width: fit-content; margin: 4px auto 2px auto; padding: 16px;
  border: 1.5px dotted rgba(255,255,255,0.35); border-radius: 50%;
  box-shadow: 0 0 60px rgba(255,255,255,0.06); }
.rv-ring-inner { padding: 9px; border: 1px solid rgba(255,255,255,0.14); border-radius: 50%; }
.rv-ring-inner img { border-radius: 50%; display: block; }
.rv-scope-meta { display: flex; justify-content: space-between; font-size: 12px; color: #A1A1A6;
  padding: 12px 6px 2px 6px; font-variant-numeric: tabular-nums; }
.rv-read { display: flex; }
.rv-cell { flex: 1; padding: 2px 22px; }
.rv-cell:first-child { padding-left: 2px; }
.rv-cell + .rv-cell { border-left: 2px dotted rgba(255,255,255,0.18); }
.rv-lab { font-size: 12.5px; color: #A1A1A6; font-weight: 500; }
.rv-num { font-size: 32px; font-weight: 700; letter-spacing: -0.025em; font-variant-numeric: tabular-nums; margin: 2px 0 10px 0; }
.rv-dots-track { height: 10px; border-radius: 999px; overflow: hidden;
  background-image: radial-gradient(rgba(255,255,255,0.20) 1.6px, transparent 1.7px); background-size: 10px 10px; }
.rv-dots-fill { height: 100%; border-radius: 999px;
  background-image: radial-gradient(#F5F5F7 1.8px, transparent 1.9px); background-size: 10px 10px; }
.rv-pips { display: flex; gap: 7px; padding-top: 1px; }
.rv-pip { width: 8px; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.16); }
.rv-pip.on { background: #F5F5F7; box-shadow: 0 0 8px rgba(255,255,255,0.45); }
.rv-sub2 { font-size: 12px; color: #A1A1A6; margin-top: 10px; font-variant-numeric: tabular-nums; }
.rv-tstrip { position: relative; height: 12px; margin: 14px 2px 0 2px;
  background-image: radial-gradient(rgba(255,255,255,0.25) 1.5px, transparent 1.6px);
  background-size: 11px 12px; background-position: center; border-radius: 999px; }
.rv-tmark { position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 999px; background: #F5F5F7;
  box-shadow: 0 0 10px rgba(255,255,255,0.6); transform: translate(-50%, -50%); }
.rv-tends { display: flex; justify-content: space-between; font-size: 11.5px; color: #A1A1A6;
  margin-top: 6px; font-variant-numeric: tabular-nums; }
.stApp, .stApp * { caret-color: transparent; }
div[data-testid="stMarkdownContainer"] p { user-select: none; -webkit-user-select: none; cursor: default; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""", unsafe_allow_html=True)


def generate_sample_fundus_image(size: int = 512, seed: int = 42) -> Image.Image:
    np.random.seed(seed)
    h, w = size, size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    fov = size * 0.46
    mask = (dist <= fov).astype(np.float32)

    red = 190 + 35 * np.sin(x / 90.0) - 40 * (dist / fov) ** 2
    green = 85 + 20 * np.cos(y / 90.0) - 30 * (dist / fov) ** 2
    blue = 20 + 8 * np.sin((x + y) / 120.0)

    dx, dy = int(w * 0.32), int(h * 0.5)
    disc = np.clip(1.0 - np.sqrt((x - dx) ** 2 + (y - dy) ** 2) / (size * 0.08), 0.0, 1.0)
    red += disc * 60
    green += disc * 85
    blue += disc * 40

    fx, fy = int(w * 0.62), int(h * 0.52)
    fovea = np.clip(1.0 - np.sqrt((x - fx) ** 2 + (y - fy) ** 2) / (size * 0.12), 0.0, 1.0)
    red -= fovea * 35
    green -= fovea * 25

    vessels = np.zeros((h, w), dtype=np.float32)
    for ang in np.linspace(-np.pi * 0.88, np.pi * 0.88, 14):
        px, py = float(dx), float(dy)
        vx, vy = np.cos(ang) * 1.8, np.sin(ang) * 1.8
        thick = 4.2
        for _ in range(int(size * 0.56)):
            ix, iy = int(px), int(py)
            if 0 <= ix < w and 0 <= iy < h:
                r = max(1, int(thick))
                vessels[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = np.maximum(
                    vessels[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1], thick / 4.2
                )
            vx += np.random.uniform(-0.16, 0.16)
            vy += np.random.uniform(-0.16, 0.16)
            px += vx
            py += vy
            thick = max(1.0, thick - 0.016)
            if not (0 <= px < w and 0 <= py < h):
                break

    green -= vessels * mask * 60.0
    red -= vessels * mask * 35.0
    return Image.fromarray(np.stack([
        np.clip(red * mask, 0, 255).astype(np.uint8),
        np.clip(green * mask, 0, 255).astype(np.uint8),
        np.clip(blue * mask, 0, 255).astype(np.uint8),
    ], axis=-1))


@st.cache_resource
def get_vessel_model(weights_path: str = "weights/vessel_unet.pth") -> VesselUNet:
    model = VesselUNet(in_channels=3, n_classes=1, bilinear=True, base_features=32)
    if weights_path and os.path.exists(weights_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Could not load weights: {e}. Using fresh weights.")
    model.eval()
    return model


def red_overlay(raw: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = raw.copy().astype(np.float32)
    hit = mask > 0
    out[hit] = (1.0 - alpha) * out[hit] + alpha * np.array([255.0, 69.0, 58.0], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def heatmap(prob: np.ndarray, name: str) -> np.ndarray:
    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap(name)
    return (cmap(prob)[:, :, :3] * 255.0).astype(np.uint8)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def icon_uri() -> str:
    with open("assets/app_icon.jpg", "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


@st.cache_data
def case_thumb(which: int) -> Image.Image:
    p = os.path.join("samples", f"fundus_sample_{which}.png")
    if os.path.exists(p):
        return Image.open(p).convert("RGB")
    return generate_sample_fundus_image(512, 42 if which == 1 else 123)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_preprocess(img_bytes: bytes, clip: float):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return preprocess_fundus_image(np.array(img), clip_limit=clip)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_prob(stacked_bytes: bytes, shape: tuple) -> np.ndarray:
    stacked = np.frombuffer(stacked_bytes, dtype=np.float32).reshape(shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return predict_vessels(get_vessel_model(), stacked, device=device)


st.markdown(
    f"<div class='rv-nav'><div class='rv-brand'><div class='rv-mark'><img src='{icon_uri()}' alt='RetinaVessel'></div>"
    "<div><div class='rv-name'>RetinaVessel</div>"
    "<div class='rv-sub'>Fundus vessel segmentation</div></div></div>"
    "<span class='rv-pill'><span class='rv-dot'></span>VesselUNet</span></div>",
    unsafe_allow_html=True,
)

if "case" not in st.session_state:
    st.session_state.case = "Benchmark fundus"

rail, main = st.columns([5, 7], gap="large")

with rail:
    with st.container(border=True):
        st.write("Case")
        upload = st.session_state.get("fundus_upload")
        v1, v2, v3 = st.columns(3)
        v1.image(case_thumb(1), width="stretch")
        v2.image(case_thumb(2), width="stretch")
        with v3:
            if upload is not None:
                st.image(upload, width="stretch")
            else:
                st.markdown("<div class='rv-plus'>+</div>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        source = st.session_state.case
        if b1.button("Benchmark", type="primary" if source == "Benchmark fundus" else "secondary", width="stretch", key="pick_bench"):
            st.session_state.case = "Benchmark fundus"
            st.rerun()
        if b2.button("Dense", type="primary" if source == "Dense microvasculature" else "secondary", width="stretch", key="pick_dense"):
            st.session_state.case = "Dense microvasculature"
            st.rerun()
        if b3.button("Upload", type="primary" if source == "Upload" else "secondary", width="stretch", key="pick_up"):
            st.session_state.case = "Upload"
            st.rerun()
        if source == "Upload":
            st.file_uploader("Fundus image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="fundus_upload", label_visibility="collapsed")
            if upload is not None:
                st.caption(f"{upload.name}")

    with st.container(border=True):
        st.write("Tune")
        t1, t2 = st.columns(2)
        clip = t1.slider("Contrast gain", 1.0, 10.0, 3.0, 0.5)
        cutoff = t2.slider("Vessel threshold", 0.10, 0.90, 0.50, 0.05)
        d1, d2 = st.columns(2)
        opacity = d1.slider("Overlay strength", 0.20, 1.0, 0.70, 0.05)
        palette = d2.selectbox("Probability palette", ["magma", "inferno", "plasma", "viridis", "jet"])
        view_size = st.slider("Viewport size", 280, 860, 560, 20)

image, title, img_bytes = None, "", None
if source == "Upload":
    if upload is not None:
        try:
            image = Image.open(upload).convert("RGB")
            image.thumbnail((1024, 1024), Image.LANCZOS)
            title = upload.name
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception as e:
            st.error(f"Could not open image: {e}")
    else:
        with st.container(border=True):
            st.write("No image selected.")
            st.caption("Upload a fundus photograph to begin.")
elif source == "Benchmark fundus":
    p = os.path.join("samples", "fundus_sample_1.png")
    image = Image.open(p).convert("RGB") if os.path.exists(p) else generate_sample_fundus_image(512, 42)
    title = "Benchmark retinal fundus"
    if os.path.exists(p):
        with open(p, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = None
else:
    p = os.path.join("samples", "fundus_sample_2.png")
    image = Image.open(p).convert("RGB") if os.path.exists(p) else generate_sample_fundus_image(512, 123)
    title = "Dense microvascular retina"
    if os.path.exists(p):
        with open(p, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = None

if image is None:
    st.stop()

if img_bytes is None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

rgb = np.array(image)
h, w = rgb.shape[:2]

t0 = time.time()
with st.spinner("Processing fundus signals..."):
    green_ch, clahe_img, frangi_img, stacked = cached_preprocess(img_bytes, clip)
prep_ms = (time.time() - t0) * 1000

t1 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with st.spinner("Running VesselUNet..."):
    prob = cached_prob(stacked.tobytes(), stacked.shape)
    binary = (prob >= cutoff).astype(np.uint8) * 255
infer_ms = (time.time() - t1) * 1000

overlay = red_overlay(rgb, binary, opacity)
heat = heatmap(prob, palette)
count = int(np.count_nonzero(binary))
density = count / (h * w) * 100.0
conf = float(np.mean(prob[binary > 0])) if count else 0.0

with main:
    st.markdown(f"<p class='rv-title'>{title}</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='rv-meta'>{w} by {h} pixels &nbsp;·&nbsp; {prep_ms:.0f} ms preparation &nbsp;·&nbsp; {infer_ms:.0f} ms inference on {device}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='rv-dotrule'></div>", unsafe_allow_html=True)

    ov_b64 = base64.b64encode(png_bytes(overlay)).decode()
    st.markdown(
        f"<div class='rv-scope'>"
        f"<div class='rv-scope-top'><span>Segmented vasculature</span><span>Ø {view_size} px · {int(opacity*100)}%</span></div>"
        f"<div class='rv-ring-outer'><div class='rv-ring-inner'>"
        f"<img src='data:image/png;base64,{ov_b64}' width='{view_size}' alt='Segmented vessels'>"
        f"</div></div>"
        f"<div class='rv-scope-meta'><span>{w} by {h} source</span><span>eyepiece view</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<p class='rv-h'>Probability and mask</p><p class='rv-s'>Model output and the thresholded result.</p>", unsafe_allow_html=True)
a, b = st.columns(2)
with a:
    with st.container(border=True):
        st.write("Vessel probability")
        st.image(heat, width="stretch")
        st.caption(f"Calibrated output from 0 to 1 in {palette}.")
with b:
    with st.container(border=True):
        st.write("Binary mask")
        st.image(binary, width="stretch")
        st.caption(f"Cutoff at {cutoff:.2f}. White marks vessels.")

st.markdown("<p class='rv-h'>Signal channels</p><p class='rv-s'>The three inputs fused by the model.</p>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.write("Green channel")
        st.image(green_ch, width="stretch")
        st.caption("Vessels read darkest here.")
with c2:
    with st.container(border=True):
        st.write("CLAHE")
        st.image(clahe_img, width="stretch")
        st.caption(f"Local correction at gain {clip:.1f}.")
with c3:
    with st.container(border=True):
        st.write("Frangi response")
        st.image(frangi_img, width="stretch")
        st.caption("Tuned to tubular structure.")

st.markdown("<div class='rv-dotrule'></div><p class='rv-h'>Results</p><p class='rv-s'>Measurements and export.</p>", unsafe_allow_html=True)
fill = min(100.0, density * 8.0)
lit = round(conf * 12)
pips = "".join(
    "<span class='rv-pip on'></span>" if i < lit else "<span class='rv-pip'></span>"
    for i in range(12)
)
with st.container(border=True):
    st.markdown(
        f"""<div class='rv-read'>
          <div class='rv-cell'><div class='rv-lab'>Vascular density</div><div class='rv-num'>{density:.2f}%</div>
            <div class='rv-dots-track'><div class='rv-dots-fill' style='width:{fill:.0f}%'></div></div>
            <div class='rv-sub2'>{count:,} vessel pixels</div></div>
          <div class='rv-cell'><div class='rv-lab'>Mean confidence</div><div class='rv-num'>{conf*100:.1f}%</div>
            <div class='rv-pips'>{pips}</div>
            <div class='rv-sub2'>certainty inside vessels</div></div>
          <div class='rv-cell'><div class='rv-lab'>Inference</div><div class='rv-num'>{infer_ms:.0f} ms</div>
            <div class='rv-sub2'>prep {prep_ms:.0f} ms</div></div>
        </div>""",
        unsafe_allow_html=True,
    )
tpos = min(96.0, max(4.0, (cutoff - 0.10) / 0.80 * 100.0))
gcol, xcol = st.columns(2, gap="medium")
with gcol:
    with st.container(border=True):
        st.write("Reading guidance")
        st.caption(
            "Density is segmented vessel area. Raise the threshold to keep only "
            "high-certainty vessels, lower it to include faint capillaries."
        )
        st.markdown(
            f"<div class='rv-tstrip'><div class='rv-tmark' style='left:{tpos:.1f}%'></div></div>"
            f"<div class='rv-tends'><span>faint · 0.10</span><span>now {cutoff:.2f}</span><span>strict · 0.90</span></div>",
            unsafe_allow_html=True,
        )
with xcol:
    with st.container(border=True):
        st.write("Export plates")
        x1, x2 = st.columns(2)
        with x1:
            st.image(binary, width="stretch")
            st.download_button("Save binary mask", png_bytes(binary),
                "retinal_vessel_binary_mask.png", "image/png", width="stretch")
        with x2:
            st.image(overlay, width="stretch")
            st.download_button("Save overlay", png_bytes(overlay),
                "retinal_vessel_overlay.png", "image/png", width="stretch")

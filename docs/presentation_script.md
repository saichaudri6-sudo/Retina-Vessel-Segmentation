# RetinaVessel — 10-Slide Presentation Script
*Focus: image-processing concepts and the Python behind them. UI is out of scope.*

---

## Slide 1 — Title
**RetinaVessel — Retinal Vessel Segmentation in Pure Python**

- One-line pitch: upload a fundus photograph, get the vessel map.
- Stack: FastAPI + OpenCV/NumPy/Scikit-image for image ops, ONNX Runtime for the network.
- Speaker note: "Modern medicine generates images faster than ophthalmologists can read them. This app answers one question automatically: where are the blood vessels in this retinal photograph?"

---

## Slide 2 — The Problem & the Physics of Fundus Images
**Why vessels, and why the green channel?**

- Fundus photography = camera through the pupil; retina ≈ 5–6 MP of soft tissue.
- Applications: diabetic retinopathy screening, hypertension, glaucoma staging.
- Vessels are the one structure both sparse and precise — vessel geometry = vital signal (tortuosity, branching angles, AV nicking).
- Key optical fact: hemoglobin absorbs green light strongly → **highest vessel-to-background contrast lives in channel 1**, not RGB average.
- Speaker note: "Everything downstream works because somebody chose the right channel. The red channel washes out vessels; the green channel is where blood absorbs."

---

## Slide 3 — Architecture of the App
**Two-layer Python design: a trainable core and a serving skin.**

- `utils/` — *unchanged trainable numerics*: `preprocessing.py` (green→CLAHE→Frangi fusion), `architecture.py` (PyTorch `VesselUNet`).
- `app/` — *serving shell*: `pipeline.py` orchestrates stages, `inference.py` runs ONNX Runtime, `imaging.py` isolates decode/encode. Serving never imports torch.
- Same Python functions power both training and the web app — the web layer *calls* `preprocess_fundus_image()`, it doesn't reimplement it.
- Library budget: NumPy + OpenCV + Scikit-image (classical CV), PyTorch (training only), ONNX Runtime (serving).
- Speaker note: "The split keeps science and hosting out of each other's way: retrain in `utils/`, deploy in `app/`, and numerical parity is checked once."

---

## Slide 4 — Stage 1: Channel Selection & Normalization
**The pipeline begins with domain knowledge, not AI.**

```python
rgb_np = np.asarray(rgb_image)          # (H, W, 3) uint8
if rgb_np.ndim == 2:                    # grayscale in → replicate
    rgb_np = np.stack([rgb_np] * 3, axis=-1)
green_ch = rgb_uint8[:, :, 1]           # NumPy slice = numpy as a DSL
```

- Robustness first: handles grayscale, RGBA, float/double inputs by normalizing dtype.
- One `[:, :, 1]` slice costs microseconds — the most valuable operation in the pipeline.
- Output: `(green, clahe, frangi, stacked)` tuple → the contract every stage obeys.
- Speaker note: "The first engineering lesson: normalize your input surface. Whatever the user uploads — PNG, JPG, grayscale, float — it becomes standard uint8 before the pipeline touches it."

---

## Slide 5 — Stage 2: CLAHE (Adaptive Contrast)
**Fixing uneven illumination with histograms, 8×8 tiles at a time.**

- Problem: fundus illumination falls off toward periphery; small vessels live in low-contrast corners.
- CLAHE = histogram equalization *per tile* with a **clip limit** that caps over-amplification of noise.
- 2 lines of OpenCV do the work:

```python
clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
clahe_img = clahe.apply(green_ch)
```

- `clip_limit` is user-tunable (slider, default 3.0) — clinical images are inconsistent; the operator adjusts interactively.
- Speaker note: "Global equalization would blow up noise in bright patches. CLAHE runs the histogram per 8×8 tile — local contrast, then the clip limits how greedy it gets."

---

## Slide 6 — Stage 3: Frangi Vesselness (the math slide)
**Multiscale Hessian analysis — "is this pixel part of a tube?"**

- Compute image Hessian (second derivatives); get eigenvalues λ1, λ2, λ3 per pixel.
- Vessel signature: **two small eigenvalues, one elongated one** — a ridge.
- 371-word paper → 4 lines of Scikit-image:

```python
sigmas = np.arange(1.0, 4.0, 0.8)          # multiscale: thin → thick vessels
frangi_raw = frangi(clahe_float, sigmas=sigmas,
                    alpha=0.5, beta=0.5, gamma=15.0,
                    black_ridges=True)     # vessels are DARK on light retina
frangi_img = (frangi_raw - f_min) / (f_max - f_min)   # normalize → [0,1]
```

- Multiscale loop = each σ is a matched filter for a different vessel width; max across scales.
- Normalization gives every stage a common [0, 1] output contract.
- Speaker note: "Alpha/beta/gamma shape the sensitivity to blob versus line structures. black_ridges=True is the domain fact that vessels appear darker than the retina in green light."

---

## Slide 7 — Stage 4: The Deep Model — VesselUNet
**U-Net: encoder-decoder with skip connections, a natural fit for segmentation.**

```python
class VesselUNet(nn.Module):        # PyTorch
    def forward(self, x):
        x1 = self.inc(x)            # Green | CLAHE | Frangi → 3 ch
        x5 = self.down4(...)        # 4× MaxPool ↓; features = base*8
        x  = self.up4(self.up3(...), x1)   # skip connections back
        return self.outc(x)         # 1×1 conv + Sigmoid → per-pixel P(vessel)
```

- `DoubleConv` = (Conv→BatchNorm→ReLU) ×2; `Down` = MaxPool + DoubleConv; `Up` = bilinear upsample + skip concat.
- `base_features=32` keeps the model small (~17 MB as ONNX).
- The network's input is the output of slides 4–6: a **3-channel fusion stack** `[Green, CLAHE, Frangi]` — classical CV amplifies structure; the CNN learns to threshold semantics.
- Speaker note: "Frangi knows geometry, not semantics — it fires on *any* dark ridge. The U-Net, fed Frangi as an input channel, learns to reject optic-disk edges and lesions and keep true vessels."

---

## Slide 8 — From Training to Serving: the ONNX Bridge
**Train in torch, serve without torch — one-time verified export.**

- Serving process never imports torch (torch is 2 GB of GPU machinery; free-tier hosts are 512 MB).
- `scripts/export_onnx.py`: `torch.onnx.export()` with fixed work-resolution + a **numerical parity test** — max output drift vs. the torch model ≈ **1e-4**.
- Runtime session config (app/inference.py):

```python
opts = SessionOptions()
opts.graph_optimization_level = ORT_ENABLE_ALL
opts.enable_cpu_mem_arena = False    # arena-free: +27 MB instead of +300 MB
opts.enable_mem_pattern = False
session.run(["prob"], {"stacked": x})   # (H,W,3) → (H,W) vessel probabilities
```

- Lazy, thread-safe single session (double-checked lock) — the model loads on first request, not at boot.
- Speaker note: "ONNX is the contract between the research stack and the deployed stack: same math, no framework lock-in, no GPU dependency."

---

## Slide 9 — Post-Processing: Probability → Pixels on Screen
**One slider becomes a mask; NumPy does the arithmetic.**

```python
def binary_of(prob, cutoff):
    return (prob >= cutoff).astype(np.uint8) * 255      # 0/255 vessel mask

def overlay_plate(rgb, mask, alpha):
    out = rgb.astype(np.float32)
    hit = mask > 0
    out[hit] = out[hit] * (1 - alpha) + tint[hit] * alpha   # vectorized blend
```

- Vascular density = `mask.mean()` — a set-theoretic count of vessel pixels, in float.
- Heat plates: Colormap LUTs (`plt.get_cmap("magma")` → 256×3 uint8 array) map probability → color.
- Timing wrappers (`time.perf_counter()`) expose per-stage costs: preprocess vs. inference.
- Speaker note: "Post-processing is where clinical control happens: the threshold slider is a recall/precision dial — lower it to recline confidence and include faint capillaries."

---

## Slide 10 — Engineering the Runtime (+ takeaways)
**What the process looks like at 512 MB, one CPU, zero GPU.**

- **Global compute lock** (`threading.Lock`): Frangi + inference run one-at-a-time per process — concurrency would multiply transient memory into OOM.
- **Session on first use**, arena-free ONNX → the whole app fits a Render free-tier dyno.
- Plate caching in the HTTP layer: URLs encode only the parameters each plate depends on → browser skips recomputing unchanged images (ETag/304).
- Takeaways:
  1. Classical CV + deep learning compose naturally — fusion stack = domain knowledge + learned semantics.
  2. Python covers the entire lifecycle: algorithms (NumPy/OpenCV), training (torch), interpolation (ONNX), serving (FastAPI).
  3. On constrained hosts, memory engineering matters as much as algorithms.
- Speaker note: "This is Python's superpower: the same language that trains the research model runs the 17-MB production inference behind a web API — nothing between them except ONNX."

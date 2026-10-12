# RetinaVessel AI: Retinal Vessel Segmentation

A clinical-grade deep learning vision application built with Streamlit and PyTorch for automated microvascular vessel segmentation from color retinal fundus photography.

---

## 🌟 Clinical & Optical Rationale

1. **Green Spectral Channel Extraction**:
   - Ocular hemoglobin exhibits peak optical absorption in the **540–577 nm (Green)** spectral window.
   - In contrast, the Red channel suffers from scleral/choroidal reflectance saturation, and the Blue channel suffers from ocular media scatter.
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   - Normalizes non-uniform illumination profiles across the retinal curvature and sharpens capillary boundaries without amplifying background noise.
3. **Multiscale Frangi Vesselness Filter**:
   - Computes multiscale eigenvalues of the 2D Hessian matrix to identify continuous tubular/ridge geometries while attenuating point noise and planar backgrounds.
4. **PyTorch VesselUNet Feature Fusion**:
   - Fuses all three representations `[Green Channel, CLAHE, Frangi]` through an encoder-decoder architecture with skip connections, outputting a calibrated probability mask in $[0, 1]$.

---

## 📁 Project Architecture

```
retina-vessel-app/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── utils/
│   ├── __init__.py             # Utils exports
│   ├── preprocessing.py        # Optical preprocessing: Green, CLAHE, Frangi
│   └── architecture.py         # PyTorch VesselUNet architecture & inference
├── scripts/
│   └── train_or_calibrate_unet.py  # Model calibration & sample generator
├── samples/                    # Diagnostic benchmark fundus images
└── weights/                    # Calibrated VesselUNet model weights (.pth)
```

---

## 🚀 Running the Application

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Interactive Controls
- **CLAHE Clip Limit (1.0 – 10.0)**: Adjusts local contrast amplification.
- **Vessel Probability Threshold (0.1 – 0.9)**: Decision boundary for binary vascular classification.
- **Red Overlay Opacity**: Transparency blend of segmented vessels overlaid on the original fundus image.
- **Heatmap Palette**: Selectable colormaps (`magma`, `inferno`, `plasma`, `viridis`, `jet`).
- **Binary Mask Export**: Direct PNG download of the segmentation mask.

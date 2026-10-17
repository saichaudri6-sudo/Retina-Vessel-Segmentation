"""Pre-processing module for retinal fundus images.

In fundus photography, the green spectral channel offers maximum optical
contrast between hemoglobin in blood vessels and the retinal background.
Contrast-Limited Adaptive Histogram Equalization (CLAHE) corrects for non-uniform
retinal illumination and local contrast decay. The Frangi vesselness filter
analyzes multiscale eigenvalues of the image Hessian matrix to amplify tubular,
vessel-like structures while attenuating planar background and point noise.
"""

from typing import Tuple
import numpy as np
import cv2
from skimage.filters import frangi


def preprocess_fundus_image(
    rgb_image: np.ndarray,
    clip_limit: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a retinal fundus image for vessel segmentation.

    Steps:
      1. Converts input to a standard RGB uint8 numpy array.
      2. Extracts the Green spectral channel (highest hemoglobin contrast).
      3. Applies CLAHE to correct non-uniform illumination and sharpen microvasculature.
      4. Computes Frangi multiscale vesselness filter on the enhanced image.
      5. Stacks Green, CLAHE, and Frangi into a 3-channel feature representation.

    Args:
        rgb_image: Input fundus image as a NumPy uint8 array (H, W, 3) in RGB order.
        clip_limit: Contrast clipping limit for CLAHE (default: 3.0).

    Returns:
        green_ch: 2D uint8 NumPy array of the green channel.
        clahe_img: 2D uint8 NumPy array of the CLAHE-enhanced image.
        frangi_img: 2D float32 NumPy array of Frangi vesselness response normalized to [0, 1].
        stacked_img: 3D float32 NumPy array of shape (H, W, 3) normalized to [0, 1]
                     comprising [Green, CLAHE, Frangi] channels.
    """
    # 1. Convert input to standard NumPy uint8 RGB array
    rgb_np = np.asarray(rgb_image)

    # Handle grayscale or alpha channel
    if rgb_np.ndim == 2:
        rgb_np = np.stack([rgb_np] * 3, axis=-1)
    elif rgb_np.ndim == 3 and rgb_np.shape[2] == 4:
        rgb_np = rgb_np[:, :, :3]

    # Normalize dtype to uint8 [0, 255]
    if np.issubdtype(rgb_np.dtype, np.floating):
        if rgb_np.max() <= 1.0:
            rgb_uint8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb_np, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb_np.astype(np.uint8)

    # 2. Extract Green channel (channel index 1 in RGB)
    green_ch = rgb_uint8[:, :, 1]

    # 3. Apply CLAHE using OpenCV
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(8, 8)
    )
    clahe_img = clahe.apply(green_ch)

    # 4. Apply Frangi vesselness filter
    # Vessels appear darker than surrounding retina in green channel,
    # so black_ridges=True detects tubular dark ridges on light background.
    clahe_float = clahe_img.astype(np.float32) / 255.0
    sigmas = np.arange(1.0, 4.0, 0.8)
    frangi_raw = frangi(
        clahe_float,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=15.0,
        black_ridges=True
    )

    # Normalize Frangi output to [0, 1] float32
    f_min = float(np.min(frangi_raw))
    f_max = float(np.max(frangi_raw))
    if f_max > f_min:
        frangi_img = ((frangi_raw - f_min) / (f_max - f_min)).astype(np.float32)
    else:
        frangi_img = np.zeros_like(frangi_raw, dtype=np.float32)

    # 5. Build 3-channel stacked representation: [Green, CLAHE, Frangi]
    green_norm = (green_ch.astype(np.float32) / 255.0)
    clahe_norm = (clahe_img.astype(np.float32) / 255.0)
    stacked_img = np.stack([green_norm, clahe_norm, frangi_img], axis=-1).astype(np.float32)

    return green_ch, clahe_img, frangi_img, stacked_img

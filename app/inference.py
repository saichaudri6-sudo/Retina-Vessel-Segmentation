"""ONNX Runtime inference session (lazy, thread-safe).

The serving process never imports torch. The trained VesselUNet is exported
once to weights/vessel_unet.onnx (~17 MB) and verified numerically against
the torch model by scripts/export_onnx.py (max drift ~1e-4).
"""

from __future__ import annotations

import threading

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from app.config import WEIGHTS_PATH

_session: InferenceSession | None = None
_init_lock = threading.Lock()
# Heavy stages (Frangi preprocess, ONNX inference) still run one at a time per
# process. Concurrent computes multiply transient memory and OOM small hosts.
compute_lock = threading.Lock()


def onnx_session() -> InferenceSession:
    """Double-checked lazy creation of the single ONNX inference session."""
    global _session
    if _session is None:
        with _init_lock:
            if _session is None:
                opts = SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.log_severity_level = 3
                # 512 MB hosts: the default CPU arena reserves hundreds of MB and
                # keeps growing across inferences. Arena-free runs in +27 MB.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                _session = InferenceSession(
                    str(WEIGHTS_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
                )
    return _session


def infer_vessels(stacked: np.ndarray) -> np.ndarray:
    """Run the ONNX UNet on a (H, W, 3) float32 stack; returns (H, W) in [0, 1]."""
    out = onnx_session().run(["prob"], {"stacked": stacked[None].transpose(0, 3, 1, 2)})
    return out[0][0, 0].astype(np.float32)

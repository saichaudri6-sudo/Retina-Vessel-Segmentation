"""Export VesselUNet to ONNX Runtime format.

Builds a single graph (UNet + Sigmoid) from weights/vessel_unet.pth, saves
weights/vessel_unet.onnx, then verifies numerical parity against the PyTorch
model across square, wide and tall inputs. The deployed app needs no torch.

Run:  python scripts/export_onnx.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import onnxruntime as ort

from utils.architecture import VesselUNet


EXPORT_PATH = os.path.join("weights", "vessel_unet.onnx")


def build_model() -> VesselUNet:
    model = VesselUNet(in_channels=3, n_classes=1, bilinear=True, base_features=32)
    weights_path = os.path.join("weights", "vessel_unet.pth")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export(model: VesselUNet) -> None:
    dummy = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        EXPORT_PATH,
        input_names=["stacked"],
        output_names=["prob"],
        dynamic_axes={
            "stacked": {2: "height", 3: "width"},
            "prob": {2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported -> {EXPORT_PATH} ({os.path.getsize(EXPORT_PATH) / 1e6:.1f} MB)")


def parity_check(model: VesselUNet) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(EXPORT_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

    shapes = [(512, 512), (640, 480), (480, 640)]
    worst = 0.0
    for h, w in shapes:
        rng = np.random.default_rng(7)
        stacked = rng.random((h, w, 3), dtype=np.float32)

        with torch.no_grad():
            ref = model(torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)).squeeze().numpy()

        out = ort_session.run(["prob"], {"stacked": stacked[None].transpose(0, 3, 1, 2)})[0][0, 0]
        diff = float(np.abs(ref - out).max())
        worst = max(worst, diff)
        print(f"  {w}x{h}: max |torch - onnx| = {diff:.2e}")

    assert worst < 1e-3, f"parity drift too large: {worst}"
    print(f"parity ok (worst {worst:.2e}) -> deployed outputs match torch")


if __name__ == "__main__":
    with torch.inference_mode():
        net = build_model()
        export(net)
        parity_check(net)

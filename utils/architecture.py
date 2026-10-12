"""PyTorch U-Net Architecture for Retinal Vessel Segmentation.

VesselUNet implements a deep encoder-decoder convolutional architecture with
residual/skip connections between corresponding contracting and expanding levels.
It ingests a 3-channel enhanced representation (Green, CLAHE, Frangi) and outputs
a single-channel probability map in [0, 1] via Sigmoid activation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """Two successive Convolution-BatchNorm-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Bilinear interpolation / ConvTranspose followed by skip concatenation and DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dimensions (handles arbitrary H, W)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution mapped through Sigmoid to yield a probability mask."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv(x))


class VesselUNet(nn.Module):
    """U-Net architecture specifically tailored for retinal vessel segmentation.

    Accepts 3-channel inputs:
      Channel 0: Green channel
      Channel 1: CLAHE-enhanced contrast
      Channel 2: Multiscale Frangi vesselness filter response

    Outputs:
      1-channel probability mask in [0, 1] representing vascular likelihood.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        base_features: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        f = base_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, (f * 16) // factor)

        self.up1 = Up(f * 16, (f * 8) // factor, bilinear)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution layers with Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def predict_vessels(
    model: VesselUNet,
    stacked_image: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Run inference with VesselUNet on a 3-channel stacked fundus image.

    Args:
        model: Instantiated VesselUNet model.
        stacked_image: NumPy array of shape (H, W, 3) with values in [0, 1].
        device: PyTorch device ('cuda' or 'cpu'). Defaults to auto-detect.

    Returns:
        prob_mask: 2D NumPy array of shape (H, W) with probability values in [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Convert (H, W, C) -> (1, C, H, W)
    tensor_img = torch.from_numpy(stacked_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(tensor_img)
        prob_mask = pred.squeeze().cpu().numpy()

    return prob_mask.astype(np.float32)

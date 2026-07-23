"""A compact 3D U-Net producing a per-voxel detection heatmap.

This is the harness's model — small enough to smoke-train, with the shape the detection task needs: a
temporal window in (as channels), one cellness map out. The production detector (its depth, width, and the
anisotropy-aware pooling that a 4x-anisotropic volume calls for) is a separate task; here the pooling is
kept isotropic and shallow so the training loop can be exercised end to end without a GPU-scale model.
"""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class DetectionUNet(nn.Module):
    """A one-downsampling 3D U-Net: temporal window in, a single cellness heatmap out."""

    def __init__(self, window: int, width: int = 16) -> None:
        super().__init__()
        self.encode = self._conv_block(window, width)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self._conv_block(width, width * 2)
        self.up = nn.ConvTranspose3d(width * 2, width, kernel_size=2, stride=2)
        self.decode = self._conv_block(width * 2, width)
        self.head = nn.Conv3d(width, 1, kernel_size=1)

    @override
    def forward(self, frames: Float[Tensor, "b t z y x"]) -> Float[Tensor, "b z y x"]:
        """Predict a cellness heatmap for the window's central frame."""
        skip = self.encode(frames)
        bottom = self.bottleneck(self.pool(skip))
        merged = self.decode(torch.cat([self.up(bottom), skip], dim=1))
        return self.head(merged).squeeze(1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Two padded 3D convolutions with normalisation and activation — one resolution level's work."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

"""A 3D U-Net producing a per-voxel detection heatmap, with anisotropy-aware pooling.

The imaging is 4x anisotropic — a voxel is 1.625 um in z against 0.40625 um in y and x — so an isotropic
pooling stride would shrink the physical receptive field four times faster in z than in y/x and blur the
axes together. Resampling to isotropic is worse: upsampling z quadruples the volume, and downsampling y/x
throws away the resolution the cells actually live at (a ~4 um cell is ~10 voxels in y/x, ~2.5 in z).
So pooling is *anisotropic* — each level's stride is chosen per axis, pooling y/x more often than z until
the physical voxel is roughly cubic, after which pooling is isotropic. The strides are the model's one
structural choice and are passed in, not hard-coded, so the harness can stay shallow and isotropic while
the detector goes deep and anisotropy-aware.
"""

from typing import TypedDict, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Checkpoint(TypedDict):
    """A saved detector: its weights and the structure needed to rebuild the network around them."""

    state_dict: dict[str, Tensor]
    strides: list[list[int]]
    window: int
    width: int


# Anisotropy is 4x, so pooling y/x twice (2*2) before touching z equalises the physical voxel; deeper
# levels then pool all three axes. This is the detector's default depth.
ANISOTROPIC_STRIDES: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (1, 2, 2), (2, 2, 2))


class DetectionUNet(nn.Module):
    """A U-Net whose per-level pooling strides make the receptive field physically isotropic."""

    def __init__(
        self,
        window: int,
        width: int = 16,
        strides: tuple[tuple[int, int, int], ...] = ((2, 2, 2),),
    ) -> None:
        super().__init__()
        levels = len(strides)
        level_channels = [width * 2**level for level in range(levels)]  # c_0 .. c_{L-1}
        bottleneck_channels = width * 2**levels
        below = [*level_channels[1:], bottleneck_channels]  # channels feeding each level's upsample

        self.encoders = nn.ModuleList(
            self._conv_block(window if level == 0 else level_channels[level - 1], level_channels[level])
            for level in range(levels)
        )
        self.pools = nn.ModuleList(nn.MaxPool3d(stride) for stride in strides)
        self.bottleneck = self._conv_block(level_channels[-1], bottleneck_channels)
        self.ups = nn.ModuleList(
            nn.ConvTranspose3d(below[level], level_channels[level], kernel_size=stride, stride=stride)
            for level, stride in enumerate(strides)
        )
        self.decoders = nn.ModuleList(
            self._conv_block(2 * level_channels[level], level_channels[level]) for level in range(levels)
        )
        self.head = nn.Conv3d(width, 1, kernel_size=1)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "DetectionUNet":
        """Rebuild the network from a saved checkpoint's structure and load its weights."""
        strides = tuple((stride[0], stride[1], stride[2]) for stride in checkpoint["strides"])
        model = cls(window=int(checkpoint["window"]), width=int(checkpoint["width"]), strides=strides)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @override
    def forward(self, frames: Float[Tensor, "b t z y x"]) -> Float[Tensor, "b z y x"]:
        """Predict a cellness heatmap for the window's central frame, at the input resolution."""
        skips = []
        features = frames
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            features = encoder(features)
            skips.append(features)
            features = pool(features)
        features = self.bottleneck(features)
        for decoder, up, skip in zip(reversed(self.decoders), reversed(self.ups), reversed(skips), strict=True):
            features = decoder(torch.cat([up(features), skip], dim=1))
        return self.head(features).squeeze(1)

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

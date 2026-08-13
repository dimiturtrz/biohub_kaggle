"""On-device, per-step generation of hard scene-pairs — dynamic, GPU, no pool and no disk.

`synthetic_scene` renders on the CPU into a fixed pool decoded once; a pool is a finite set the head can
memorise instead of learning the rule. This generates a FRESH batch of scene-pairs every training step, on the
training device, so the corpus is effectively infinite and the head has to learn "follow the motion" rather
than recognise 300 scenes. It is also faster: the render is a scatter of unit peaks followed by a separable
Gaussian blur, both `conv3d`-speed on the GPU, so generation overlaps training rather than gating it on a CPU
loop.

Each pair is two frames of one crowded population moved one step, with a fast-mover tail so a cell's true
successor lands far while a neighbour sits near — the contested case, its answer known because we placed both
ends. The output is exactly the `(frames, centres, edge matrix)` a `PairSample` carries, already on-device, so
it drops into the batched trainer beside the real pairs with no adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from celltrack.data.joint_dataset import PairSample
from celltrack.data.synthetic_scene import _SCENE_DOWNSAMPLE, SceneConfig


@dataclass(frozen=True)
class GpuScenes:
    """A fresh batch of hard scene-pairs generated on `device` per call — the dynamic synthetic corpus."""

    config: SceneConfig
    device: str

    def batch(self, batch_size: int, generator: torch.Generator) -> list[PairSample]:
        """`batch_size` independent scene-pairs, each a crowded population advanced one fast-mover step."""
        config = self.config
        shape = torch.tensor(config.volume_shape, dtype=torch.float32, device=self.device)
        low = torch.full((3,), config.margin_vox, dtype=torch.float32, device=self.device)
        high = shape - config.margin_vox
        cells = config.n_cells

        start = low + (high - low) * torch.rand(batch_size, cells, 3, generator=generator, device=self.device)
        step = config.speed_um_mean / config.spacing_um
        spread = config.speed_um_std / config.spacing_um
        direction = torch.randn(batch_size, cells, 3, generator=generator, device=self.device)
        speed = step + torch.randn(batch_size, cells, 1, generator=generator, device=self.device).abs() * spread
        velocity = direction / direction.norm(dim=2, keepdim=True).clamp_min(1e-9) * speed
        nxt = torch.clamp(start + velocity, low, high)

        first = self._render(start)  # (B, Z, Y, X)
        second = self._render(nxt)
        lift = torch.tensor([1, *_SCENE_DOWNSAMPLE], dtype=torch.float32, device=self.device)[1:]  # (z, y, x) lift
        eye = torch.eye(cells, device=self.device)
        return [
            PairSample(
                frame_t=first[index],
                frame_t1=second[index],
                source_centres=(start[index] * lift).round().to(torch.int64),
                target_centres=(nxt[index] * lift).round().to(torch.int64),
                edge_matrix=eye,
            )
            for index in range(batch_size)
        ]

    def _render(self, centres: Float[Tensor, "b n 3"]) -> Float[Tensor, "b z y x"]:
        """Scatter a unit peak at every cell centre, then a separable Gaussian blur — blobs at `conv3d` speed."""
        batch_size, cells = centres.shape[:2]
        z, y, x = self.config.volume_shape
        volume = torch.zeros(batch_size, 1, z, y, x, device=self.device)
        voxel = (
            centres.round()
            .to(torch.int64)
            .clamp(
                min=torch.zeros(3, dtype=torch.int64, device=self.device),
                max=torch.tensor([z - 1, y - 1, x - 1], device=self.device),
            )
        )
        flat = (voxel[..., 0] * y * x + voxel[..., 1] * x + voxel[..., 2]).reshape(batch_size, cells)
        peaks = volume.reshape(batch_size, 1, -1)
        peaks.scatter_(2, flat.unsqueeze(1), self.config.peak_intensity)  # one peak per occupied voxel
        return self._blur(peaks.reshape(batch_size, 1, z, y, x))[:, 0]

    def _blur(self, volume: Float[Tensor, "b c z y x"]) -> Float[Tensor, "b c z y x"]:
        """Separable 3D Gaussian — one 1D convolution per axis, so the cost is O(K) not O(K^3) per voxel."""
        radius = math.ceil(3.0 * self.config.blob_sigma_vox)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        kernel = torch.exp(-(offsets**2) / (2.0 * self.config.blob_sigma_vox**2))
        kernel = kernel / kernel.max()  # peak-normalised so a lone cell reads `peak_intensity`, not area-normalised
        for axis in range(2, 5):
            shape = [1, 1, 1, 1, 1]
            shape[axis] = kernel.numel()
            padding = [0, 0, 0]
            padding[axis - 2] = radius
            volume = torch.nn.functional.conv3d(volume, kernel.reshape(shape), padding=tuple(padding))
        return volume

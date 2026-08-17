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

        # A cell keeps its brightness across the pair (drawn once, indexed in both frames), like the CPU scene.
        log = torch.randn(batch_size, cells, generator=generator, device=self.device) * config.intensity_log_std
        intensities = config.peak_intensity * log.exp()
        first = self._render(start, intensities, generator)  # (B, Z, Y, X)
        second = self._render(nxt, intensities, generator)
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

    def _render(
        self, centres: Float[Tensor, "b n 3"], intensities: Float[Tensor, "b n"], generator: torch.Generator
    ) -> Float[Tensor, "b z y x"]:
        """Scatter each cell's peak, a separable Gaussian blur, then background + noise — blobs at `conv3d` speed."""
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
        # Reduce by amax so colliding cells read as one peak of the right height, matching the CPU scene's max.
        peaks.scatter_reduce_(2, flat.unsqueeze(1), intensities.unsqueeze(1), reduce="amax", include_self=True)
        blurred = self._blur(peaks.reshape(batch_size, 1, z, y, x))[:, 0]
        return self._corrupt(blurred, generator)

    def _blur(self, volume: Float[Tensor, "b c z y x"]) -> Float[Tensor, "b c z y x"]:
        """Separable 3D Gaussian, z-anisotropic — one 1D convolution per axis, so cost is O(K) not O(K^3)."""
        sigma_axis = (
            self.config.blob_sigma_vox * self.config.sigma_z_ratio,
            self.config.blob_sigma_vox,
            self.config.blob_sigma_vox,
        )
        for axis in range(2, 5):
            sigma = sigma_axis[axis - 2]
            radius = math.ceil(3.0 * sigma)
            offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
            kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
            kernel = kernel / kernel.max()  # peak-normalised so a lone cell reads its own peak, not area-normalised
            shape = [1, 1, 1, 1, 1]
            shape[axis] = kernel.numel()
            padding = [0, 0, 0]
            padding[axis - 2] = radius
            volume = torch.nn.functional.conv3d(volume, kernel.reshape(shape), padding=tuple(padding))
        return volume

    def _corrupt(self, volume: Float[Tensor, "b z y x"], generator: torch.Generator) -> Float[Tensor, "b z y x"]:
        """Constant background then Poisson shot + Gaussian read noise — the torch twin of the CPU scene's."""
        config = self.config
        out = volume + config.background_level
        if config.noise_shot_photons > 0.0:
            rate = out.clamp_min(0.0) * config.noise_shot_photons
            out = torch.poisson(rate, generator=generator) / config.noise_shot_photons
        if config.noise_read_std > 0.0:
            noise = torch.randn(out.shape, generator=generator, device=self.device) * config.noise_read_std
            out = out + noise
        return out.clamp_min(0.0)

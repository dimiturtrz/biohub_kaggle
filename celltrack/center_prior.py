"""DeepCenter: pilkwang's centre-prior U-Net, mounted as a second-opinion veto over recovered nodes.

The geometry passes — division recovery and gap-closing — reconnect cells a one-to-one linker drops, but
on real detections a loose geometric gate invents more false links than true ones, and every false link is
scored twice (against the term it reached for and against the larger edge term). DeepCenter is a separate
`DeepCenterUNet3D` centre-heatmap detector, trained only to answer "is there a cell centre here?"; the
frontier gates each recovered candidate on it, keeping the fork or bridge only when a second, independent
model confirms a centre at that voxel. That is the precision half those passes lack on their own.

The mount mirrors the detector's: the heatmap is a pure function of (weights, video), so it is forwarded
once per video and cached (`ResponseCache`), and every later read-out is a cheap max-pool over the cached
volume. The read-out — `CenterConfirmer.confirm` — maps a full-resolution `(z, y, x)` voxel to the pooled
heatmap grid, takes the max over a small anisotropic window (the centre may sit a voxel off), and thresholds
it. `CenterPriorScorer` is the cached forward-once/replay wrapper; `CenterPriorVeto` binds one video's
heatmaps and hands a stage a `CenterConfirmer` at the stage's own threshold and window.

The architecture and preprocessing are pilkwang's (`train_full_frame_center_detector.py`): a three-level
3D U-Net over an XY-block-mean-pooled, per-frame percentile-normalised frame. The model is MOUNTED, never
trained here — this repo owns only the inference recipe, the cached read-out, and the veto API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from celltrack.response_cache import ResponseCache

_POOL_MULTIPLE = 8  # three MaxPool3d(2) levels: each spatial dim must divide 8 for the skip-concats to align
_BLOCK_NEST = ".block."  # pilkwang's per-level `ConvBlock3d.block` nesting, flattened out on load (see from_pack)


@dataclass(frozen=True)
class CenterPriorRecipe:
    """The fixed preprocessing around the network — XY pooling and per-frame percentile normalisation."""

    pool_factor: int = 4
    norm_lo_pct: float = 50.0
    norm_hi_pct: float = 99.5
    clip_lo: float = -0.5
    clip_hi: float = 6.0

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "CenterPriorRecipe":
        """Rebuild the recipe from a saved DeepCenter config, tolerating its superset of training keys."""
        return cls(
            pool_factor=int(config.get("pool_factor", 4)),  # type: ignore[arg-type]
            norm_lo_pct=float(config.get("norm_lo_pct", 50.0)),  # type: ignore[arg-type]
            norm_hi_pct=float(config.get("norm_hi_pct", 99.5)),  # type: ignore[arg-type]
            clip_lo=float(config.get("norm_clip_lo", -0.5)),  # type: ignore[arg-type]
            clip_hi=float(config.get("norm_clip_hi", 6.0)),  # type: ignore[arg-type]
        )

    def pooled_and_normalised(self, frame: Float[np.ndarray, "z y x"]) -> Float[np.ndarray, "z y x"]:
        """One frame XY-block-mean-pooled then percentile-normalised — the network's input, as it was trained."""
        pooled = self._block_mean_xy(frame.astype(np.float32, copy=False))
        lo = float(np.percentile(pooled, self.norm_lo_pct))
        hi = float(np.percentile(pooled, self.norm_hi_pct))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(pooled, dtype=np.float32)
        return np.clip((pooled - lo) / (hi - lo), self.clip_lo, self.clip_hi).astype(np.float32)

    def _block_mean_xy(self, volume: Float[np.ndarray, "z y x"]) -> Float[np.ndarray, "z y x"]:
        """Average each `pool_factor x pool_factor` XY block, cropping the ragged edge — the trained pooling."""
        factor = self.pool_factor
        if factor <= 1:
            return volume
        z, y, x = volume.shape
        y2, x2 = (y // factor) * factor, (x // factor) * factor
        return volume[:, :y2, :x2].reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


@dataclass(frozen=True)
class CenterConfirmer:
    """One video's cached heatmaps plus a decision rule: does a candidate voxel sit on a confirmed centre?

    Bound to a single video (the heatmaps) and a single stage's gate (the threshold and window). A candidate
    is confirmed when the maximum heatmap value over a small `(z, y, x)` window around its pooled voxel meets
    the threshold — the max, not the point value, because the recovered centre may land a voxel off the peak.
    """

    heatmaps: tuple[Float[np.ndarray, "z y x"], ...]
    pool_factor: int
    threshold: float
    window_z: int
    window_yx: int

    def confirm(self, timepoint: int, point_zyx: Int[np.ndarray, "3"]) -> bool:
        """Whether the centre prior confirms a cell at this voxel; True (fail-open) when the frame is missing."""
        if timepoint < 0 or timepoint >= len(self.heatmaps):
            return True
        heatmap = self.heatmaps[timepoint]
        if heatmap.size == 0:
            return True
        z = round(float(point_zyx[0]))
        y = round(float(point_zyx[1]) / max(self.pool_factor, 1))
        x = round(float(point_zyx[2]) / max(self.pool_factor, 1))
        patch = heatmap[
            max(0, z - self.window_z) : z + self.window_z + 1,
            max(0, y - self.window_yx) : y + self.window_yx + 1,
            max(0, x - self.window_yx) : x + self.window_yx + 1,
        ]
        if patch.size == 0:
            return True
        return float(np.max(patch)) >= self.threshold


class CenterVetoConfig(BaseModel):
    """A stage's centre-prior gate — the confirm threshold and the anisotropic max-pool window (pooled voxels).

    Carried on a division/gap spec so a `PipelineConfig` expresses the vetoed stage declaratively; it is
    resolved to a `CenterConfirmer` once a video's `CenterPriorVeto` is in hand. The window is anisotropic
    because Z is already at full resolution while Y/X are pooled — the frontier's `(1, 2, 2)` default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(0.25, ge=0, le=1)
    window_z: int = Field(1, ge=0)
    window_yx: int = Field(2, ge=0)

    def resolve(self, veto: "CenterPriorVeto | None") -> "CenterConfirmer | None":
        """This gate against a video's veto — a bound confirmer, or None when no heatmaps are supplied."""
        return veto.for_gate(self) if veto is not None else None


@dataclass(frozen=True)
class CenterPriorVeto:
    """One video's cached heatmaps, ready to hand a stage a `CenterConfirmer` at the stage's own gate."""

    heatmaps: tuple[Float[np.ndarray, "z y x"], ...]
    recipe: CenterPriorRecipe

    def for_gate(self, gate: CenterVetoConfig) -> CenterConfirmer:
        """A confirmer over this video's heatmaps at a stage's configured threshold and window."""
        return self.confirmer(gate.threshold, gate.window_z, gate.window_yx)

    def confirmer(self, threshold: float, window_z: int, window_yx: int) -> CenterConfirmer:
        """A confirmer over this video's heatmaps at a given threshold and anisotropic window."""
        return CenterConfirmer(
            heatmaps=self.heatmaps,
            pool_factor=self.recipe.pool_factor,
            threshold=threshold,
            window_z=window_z,
            window_yx=window_yx,
        )


class CenterPrior(nn.Module):
    """Pilkwang's three-level 3D U-Net centre-prior, mounted — forwards a video into per-frame heatmaps.

    The architecture is `DeepCenterUNet3D` verbatim (mounted, never trained here); the per-level feature
    block is pilkwang's two-conv/group-norm/SiLU stack, built as a plain `Sequential` so the whole network is
    one class rather than a helper hierarchy. Their checkpoint nests each block one level deeper (a
    `ConvBlock3d.block` attribute), so `from_pack` flattens that one prefix out of the state-dict keys.
    """

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1, self.down1 = self._block(1, c), nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2, self.down2 = self._block(c, c * 2), nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3, self.down3 = self._block(c * 2, c * 4), nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self._block(c * 4, c * 8)
        self.up3, self.dec3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2), self._block(c * 8, c * 4)
        self.up2, self.dec2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2), self._block(c * 4, c * 2)
        self.up1, self.dec1 = nn.ConvTranspose3d(c * 2, c, 2, 2), self._block(c * 2, c)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """One U-Net level: two padded 3x3x3 convolutions, each group-normalised and SiLU-activated."""
        groups = min(_POOL_MULTIPLE, out_channels)
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    @override
    def forward(self, volume: Float[Tensor, "b c z y x"]) -> Float[Tensor, "b one z y x"]:
        """The centre-heatmap logit volume for one already-normalised, already-pooled input."""
        e1 = self.enc1(volume)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    @classmethod
    def from_pack(cls, pack: Path, map_location: str = "cpu") -> tuple["CenterPrior", CenterPriorRecipe]:
        """Load the published DeepCenter checkpoint (`best.pt` state + `config.json` recipe)."""
        config = json.loads((pack / "config.json").read_text())
        model = cls(base_channels=int(config.get("base_channels", 24)))
        checkpoint = torch.load(pack / "best.pt", map_location=map_location, weights_only=False)
        state = {key.replace(_BLOCK_NEST, "."): value for key, value in checkpoint["model_state"].items()}
        model.load_state_dict(state)
        return model, CenterPriorRecipe.from_config(config)

    @torch.no_grad()
    def heatmaps(self, path: Path, recipe: CenterPriorRecipe, device: str) -> list[Float[np.ndarray, "z y x"]]:
        """Each frame's centre-probability heatmap on the pooled grid — the cacheable forward-once response."""
        self.eval()
        array = zarr.open_array(path / "0")
        return [
            self._frame_heatmap(np.asarray(array[timepoint], dtype=np.float32), recipe, device)
            for timepoint in range(int(array.shape[0]))
        ]

    def _frame_heatmap(
        self, frame: Float[np.ndarray, "z y x"], recipe: CenterPriorRecipe, device: str
    ) -> Float[np.ndarray, "z y x"]:
        """One frame's heatmap: preprocess, pad to the pooling multiple, forward, sigmoid, crop back."""
        image = recipe.pooled_and_normalised(frame)
        padded, shape = self._pad_to_multiple(image)
        tensor = torch.from_numpy(padded[None, None]).to(device=device, dtype=torch.float32)
        logits = torch.sigmoid(self(tensor))[0, 0].detach().cpu().numpy().astype(np.float32)
        return logits[: shape[0], : shape[1], : shape[2]]

    @staticmethod
    def _pad_to_multiple(
        image: Float[np.ndarray, "z y x"],
    ) -> tuple[Float[np.ndarray, "z y x"], tuple[int, int, int]]:
        """Zero-pad each dim up to a multiple of the pooling factor so the U-Net's skip-concats align."""
        shape = image.shape
        pads = [(-size) % _POOL_MULTIPLE for size in shape]
        if not any(pads):
            return image, shape
        padded = np.pad(image, [(0, pad) for pad in pads], mode="constant")
        return padded.astype(np.float32, copy=False), shape


@dataclass(frozen=True)
class CenterPriorScorer:
    """A mounted centre prior plus its heatmap cache: forwards each video once, replays the veto after.

    The parallel of `DetectorScorer` for the veto family — the expensive forward is a pure function of
    (weights, video) and is cached by video key, so every stage's confirmation replays a cheap max-pool.
    """

    model: CenterPrior
    recipe: CenterPriorRecipe
    cache: ResponseCache
    device: str

    def veto(self, video_key: str, path: Path) -> CenterPriorVeto:
        """This video's heatmaps as a `CenterPriorVeto`, forwarding-and-caching on a first miss."""
        heatmaps = self.cache.responses(video_key, lambda: self.model.heatmaps(path, self.recipe, self.device))
        return CenterPriorVeto(heatmaps=tuple(heatmaps), recipe=self.recipe)

"""The temporal-U-Net detector strategy: the peak read-out over the learnable net (`celltrack.models`).

The pure network — backbone, detection head, forward, weight (de)serialisation, per-frame video read — is
the nn.Module in `celltrack.models.temporal_unet_detector`. This is the detection READ-OUT wrapped around it:
forward the whole video into per-voxel cellness logits, then recover cell centres as the suppressed local
maxima above a threshold. The read-out is the half that depends on `detectors.peaks` (the equality-NMS peak
extractor), which is exactly why it lives here and not with the net — the net imports no detector code, so
`detectors -> models` is the only direction and there is no cycle.

`TemporalUNetDetector` here is the net subclass that adds that read-out: the trainer optimises it, the fold
evaluator and the Kaggle kernel both call `.detections(...)`. Logit volumes are the cacheable, blend-friendly
half (`logit_volumes`); the peak read-out (`graph_from_volumes`) is the cheap half a threshold change replays.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.detectors.peaks import PeakExtractor
from celltrack.models.temporal_unet_detector import DetectorRecipe
from celltrack.models.temporal_unet_detector import TemporalUNetDetector as _TemporalUNetNet
from celltrack.precision import AutocastPolicy
from core.data.tracks import TrackGraph
from core.geometry import Spacing


class TemporalUNetDetector(_TemporalUNetNet):
    """The pure net plus the peak read-out — forward a video to cellness logits, then read centres out."""

    def detections(self, path: Path, threshold: float, recipe: DetectorRecipe, device: str) -> TrackGraph:
        """Detect cell centres in every frame of a video, returned as an edge-free `TrackGraph`.

        The forward pass and the peak read-out are split so the response can be cached (see
        `logit_volumes`): here they run back to back, the live path. Coordinates come back in the
        original full-resolution voxel grid (peaks scaled by the downsample), so the graph drops straight
        into a linker built for the video's native spacing.
        """
        volumes = self.logit_volumes(path, recipe, device)
        return self.graph_from_volumes(volumes, self.voxel_scale(path), threshold, recipe, device)

    def logit_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[Float[np.ndarray, "z y x"]]:
        """Each frame's per-voxel cellness **logit** on the downsampled grid — the cacheable response to blend.

        Logits, not probabilities, are the multi-seed blend space: they are unsaturated, so averaging seeds
        keeps the peak distinctions the equality-NMS reads. `probability_volumes` is the sigmoid of these.
        """
        return self._response(path, recipe, device, as_probability=False)

    def probability_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[Float[np.ndarray, "z y x"]]:
        """Each frame's per-voxel cellness probability — the single-seed cacheable response (sigmoid on device)."""
        return self._response(path, recipe, device, as_probability=True)

    @torch.no_grad()
    def _response(
        self, path: Path, recipe: DetectorRecipe, device: str, *, as_probability: bool
    ) -> list[Float[np.ndarray, "z y x"]]:
        """Forward the whole video into per-frame volumes, taking the sigmoid on the device before the host copy.

        The forward is the expensive, weights-and-video-determined half; a `ResponseCache` runs it once and
        every later read-out replays cheaply. Volumes are the downsampled shape (16x fewer voxels) — small.
        """
        self.eval()
        source = self._open_source(path)
        volumes: list[Float[np.ndarray, "z y x"]] = []
        for timepoint in range(int(source.array.shape[0])):
            logits = self._logits(self._read_frame(source, timepoint, recipe.downsample, device), recipe.tta)
            volumes.append((torch.sigmoid(logits) if as_probability else logits).cpu().numpy())
        return volumes

    @classmethod
    def graph_from_volumes(
        cls,
        volumes: list[Float[np.ndarray, "z y x"]],
        scale: tuple[float, float, float],
        threshold: float,
        recipe: DetectorRecipe,
        device: str,
    ) -> TrackGraph:
        """Read cell centres out of cached **logit** volumes — the cheap, replayable half of detection.

        Suppression runs on the logits, not their sigmoid: `sigmoid` saturates every confident logit (>~16)
        to an identical fp32 `1.0`, so the valley separating two touching nuclei collapses into the same flat
        plateau as their peaks and the connected-component labelling merges both cells into one centre. The
        logits keep that valley, so the reference read-out (`max_pool3d(logits)` equality, threshold on the
        sigmoid) recovers one centre per cell. The prob-space `threshold` becomes the equivalent logit floor.
        """
        extractor = cls._extractor(scale, recipe, device)
        floor = cls._logit_floor(threshold)
        stamped: list[np.ndarray] = []
        for timepoint, volume in enumerate(volumes):
            peaks = cls._capped_peaks(extractor, volume, floor, recipe.keep_per_frame)
            scaled = peaks * np.array(recipe.downsample)
            stamped.append(np.column_stack([np.full(len(scaled), timepoint), scaled]).astype(np.int64))
        nodes = np.concatenate(stamped) if stamped else np.empty((0, 4), dtype=np.int64)
        return TrackGraph(np.arange(len(nodes), dtype=np.int64), nodes, np.empty((0, 2), np.int64))

    def voxel_scale(self, path: Path) -> tuple[float, float, float]:
        """The video's `(z, y, x)` micrometre voxel scale — read cheaply from the OME metadata for the read-out."""
        return self._open_source(path).scale

    def _logits(self, frame: Float[Tensor, "z y x"], tta: bool) -> Float[Tensor, "z y x"]:  # noqa: FBT001
        """Detection logits for a frame, averaged over the flip-TTA ensemble when enabled.

        The forward runs at the device's accelerated precision and comes back fp32: the peak read-out compares
        a volume against its max-pool for equality, and half precision ties a nucleus to the valley beside it.
        """
        with AutocastPolicy.of(str(frame.device)):
            logits = self(frame)
            if tta:
                for dims in [(-1,), (-2,), (-2, -1)]:
                    logits = logits + self(frame.flip(dims)).flip(dims)
                logits = logits / 4
        return logits.float()

    @staticmethod
    def _extractor(scale: tuple[float, float, float], recipe: DetectorRecipe, device: str) -> PeakExtractor:
        """A peak extractor on the downsampled voxel grid — its NMS collapses a saturated plateau to one centre."""
        spacing = Spacing(
            z=scale[0] * recipe.downsample[0],
            y=scale[1] * recipe.downsample[1],
            x=scale[2] * recipe.downsample[2],
        )
        return PeakExtractor(spacing, scale_um=recipe.pool_kernel_um / 2.0, device=device)

    @staticmethod
    def _logit_floor(threshold: float) -> float:
        """The logit-space floor for a probability `threshold` — total over `[0, 1]` (0 keeps all, 1 keeps none)."""
        if threshold <= 0.0:
            return -math.inf
        if threshold >= 1.0:
            return math.inf
        return math.log(threshold / (1.0 - threshold))

    @staticmethod
    def _capped_peaks(
        extractor: PeakExtractor, logits: Float[np.ndarray, "z y x"], floor: float, keep: int
    ) -> Int[np.ndarray, "k 3"]:
        """Suppressed logit maxima above the logit `floor`, capped to the `keep` strongest — bounds the cost matrix."""
        coordinates, values = extractor.maxima(logits, floor=floor)
        if len(coordinates) > keep:
            coordinates = coordinates[np.argsort(values)[::-1][:keep]]
        return coordinates

"""The dual-seed detector as one cache-backed scorer: forward each seed once, blend, read peaks out.

Detection is two moves. The forward — the expensive, GPU, weights-and-video-determined half — runs once per
seed and is cached, so every later read-out replays cheaply off the host. The read-out — blend the seeds'
logits, suppress, threshold — is the cheap half a threshold change re-runs alone. `BlendDetectorScorer` is
that pairing: the frontier's dual-seed detector, no retraining, mounted for `CellTracker`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from celltrack.detectors.response_cache import ResponseStore
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph

# The published kernels guard each frame's dual-seed blend against a detection COLLAPSE: they count the primary
# seed's peaks and the blended peaks at a high detection threshold (0.96875, `cfg.det_threshold`, set from
# `BIOHUB_DET_THRESHOLD="0.96875"` at biohub-0-927-lb_code.py:17) and REVERT that frame to the primary detection
# when the blend retains under 90% of them (`minimum_retention = 0.90`, :1128-1187). The blend must not silently
# lose cells a single seed found; where it does, the safer primary is kept. Off unless `kernel_faithful` is set.
_KERNEL_DET_GUARD_THRESHOLD = 0.96875
_KERNEL_DET_GUARD_FRAC = 0.90


@dataclass(frozen=True)
class DetectionFusionOptions:
    """The two per-frame dual-seed fidelity switches, travelling as one so `nodes` keeps a config, not a flag list.

    Both default OFF, which is the byte-identical shipped read-out: `align_moments` is the calibrated fusion's
    detection half (rescale each secondary onto the primary's moments before the blend), `kernel_faithful` is the
    published kernel's per-frame retention guard (E6). They are one object because both describe the SAME thing —
    how faithfully this read-out reproduces the frontier's dual-seed detection — and both are set together.
    """

    align_moments: bool = False
    kernel_faithful: bool = False


@dataclass(frozen=True)
class BlendDetectorScorer:
    """Several trained detectors blended at the logit level — one scorer over an N-seed ensemble.

    Each seed forwards once into its own cache; the read-out averages their logit volumes (the honest,
    unsaturated blend space — near-1.0 probabilities lose the peak distinctions the NMS needs) and shares
    the peak read-out, which suppresses on those blended logits. Mounting a second public seed and averaging
    is the frontier's dual-seed detector — no retraining. The mean runs on the device; only the small blended
    logit volume returns to the host for the peak read-out.
    """

    # The std-ratio clamp for the calibrated detection blend, matched to `edges.SeedMomentAlignment`'s own: a
    # near-degenerate frame must not rescale the secondary by an arbitrary factor. Inlined rather than imported
    # because `detectors` sits below `edges` in the import layering — the formula is three lines, the layer is a rule.
    _STD_RATIO_MIN = 0.5
    _STD_RATIO_MAX = 2.0

    detectors: tuple[tuple[TemporalUNetDetector, ResponseStore], ...]
    recipe: DetectorRecipe
    device: str

    def nodes(
        self,
        video_key: str,
        path: Path,
        threshold: float,
        weights: tuple[float, ...] | None = None,
        *,
        fusion: DetectionFusionOptions | None = None,
    ) -> TrackGraph:
        """Blend every seed's cached logits on the device and read the peaks out as an edge-free graph.

        `weights` (summing to one, one per seed) tilts the logit blend toward a stronger seed; None is the equal
        mean the frontier ships. The blend is in logit space either way — probabilities saturate and lose peaks.

        `fusion` carries the two dual-seed fidelity switches (`DetectionFusionOptions`); None is the shipped,
        byte-identical read-out. `align_moments` puts each secondary seed's per-frame logit volume onto the
        primary's scale (z-score / clamped std-ratio) before the weighted sum — the detection half of the
        calibrated dual-seed fusion. `kernel_faithful` adds the published kernel's per-frame retention guard
        (E6): a frame whose blended high-threshold peak count collapses below 90% of the primary seed's reverts
        to the primary detection, so the blend cannot silently drop cells a single seed found.
        """
        fusion = fusion or DetectionFusionOptions()
        stacks = [self._cached_logits(video_key, path, detector, cache) for detector, cache in self.detectors]
        blended = [
            self._blend(frames, weights, align_moments=fusion.align_moments) for frames in zip(*stacks, strict=True)
        ]
        scale = self.detectors[0][0].voxel_scale(path)
        if fusion.kernel_faithful:
            blended = self._retained(stacks[0], blended, scale)
        return TemporalUNetDetector.graph_from_volumes(blended, scale, threshold, self.recipe, self.device)

    def _retained(
        self,
        primary: list[Float[np.ndarray, "z y x"]],
        blended: list[Float[np.ndarray, "z y x"]],
        scale: tuple[float, float, float],
    ) -> list[Float[np.ndarray, "z y x"]]:
        """The kernel's per-frame guard: keep the blend unless its high-threshold peak count collapsed vs primary."""
        primary_counts = self._peak_counts(primary, scale)
        blended_counts = self._peak_counts(blended, scale)
        floor = _KERNEL_DET_GUARD_FRAC * primary_counts
        collapsed = (primary_counts > 0) & (blended_counts < floor)
        return [primary[f] if collapsed[f] else blended[f] for f in range(len(blended))]

    def _peak_counts(
        self, frames: list[Float[np.ndarray, "z y x"]], scale: tuple[float, float, float]
    ) -> Float[np.ndarray, " f"]:
        """Detected-cell count per frame at the guard's high threshold, read off the shared peak extractor."""
        graph = TemporalUNetDetector.graph_from_volumes(
            frames, scale, _KERNEL_DET_GUARD_THRESHOLD, self.recipe, self.device
        )
        return np.bincount(graph.coordinates[:, 0], minlength=len(frames))

    def _cached_logits(
        self, video_key: str, path: Path, detector: TemporalUNetDetector, cache: ResponseStore
    ) -> list[Float[np.ndarray, "z y x"]]:
        return cache.responses(
            video_key,
            lambda: detector.logit_volumes(path, self.recipe, self.device),
            kind="logit",
            fingerprint=self.recipe.fingerprint(),
        )

    def _blend(
        self,
        frames: tuple[Float[np.ndarray, "z y x"], ...],
        weights: tuple[float, ...] | None,
        *,
        align_moments: bool = False,
    ) -> Float[np.ndarray, "z y x"]:
        """Combine the seeds' logits on the device — an equal mean, or a weighted sum when weights are given.

        With `align_moments` each seed after the first is rescaled onto seed 0's per-frame mean and spread before
        the combination, so an independently trained secondary contributes its ranking rather than its scale.
        """
        stacked = torch.as_tensor(np.stack(frames), device=self.device)
        if align_moments:
            stacked = self._aligned(stacked)
        if weights is None:
            return stacked.mean(dim=0).cpu().numpy()
        shape = (-1, *([1] * (stacked.ndim - 1)))
        tilt = torch.as_tensor(weights, device=self.device, dtype=stacked.dtype).reshape(shape)
        return (stacked * tilt).sum(dim=0).cpu().numpy()

    def _aligned(self, stacked: Float[torch.Tensor, "s z y x"]) -> Float[torch.Tensor, "s z y x"]:
        """Every seed after the first re-expressed on seed 0's logit moments — `(v - mean_k) * std_0/std_k + mean_0`."""
        reference = stacked[0]
        ref_mean, ref_std = reference.mean(), reference.std()
        floor = torch.finfo(stacked.dtype).eps
        aligned = [reference]
        for volume in stacked[1:]:
            ratio = (ref_std / volume.std().clamp(min=floor)).clamp(self._STD_RATIO_MIN, self._STD_RATIO_MAX)
            aligned.append((volume - volume.mean()) * ratio + ref_mean)
        return torch.stack(aligned)

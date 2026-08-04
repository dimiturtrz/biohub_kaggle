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

from celltrack.response_cache import ResponseCache
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph


@dataclass(frozen=True)
class BlendDetectorScorer:
    """Several trained detectors blended at the logit level — one scorer over an N-seed ensemble.

    Each seed forwards once into its own cache; the read-out averages their logit volumes (the honest,
    unsaturated blend space — near-1.0 probabilities lose the peak distinctions the NMS needs) and shares
    the peak read-out, which suppresses on those blended logits. Mounting a second public seed and averaging
    is the frontier's dual-seed detector — no retraining. The mean runs on the device; only the small blended
    logit volume returns to the host for the peak read-out.
    """

    detectors: tuple[tuple[TemporalUNetDetector, ResponseCache], ...]
    recipe: DetectorRecipe
    device: str

    def nodes(self, video_key: str, path: Path, threshold: float) -> TrackGraph:
        """Blend every seed's cached logits on the device and read the peaks out as an edge-free graph."""
        stacks = [self._cached_logits(video_key, path, detector, cache) for detector, cache in self.detectors]
        blended = [self._blend(frames) for frames in zip(*stacks, strict=True)]
        scale = self.detectors[0][0].voxel_scale(path)
        return TemporalUNetDetector.graph_from_volumes(blended, scale, threshold, self.recipe, self.device)

    def _cached_logits(
        self, video_key: str, path: Path, detector: TemporalUNetDetector, cache: ResponseCache
    ) -> list[Float[np.ndarray, "z y x"]]:
        return cache.responses(video_key, lambda: detector.logit_volumes(path, self.recipe, self.device))

    def _blend(self, frames: tuple[Float[np.ndarray, "z y x"], ...]) -> Float[np.ndarray, "z y x"]:
        """Mean the seeds' logits on the device — the unsaturated volume the equality-NMS suppresses on."""
        stacked = torch.as_tensor(np.stack(frames), device=self.device)
        return stacked.mean(dim=0).cpu().numpy()

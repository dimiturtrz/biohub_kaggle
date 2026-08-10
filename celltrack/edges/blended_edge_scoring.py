"""Several seeds' edge transformers, blended in logit space — the edge analogue of the detector seed blend.

The detector already averages two seeds' per-frame logits (probabilities saturate near 0/1 and blend to mush;
logits keep the seeds' disagreement). The edge transformer earns the same treatment: each seed scores every
gap from its own UNet features and transformer head, we take a convex combination of their pre-softmax logits,
and only then soft-max over the sources of each target. On the 4-movie proxy this lifts the assignment linker
from 0.9125→0.9146 (with the density gap-bridge, 0.9134→0.9154) at 0.8·seed1 + 0.2·seed2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from celltrack.models.edge_transformer import EdgeTransformerScorer, PrecomputedEdgeAffinity
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from core.data.tracks import TrackGraph


class BlendedEdgeTransformerScorer:
    """A convex blend of several `EdgeTransformerScorer` seeds in logit space, then softmax-over-sources.

    The gap node order (shared detector nodes) is identical across seeds, so each seed's `s×t` logit matrix
    aligns row-for-row and the weighted sum is well defined. Drop-in for `EdgeTransformerScorer` in the linker
    pipeline: `affinities` returns the same `PrecomputedEdgeAffinity` the assignment cost consumes.
    """

    def __init__(self, scorers: tuple[EdgeTransformerScorer, ...], weights: tuple[float, ...]) -> None:
        if len(scorers) != len(weights):
            raise ValueError(f"got {len(scorers)} scorers but {len(weights)} weights")
        self.scorers = scorers
        self.weights = weights

    @classmethod
    def from_packs(
        cls, packs: tuple[Path, ...], weights: tuple[float, ...], device: str = "cpu"
    ) -> BlendedEdgeTransformerScorer:
        """Mount one `EdgeTransformerScorer` per pack and pair each with its blend weight."""
        scorers = tuple(EdgeTransformerScorer.from_pack(pack, device) for pack in packs)
        return cls(scorers, weights)

    @torch.no_grad()
    def affinities(self, path: Path, detections: TrackGraph, device: str) -> PrecomputedEdgeAffinity:
        """The blended source→target probability matrices, one per scored frame gap of the video."""
        source = TemporalUNetDetector._open_source(path)  # noqa: SLF001
        timepoints = detections.timepoints()
        positions = detections.positions().astype(np.float32)
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for timepoint in np.unique(timepoints)[:-1].tolist():
            src_rows = np.flatnonzero(timepoints == timepoint)
            tgt_rows = np.flatnonzero(timepoints == timepoint + 1)
            if len(src_rows) == 0 or len(tgt_rows) == 0:
                continue
            src_pos, tgt_pos = positions[src_rows], positions[tgt_rows]
            weighted = torch.stack(
                [
                    weight * scorer._gap_logits(source, int(timepoint), src_pos, tgt_pos, device)  # noqa: SLF001
                    for scorer, weight in zip(self.scorers, self.weights, strict=True)
                ]
            )
            by_timepoint[int(timepoint)] = torch.softmax(weighted.sum(dim=0), dim=0).cpu().numpy()
        return PrecomputedEdgeAffinity(by_timepoint)

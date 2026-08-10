"""Several seeds' edge transformers, blended in logit space — the edge analogue of the detector seed blend.

The detector already averages two seeds' per-frame logits (probabilities saturate near 0/1 and blend to mush;
logits keep the seeds' disagreement). The edge transformer earns the same treatment: each seed scores every
gap from its own UNet features and transformer head, we take a convex combination of their pre-softmax logits,
and only then soft-max over the sources of each target. On the 4-movie proxy this lifts the assignment linker
from 0.9125→0.9146 (with the density gap-bridge, 0.9134→0.9154) at 0.8·seed1 + 0.2·seed2.

Optionally the blended probabilities are then fused with a second, reversed pass — see
`BlendedEdgeTransformerScorer.fuse`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer, PrecomputedEdgeAffinity
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from core.data.tracks import TrackGraph

_DENOMINATOR_FLOOR = 1e-12


class BlendedEdgeTransformerScorer:
    """A convex blend of several `EdgeTransformerScorer` seeds in logit space, then softmax-over-sources.

    The gap node order (shared detector nodes) is identical across seeds, so each seed's `s×t` logit matrix
    aligns row-for-row and the weighted sum is well defined. Drop-in for `EdgeTransformerScorer` in the linker
    pipeline: `affinities` returns the same `PrecomputedEdgeAffinity` the assignment cost consumes.

    With `bidirectional` the gap is additionally scored backwards and the two normalisations are fused by
    `fuse`. Fusion happens at the probability level, after the seed blend — the seeds still combine as logits.
    """

    def __init__(
        self,
        scorers: tuple[EdgeTransformerScorer, ...],
        weights: tuple[float, ...],
        *,
        bidirectional: bool = False,
    ) -> None:
        if len(scorers) != len(weights):
            raise ValueError(f"got {len(scorers)} scorers but {len(weights)} weights")
        self.scorers = scorers
        self.weights = weights
        self.bidirectional = bidirectional

    @classmethod
    def from_packs(
        cls,
        packs: tuple[Path, ...],
        weights: tuple[float, ...],
        device: str = "cpu",
        *,
        bidirectional: bool = False,
    ) -> BlendedEdgeTransformerScorer:
        """Mount one `EdgeTransformerScorer` per pack and pair each with its blend weight."""
        scorers = tuple(EdgeTransformerScorer.from_pack(pack, device) for pack in packs)
        return cls(scorers, weights, bidirectional=bidirectional)

    def with_bidirectional(self, *, bidirectional: bool) -> BlendedEdgeTransformerScorer:
        """The same mounted seeds under the other fusion setting — so a sweep re-points the knob, not the mount."""
        return BlendedEdgeTransformerScorer(self.scorers, self.weights, bidirectional=bidirectional)

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
            gap = EdgeGap(source, int(timepoint), positions[src_rows], positions[tgt_rows], device)
            by_timepoint[gap.timepoint] = self._gap_probabilities(gap).cpu().numpy()
        return PrecomputedEdgeAffinity(by_timepoint)

    def _gap_probabilities(self, gap: EdgeGap) -> Float[Tensor, "s t"]:
        """One gap's source→target probabilities — the forward normalisation, optionally fused with the reverse."""
        forward = torch.softmax(self._blended_logits(gap, reverse=False), dim=0)
        if not self.bidirectional:
            return forward
        reverse = torch.softmax(self._blended_logits(gap, reverse=True), dim=0).T
        return self.fuse(forward, reverse)

    def _blended_logits(self, gap: EdgeGap, *, reverse: bool) -> Float[Tensor, "early late"]:
        """The seeds' logits for one gap direction, convex-combined before any normalisation."""
        weighted = torch.stack(
            [
                weight * scorer._gap_logits(gap, reverse=reverse)  # noqa: SLF001
                for scorer, weight in zip(self.scorers, self.weights, strict=True)
            ]
        )
        return weighted.sum(dim=0)

    @staticmethod
    def fuse(forward: Float[Tensor, "s t"], reverse: Float[Tensor, "s t"]) -> Float[Tensor, "s t"]:
        """The harmonic mean of the two directions' probabilities — a pair scores high only if both agree.

        `forward` is normalised over the sources of each target (who is this cell's parent?); `reverse` is the
        same gap scored with the temporal pair swapped and transposed back, so it is normalised over the targets
        of each source (which successor does this cell claim?). They are different questions of the same weights.

        Our dense mislinks are affinity-inverted — a confident wrong NEAR neighbour (P 0.686) beats the true far
        successor (P 0.134). Such a neighbour tends to win forwards and lose backwards, because it already has a
        better parent of its own. The harmonic mean is dominated by the smaller of its two arguments, so it
        demands mutual consistency; an arithmetic mean would let the one confident direction carry the disagreed
        pair through at half its strength, which is exactly the failure mode.

        Zero-safe by construction: the denominator is floored, and a pair both directions call impossible fuses
        to 0 rather than 0/0. Flooring rather than `torch.where`-ing keeps a single expression, and `where`
        would evaluate the NaN branch anyway.
        """
        total = forward + reverse
        return 2.0 * forward * reverse / total.clamp(min=_DENOMINATOR_FLOOR)

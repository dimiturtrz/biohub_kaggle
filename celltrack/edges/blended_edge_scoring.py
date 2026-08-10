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

from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer, PrecomputedEdgeAffinity, video_gaps
from celltrack.models.prior_velocity import GapHistory
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
        """The blended source→target probability matrices, one per scored frame gap of the video.

        Gaps arrive in ascending time so each seed can carry its own `GapHistory` forward — the prior-velocity
        state a widened head reads. Per SEED, not per blend: the velocity a head consumes has to be the one its
        own weights produced (that is what its trainer fed it), and two seeds need not even agree on whether
        they were widened for the feature at all. A blend of unwidened seeds carries empty histories and is
        byte-identical to what this computed before the feature existed.
        """
        histories = tuple(GapHistory() for _ in self.scorers)
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for gap in video_gaps(path, detections, device):
            probabilities, histories = self._gap_probabilities(gap, histories)
            by_timepoint[gap.timepoint] = probabilities.cpu().numpy()
        return PrecomputedEdgeAffinity(by_timepoint)

    def _gap_probabilities(
        self, gap: EdgeGap, histories: tuple[GapHistory, ...]
    ) -> tuple[Float[Tensor, "s t"], tuple[GapHistory, ...]]:
        """One gap's probabilities and each seed's carried history — forward, optionally fused with the reverse."""
        blended, carried = self._forward_logits(gap, histories)
        forward = torch.softmax(blended, dim=0)
        if not self.bidirectional:
            return forward, carried
        reverse = torch.softmax(self._reverse_logits(gap), dim=0).T
        return self.fuse(forward, reverse), carried

    def _forward_logits(
        self, gap: EdgeGap, histories: tuple[GapHistory, ...]
    ) -> tuple[Float[Tensor, "s t"], tuple[GapHistory, ...]]:
        """The seeds' forward logits convex-combined, plus the history each seed hands the next gap."""
        scored = [
            scorer._seed_logits(gap, history)  # noqa: SLF001
            for scorer, history in zip(self.scorers, histories, strict=True)
        ]
        weighted = torch.stack([weight * logits for weight, (logits, _) in zip(self.weights, scored, strict=True)])
        return weighted.sum(dim=0), tuple(history for _, history in scored)

    def _reverse_logits(self, gap: EdgeGap) -> Float[Tensor, "t s"]:
        """The seeds' logits for the gap scored with the temporal pair SWAPPED, combined before normalisation.

        The reverse direction carries no prior velocity — see `EdgeTransformerScorer._gap_logits` for why its
        sources have no non-circular history — so a widened head reads zeros here and no state crosses gaps.
        """
        weighted = torch.stack(
            [
                weight * scorer._gap_logits(gap, reverse=True)  # noqa: SLF001
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

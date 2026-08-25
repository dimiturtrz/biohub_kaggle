"""Putting two edge seeds' logits on ONE scale before they are blended — and reporting the scales themselves.

`BlendedEdgeTransformerScorer` sums the seeds' RAW logits with fixed weights and soft-maxes the sum. Two
independently trained heads share no logit SCALE, and in a blend-then-softmax that is not a nuisance: a scale
difference IS a temperature difference, so the sharper-scaled seed dominates every target's distribution
whatever weight it carries. Down-weighting it compensates in the wrong units and throws away its ranking,
which is the part that was worth having.

Our shipped 0.8/0.2 therefore admits two readings — seed 1's head is genuinely better, or the scales differ
and 0.8/0.2 is a units fix wearing quality's clothes. This module supplies both halves of the answer: the
alignment (opt-in, default off) and `SeedMomentAlignment.moments`, the measurement
`BlendedEdgeTransformerScorer.seed_logit_moments` reports per gap per seed so the question is settled by
reading the numbers rather than inferred from the shape of a sweep. Under the second reading, aligning moves
the best blend toward even AND does not cost score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class LogitMoments:
    """One seed's logit mean and spread over one gap's whole candidate matrix — the scale a blend has to share."""

    mean: float
    std: float


class SeedMomentAlignment:
    """Rescales a secondary seed's logits onto a reference seed's moments: `(l - mean_k) * std_0/std_k + mean_0`.

    Moments are taken PER GAP, over that gap's whole `s x t` matrix, not once per video. Three reasons, in
    order of weight. The consumer is per-gap — the softmax that turns these logits into probabilities is taken
    inside one gap, so the temperature that matters is the one that gap presents. The producer is per-gap too:
    the blend is a streaming pass over gaps in ascending time (each seed carries its own history forward), so a
    video-global moment would need either a second full pass or a running estimate whose value depends on how
    far through the video the gap sits — a different transform for the same inputs. And gap difficulty varies
    with cell count and crowding, which moves both seeds' spread together; a per-gap ratio divides that common
    factor out, while a global one bakes the video's average difficulty into every gap.

    The std ratio is clamped, as the frontier clamps it: an unclamped ratio lets a near-degenerate gap (one
    source, or a frame where one seed is uniformly unsure) rescale the other seed by an arbitrary factor, which
    is noise amplification dressed as calibration. Clamping keeps the correction inside the range a real scale
    difference lives in and turns a degenerate gap into a bounded no-op.
    """

    STD_RATIO_MIN = 0.5
    STD_RATIO_MAX = 2.0
    # The kernel-faithful per-column path floors each column's spread at a fixed epsilon before the ratio, as the
    # published kernel does (`clamp_min(1e-4)`), rather than at the dtype eps the global path uses — a
    # near-degenerate single-source column then rescales by a bounded factor instead of an arbitrary one.
    PER_COLUMN_STD_FLOOR = 1e-4

    @classmethod
    def align(cls, logits: Float[Tensor, "s t"], reference: Float[Tensor, "s t"]) -> Float[Tensor, "s t"]:
        """`logits` re-expressed in `reference`'s units — same ranking, `reference`'s centre and spread."""
        source, target = cls.moments(logits), cls.moments(reference)
        floor = torch.finfo(logits.dtype).eps
        ratio = min(max(target.std / max(source.std, floor), cls.STD_RATIO_MIN), cls.STD_RATIO_MAX)
        return (logits - source.mean) * ratio + target.mean

    @classmethod
    def align_per_column(cls, logits: Float[Tensor, "s t"], reference: Float[Tensor, "s t"]) -> Float[Tensor, "s t"]:
        """`logits` re-expressed in `reference`'s units PER TARGET COLUMN — the published kernel's calibration.

        The global `align` shares one scalar mean/std over the whole `s x t` matrix; the frontier kernel instead
        z-scores each TARGET column by its OWN moments over the sources (`mean/std(dim=source, keepdim=True)`),
        so a target where the primary is sharp and one where it is flat are each put on the primary's local
        scale rather than on the video-gap's average. Reduction is over the source axis (`dim=0`), matching the
        kernel's `dim=1` on its `(1, n_src, n_tgt)` tensor. The std ratio is clamped exactly as the global path.
        """
        source_mean = logits.mean(dim=0, keepdim=True)
        target_mean = reference.mean(dim=0, keepdim=True)
        source_std = logits.std(dim=0, correction=0, keepdim=True).clamp_min(cls.PER_COLUMN_STD_FLOOR)
        target_std = reference.std(dim=0, correction=0, keepdim=True).clamp_min(cls.PER_COLUMN_STD_FLOOR)
        ratio = (target_std / source_std).clamp(cls.STD_RATIO_MIN, cls.STD_RATIO_MAX)
        return (logits - source_mean) * ratio + target_mean

    @staticmethod
    def moments(logits: Float[Tensor, "s t"]) -> LogitMoments:
        """The logits' mean and POPULATION standard deviation — the spread of the numbers in hand.

        Population (`correction=0`) rather than a sample estimate: we are describing this gap's matrix, not
        estimating a wider distribution's parameter from it, and a one-candidate gap has an honest population
        spread of zero where its sample spread is undefined. The clamp then absorbs that zero instead of a NaN
        propagating into every probability downstream.
        """
        return LogitMoments(mean=float(logits.mean()), std=float(logits.std(correction=0)))

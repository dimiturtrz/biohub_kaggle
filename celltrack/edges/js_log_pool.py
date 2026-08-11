"""Fusing several views of the same frame gap by a reliability-weighted LOGARITHMIC opinion pool.

An arithmetic mean of several views' association probabilities is a soft OR: one optimistic view is enough to
carry a pair over the linker's threshold, so the candidate graph inflates with edges only one presentation of
the scene ever believed. A logarithmic pool — the weighted GEOMETRIC mean of the distributions — is a soft
AND: a pair survives only if every view that is being listened to grants it mass, and one view's confident
outlier is pulled down rather than averaged in.

Which views are listened to is not a free parameter. Each view's Jensen-Shannon divergence from the consensus
distribution, divided by the MEDIAN of those divergences, gives a scale-free measure of how far out of line it
is; `w = 1/(1 + JS / median JS)` turns that into a weight with no constant to tune — the median is the unit.
Views that agree get identical weights whatever their common divergence happens to be.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


class JensenShannonLogPool:
    """A logarithmic opinion pool over several views' association logits, weighted by each view's JS reliability.

    Inputs are per-view logit matrices stacked on a leading view axis with the IDENTITY view first; the pool
    normalises them over the sources of each target (the head's own normalisation) before fusing, so it pools
    distributions rather than unnormalised scores.
    """

    @classmethod
    def pool(cls, view_logits: Float[Tensor, "v s t"]) -> Float[Tensor, "s t"]:
        """The views fused into one logit matrix on the identity view's scale.

        Two properties make this a drop-in rather than a re-tune. The weights are CONVEX (they sum to one over
        views), so the pooled log-probabilities carry one view's temperature instead of `v` views' worth of
        sharpening. And the result is re-centred per target onto the identity view's logit mean, restoring the
        additive offset a log-probability drops. Together they mean identical views pool back to the identity
        view exactly, and a threshold tuned on the single-view path reads the same here.
        """
        log_probabilities = torch.log_softmax(view_logits, dim=1)
        weights = cls.reliability(log_probabilities.exp())
        pooled = (weights.unsqueeze(1) * log_probabilities).sum(dim=0)
        return pooled - pooled.mean(dim=0, keepdim=True) + view_logits[0].mean(dim=0, keepdim=True)

    @classmethod
    def reliability(cls, distributions: Float[Tensor, "v s t"]) -> Float[Tensor, "v t"]:
        """Each view's convex weight for each target: `1/(1 + JS/median JS)`, normalised to sum to one.

        The divergence is measured against the views' arithmetic consensus, per target, so a view is judged on
        the question the pool is about to answer rather than on the gap as a whole. Views that all agree carry
        a zero divergence and a zero median, which the floor turns into a zero ratio and hence exactly uniform
        weights — the unanimous case reduces to a plain geometric mean with nothing to tune.
        """
        floor = torch.finfo(distributions.dtype).eps
        consensus = distributions.mean(dim=0)
        midpoint = 0.5 * (distributions + consensus)
        divergence = 0.5 * (
            cls._relative_entropy(distributions, midpoint, floor) + cls._relative_entropy(consensus, midpoint, floor)
        )
        ratio = divergence / divergence.median(dim=0).values.clamp(min=floor)
        weights = 1.0 / (1.0 + ratio)
        return weights / weights.sum(dim=0, keepdim=True)

    @staticmethod
    def _relative_entropy(
        left: Float[Tensor, "*v s t"], right: Float[Tensor, "v s t"], floor: float
    ) -> Float[Tensor, "v t"]:
        """`KL(left || right)` summed over the source axis, floored so an underflowed zero contributes zero."""
        return (left * ((left + floor).log() - (right + floor).log())).sum(dim=-2)

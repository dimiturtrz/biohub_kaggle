"""The frontier's margin-gated calibrated dual-seed edge fusion: blend a secondary seed in only where it helps.

`CalibratedFusion` is a pure, model-free function of two seeds' per-gap logit matrices — the published 0.926
kernels' edge lever — and `CalibratedFusionOptions` carries the frontier defaults it runs under. Kept beside
its producer and imported by the blend scorer, so the seed-combination math is unit-testable on CPU without
mounting a head.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from celltrack.edges.seed_moment_alignment import SeedMomentAlignment

# A margin is the gap between the top TWO parents — below that count the column has no ambiguity to gate on.
_MIN_SOURCES_FOR_MARGIN = 2


class CalibratedFusionOptions(BaseModel):
    """The frontier's #1 edge lever: a margin-gated, calibrated blend of a SECOND independent seed into the primary.

    Two public 0.926 kernels fuse a second, independently-seeded head into the primary the SAME way — not the
    convex logit sum this module's ordinary blend takes, but a GATED one. The secondary's logits are z-score /
    std-ratio calibrated onto the primary's scale (`SeedMomentAlignment`), then mixed in at `blend_w` ONLY on the
    targets where the primary is genuinely unsure AND the secondary independently agrees on the parent — the
    `low_margin_consensus` gate: the primary's top-2 parent-softmax margin is below `margin_thresh` and both heads
    pick the same source. Everywhere the primary is already decided, it is left untouched.

    A per-frame retention guard reverts the whole frame to primary-only if the blend collapses that frame's
    candidate count (edges above `cand_thresh`) below `retention_frac` of the primary's — the anti-collapse the
    kernels ship. Default OFF (the field on `EdgeBlendOptions` is None); the defaults here ARE the frontier's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin_thresh: float = 0.35
    blend_w: float = 0.15
    cand_thresh: float = 0.48
    retention_frac: float = 0.90
    # OFF by default: the variant path (a scalar-global calibration, a BINARY full-weight gate across the whole
    # sub-margin band, and the edge-level retention guard). ON reproduces the published 0.926 kernel's fusion
    # BYTE-FOR-BYTE where it lives — per-target-column z-score calibration (`SeedMomentAlignment.align_per_column`)
    # and a CONTINUOUS per-column ramp `blend_w * clamp((margin_thresh - margin)/margin_thresh, 0, 1)`. The
    # kernel's retention guard is a DETECTION-stage mechanism (it counts detected-cell peaks, not edges), so it
    # is NOT part of this edge fusion; faithful mode therefore drops the edge-level guard rather than approximate
    # it in the wrong stage. When False every branch below is the variant's, so the matrix is unchanged.
    kernel_faithful: bool = False


class CalibratedFusion:
    """Margin-gated calibrated dual-seed edge fusion: blend a secondary seed into the primary only where it helps.

    A pure function of two seeds' per-gap logit matrices (primary first, secondary second), each `[s, t]` —
    sources down the rows, targets across the columns, the softmax-over-sources convention the whole edge path
    uses. Stateless and model-free, so the whole gate is unit-testable on CPU without mounting a head.
    """

    def __init__(self, options: CalibratedFusionOptions) -> None:
        self.options = options

    def fuse(self, primary: Float[Tensor, "s t"], secondary: Float[Tensor, "s t"]) -> Float[Tensor, "s t"]:
        """The fused source→target probabilities for one gap — primary everywhere the gate does not fire.

        The secondary is calibrated onto the primary's scale FIRST (a raw logit sum of two independently trained
        heads mixes temperatures, not opinions — the same reason the ordinary blend has `align_seed_moments`),
        then mixed in at `blend_w` on the gated target columns only. The retention guard is the last word: a
        frame whose blend starves its own candidate count reverts whole to the primary.
        """
        aligned = self._aligned_secondary(primary, secondary)
        primary_probabilities = torch.softmax(primary, dim=0)
        if self.options.kernel_faithful:
            return self._faithful_fuse(primary, primary_probabilities, aligned)
        gate = self._low_margin_consensus(primary, primary_probabilities, aligned)
        weight = self.options.blend_w
        blended_logits = torch.where(gate.unsqueeze(0), (1.0 - weight) * primary + weight * aligned, primary)
        blended = torch.softmax(blended_logits, dim=0)
        return blended if self._retained(primary_probabilities, blended) else primary_probabilities

    def _aligned_secondary(
        self, primary: Float[Tensor, "s t"], secondary: Float[Tensor, "s t"]
    ) -> Float[Tensor, "s t"]:
        """The secondary on the primary's scale — per-target-column in faithful mode, scalar-global in the variant."""
        if self.options.kernel_faithful:
            return SeedMomentAlignment.align_per_column(secondary, primary)
        return SeedMomentAlignment.align(secondary, primary)

    def _faithful_fuse(
        self,
        primary_logits: Float[Tensor, "s t"],
        primary_probabilities: Float[Tensor, "s t"],
        aligned_secondary: Float[Tensor, "s t"],
    ) -> Float[Tensor, "s t"]:
        """The published kernel's fusion: a CONTINUOUS per-column ramp blend, no edge-level retention guard.

        The blend weight ramps `blend_w * clamp((margin_thresh - margin)/margin_thresh, 0, 1)` — ~0 as the
        primary's top-2 parent margin approaches `margin_thresh`, reaching `blend_w` only as the margin goes to
        zero — and is zeroed on every target where the two heads pick different parents. A single-source column
        (margin 1.0 by construction) rides the ramp to weight 0, matching the kernel's `n_src >= 2` bail. The
        weight is per column, broadcast over the sources, and the mix is `(1 - w) * primary + w * secondary`.
        """
        weight = self._faithful_blend_weight(primary_logits, primary_probabilities, aligned_secondary)
        blended_logits = (1.0 - weight) * primary_logits + weight * aligned_secondary
        return torch.softmax(blended_logits, dim=0)

    def _faithful_blend_weight(
        self,
        primary_logits: Float[Tensor, "s t"],
        primary_probabilities: Float[Tensor, "s t"],
        aligned_secondary: Float[Tensor, "s t"],
    ) -> Float[Tensor, "1 t"]:
        """The per-target continuous ramp weight, zeroed where the two heads disagree on the parent — kernel-faithful.

        Broadcast-shaped `[1, t]` so `(1 - w) * primary + w * secondary` mixes each column at its own weight.
        """
        margin = self._top2_margin(primary_probabilities)
        threshold = self.options.margin_thresh
        uncertainty = ((threshold - margin) / threshold).clamp(0.0, 1.0)
        weight = self.options.blend_w * uncertainty
        same_parent = primary_logits.argmax(dim=0) == aligned_secondary.argmax(dim=0)
        weight = torch.where(same_parent, weight, torch.zeros_like(weight))
        return weight.unsqueeze(0)

    def _low_margin_consensus(
        self,
        primary_logits: Float[Tensor, "s t"],
        primary_probabilities: Float[Tensor, "s t"],
        aligned_secondary: Float[Tensor, "s t"],
    ) -> Float[Tensor, " t"]:
        """The per-target gate: the primary is unsure (top-2 margin below threshold) AND both heads agree on the parent.

        Same-parent is argmax over sources; taken on the logits, which share the softmax's per-column ranking, so
        the check needs no second normalisation. A target with a single candidate source has no margin to be
        unsure about, so the gate is off there by construction (see `_top2_margin`).
        """
        low_margin = self._top2_margin(primary_probabilities) < self.options.margin_thresh
        same_parent = primary_logits.argmax(dim=0) == aligned_secondary.argmax(dim=0)
        return low_margin & same_parent

    @staticmethod
    def _top2_margin(probabilities: Float[Tensor, "s t"]) -> Float[Tensor, " t"]:
        """Each target column's gap between its two most probable parents — 1.0 where it has fewer than two."""
        if probabilities.shape[0] < _MIN_SOURCES_FOR_MARGIN:
            return probabilities.new_ones(probabilities.shape[1])
        top2 = torch.topk(probabilities, k=2, dim=0).values
        return top2[0] - top2[1]

    def _retained(self, primary: Float[Tensor, "s t"], blended: Float[Tensor, "s t"]) -> bool:
        """Whether the blend kept enough candidates to stand — the anti-collapse revert-to-primary guard.

        A candidate is an edge above `cand_thresh`; a frame whose blend drops below `retention_frac` of the
        primary's candidate count is a frame the fusion made worse, so the caller keeps the primary there.
        """
        primary_candidates = float((primary > self.options.cand_thresh).sum())
        blended_candidates = float((blended > self.options.cand_thresh).sum())
        return blended_candidates >= self.options.retention_frac * primary_candidates

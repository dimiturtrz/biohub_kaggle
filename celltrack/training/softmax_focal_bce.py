"""Focal BCE over source-softmaxed logits — the link association objective, named by its form, not its use.

`SoftmaxFocalBCE` soft-maxes the source→target logits across the source axis (so each target has one parent),
then applies focal binary cross-entropy over the rows and columns the annotation touches. The edge finetuner
and the joint trainer both pick it for links; neither owns it.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

_FOCAL_POWER = 2.0


class SoftmaxFocalBCE:
    """Focal BCE over logits soft-maxed across the source axis — the frontier link loss (one parent per target)."""

    @staticmethod
    def of(logits: Float[Tensor, "s t"], target: Float[Tensor, "s t"]) -> Float[Tensor, ""]:
        """Focal BCE over the rows and columns the target touches; the source-softmax enforces a single parent."""
        active = (target.sum(dim=1) > 0).unsqueeze(1) | (target.sum(dim=0) > 0).unsqueeze(0)
        if not active.any():
            return logits.new_zeros(())
        probability = torch.softmax(logits, dim=0)
        bce = F.binary_cross_entropy(probability, target, reduction="none")
        p_t = probability * target + (1 - probability) * (1 - target)
        return (((1 - p_t) ** _FOCAL_POWER) * bce)[active].mean()

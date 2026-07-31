"""The masked classification objective — the detector's centre voxels as a class-balanced decision.

The floor objective (`MaskedDetectionLoss`) regresses a Gaussian heatmap with squared error. That trains a
detector whose centres *wander* frame to frame: a soft bump has no sharp notion of "here and nowhere else",
so the argmax drifts with the noise, and a linker fed drifting centres pairs the wrong successors. The
winning public detectors instead treat detection as a per-voxel *classification* — is this the centre of a
cell — trained with binary cross-entropy on the logits the U-Net already emits, then read out at a high
probability threshold. Confident, few, stable centres are what a linker needs.

The class balance is the load-bearing detail. Centre voxels are a vanishing fraction of the volume, so an
unweighted BCE collapses to predicting background everywhere. Each positive is weighted `1/N_+` (the
positive class sums to one) and each negative `alpha/N_-` (the negative class sums to `alpha`), so the two
classes contribute on the same order with a deliberate lean towards recall — precision is recovered at
inference by the high readout threshold, not by drowning the positives here. The same supervision mask as
the heatmap target still gates the loss: ambiguous bright regions that may hide un-annotated cells carry no
gradient.
"""

import math
from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn import functional

# A voxel is a positive (a cell centre) when the Gaussian target has not yet decayed past one standard
# deviation — the same "near a centre" boundary the supervision mask is built from, so labels and mask agree.
_POSITIVE_LABEL = math.exp(-0.5)

# The negative class sums to this against the positive class's one: negatives outnumber positives by orders
# of magnitude, so equal per-class mass would swamp the centres; a light negative lean trades a permissive
# detector (recovered by the inference threshold) for one that never learns to fire.
_DEFAULT_NEGATIVE_WEIGHT = 0.01


class MaskedBCEClassificationLoss(torch.nn.Module):
    """Class-balanced binary cross-entropy over the supervised voxels, on the U-Net's raw logits."""

    def __init__(self, negative_weight: float = _DEFAULT_NEGATIVE_WEIGHT) -> None:
        super().__init__()
        self._negative_weight = negative_weight

    @override
    def forward(
        self,
        prediction: Float[Tensor, "b z y x"],
        target: Float[Tensor, "b z y x"],
        mask: Bool[Tensor, "b z y x"],
    ) -> Float[Tensor, ""]:
        """Class-balanced BCE-with-logits; positives weigh `1/N_+`, negatives `alpha/N_-`, mask gates both."""
        positive = (target > _POSITIVE_LABEL) & mask
        negative = mask & ~positive
        positive_count = torch.clamp(positive.sum(), min=1.0)
        negative_count = torch.clamp(negative.sum(), min=1.0)
        weight = positive / positive_count + self._negative_weight * negative / negative_count
        labels = positive.to(prediction.dtype)
        per_voxel = functional.binary_cross_entropy_with_logits(prediction, labels, reduction="none")
        return (weight * per_voxel).sum()

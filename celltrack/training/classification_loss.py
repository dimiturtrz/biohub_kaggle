"""The masked classification objective — the detector's centre voxels as a class-balanced decision.

The floor objective (`MaskedDetectionLoss`) regresses a Gaussian heatmap with squared error. That trains a
detector whose centres *wander* frame to frame: a soft bump has no sharp notion of "here and nowhere else",
so the argmax drifts with the noise, and a linker fed drifting centres pairs the wrong successors. The
winning public detectors instead treat detection as a per-voxel *classification* — is this the centre of a
cell — trained with binary cross-entropy on the logits the U-Net already emits, then read out at a high
probability threshold. Confident, few, stable centres are what a linker needs.

The class balance is the load-bearing detail, and it is *derived*, not tuned. Centre voxels are a vanishing
fraction of the volume, so an unweighted BCE collapses to predicting background everywhere; but a fixed
positive/negative weight is a magic number that assumes a composition every crop violates. Instead each
class's total loss mass is normalised to one *per batch* — every positive weighs `1/N_+`, every negative
`1/N_-`, both counted in that batch — so the two classes always contribute equally whatever the crop
happened to contain. (A single global weight, in contrast, saturated the detector: give the negatives a
hundredth of the mass and it learns to fire at probability one almost everywhere.) The same supervision
mask as the heatmap target still gates the loss: ambiguous bright regions that may hide un-annotated cells
carry no gradient.
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


class MaskedBCEClassificationLoss(torch.nn.Module):
    """Binary cross-entropy over the supervised voxels, each class's per-batch loss mass normalised to one."""

    @override
    def forward(
        self,
        prediction: Float[Tensor, "b z y x"],
        target: Float[Tensor, "b z y x"],
        mask: Bool[Tensor, "b z y x"],
    ) -> Float[Tensor, ""]:
        """Class-balanced BCE-with-logits; positives weigh `1/N_+`, negatives `1/N_-`, both from this batch."""
        positive = (target > _POSITIVE_LABEL) & mask
        negative = mask & ~positive
        weight = positive / torch.clamp(positive.sum(), min=1.0) + negative / torch.clamp(negative.sum(), min=1.0)
        labels = positive.to(prediction.dtype)
        per_voxel = functional.binary_cross_entropy_with_logits(prediction, labels, reduction="none")
        return (weight * per_voxel).sum()

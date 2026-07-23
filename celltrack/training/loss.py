"""The masked detection loss — the mechanism that keeps unlabelled cells out of the objective.

The loss is an ordinary squared error between the predicted heatmap and the target, but weighted by the
supervision mask before it is reduced. Where the mask is zero the residual contributes nothing, so no
gradient flows back through those voxels: the ambiguous bright regions that may hide un-annotated cells
neither pull the prediction up nor push it down. This is the load-bearing property of the whole training
scheme, and it is exactly what the unit test pins.
"""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor


class MaskedDetectionLoss(torch.nn.Module):
    """Squared error over the supervised voxels only, averaged by the amount of supervision."""

    @override
    def forward(
        self,
        prediction: Float[Tensor, "b z y x"],
        target: Float[Tensor, "b z y x"],
        mask: Bool[Tensor, "b z y x"],
    ) -> Float[Tensor, ""]:
        """The mask-weighted mean squared error; voxels outside the mask carry no gradient."""
        weight = mask.to(prediction.dtype)
        squared_error = weight * (prediction - target) ** 2
        supervised = weight.sum()
        return squared_error.sum() / torch.clamp(supervised, min=1.0)

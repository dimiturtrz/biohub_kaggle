"""Class-balanced binary cross-entropy — the per-voxel detection objective, named by its form, not its use.

A loss is a general tool: `BalancedBCE` is BCE with per-frame count-normalised class weights (positives weigh
1/n_pos, negatives neg_weight/n_neg), so a crowded and a sparse frame contribute equally. The detector trainer
and the joint trainer both pick it for cell centres; neither owns it.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class BalancedBCE:
    """Binary cross-entropy with per-frame count-normalised class weights — positives 1/n_pos, negatives w/n_neg."""

    @staticmethod
    def of(logits: Float[Tensor, "b z y x"], centres: list[Int[Tensor, "n 3"]], neg_weight: float) -> Float[Tensor, ""]:
        """Each frame weighs its positive voxels 1/n_pos and its negatives neg_weight/n_neg, averaged over the batch.

        The batch mean keeps the loss (and thus the learning rate) at the same scale regardless of batch size.
        Positives are the voxels named by `centres`; out-of-bounds centres are dropped, an empty frame is defined.
        """
        target = torch.zeros_like(logits)
        weight = torch.empty_like(logits)
        bounds = torch.tensor(logits.shape[1:], device=logits.device)
        for index, coords in enumerate(centres):
            coords = coords.to(logits.device)
            inside = (coords >= 0).all(dim=1) & (coords < bounds).all(dim=1)
            kept = coords[inside]
            target[index, kept[:, 0], kept[:, 1], kept[:, 2]] = 1.0
            n_pos = target[index].sum().clamp(min=1)
            n_neg = (target[index].numel() - n_pos).clamp(min=1)
            weight[index] = torch.where(target[index] == 1.0, 1.0 / n_pos, neg_weight / n_neg)
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / logits.shape[0]

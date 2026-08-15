"""Class-balanced binary cross-entropy — the per-voxel detection objective, named by its form, not its use.

A loss is a general tool: `BalancedBCE` is BCE with per-frame count-normalised class weights (positives weigh
1/n_pos, negatives neg_weight/n_neg), so a crowded and a sparse frame contribute equally. The detector trainer
and the joint trainer both pick it for cell centres; neither owns it.

Our annotation covers ~1-2% of the cells in a frame, so a zero target calls ~98% of the REAL cells background.
FROM SCRATCH that is survivable, and the published pack reached 0.993 recall under it: the false-negative
pressure is diffuse, and the positives teach "cellness" faster than the mislabelled negatives unteach it.
FINE-TUNING a SATURATED detector inverts the balance — the model already fires on thousands of unannotated
true cells, and every one of them emits a confident "you are wrong" gradient aimed exactly at a correct
detection, so the better the detector the harder this loss pushes it down. `ignore_above` is the fix: a voxel
where the response is strong but no annotation exists is UNDECIDABLE from our labels (unannotated true cell
or false positive), and supervising it asserts the second. Such voxels are dropped from the negative term
entirely — zero loss and zero weight — and the negative normalisation counts only what is actually supervised,
so masking cannot silently rescale the objective. The property is self-balancing: as the detector sharpens,
its responses concentrate on cells it is right about, and the masked fraction falls back toward zero.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


class BalancedBCE:
    """Binary cross-entropy with per-frame count-normalised class weights — positives 1/n_pos, negatives w/n_neg."""

    @staticmethod
    def of(
        logits: Float[Tensor, "b z y x"],
        centres: list[Int[Tensor, "n 3"]],
        neg_weight: float,
        ignore_above: float | None = None,
    ) -> Float[Tensor, ""]:
        """Each frame weighs its positive voxels 1/n_pos and its negatives neg_weight/n_neg, averaged over the batch.

        The batch mean keeps the loss (and thus the learning rate) at the same scale regardless of batch size.
        Positives are the voxels named by `centres`; out-of-bounds centres are dropped, an empty frame is defined.

        `ignore_above` is the sigmoid response over which an UNANNOTATED voxel stops being supervised: it then
        contributes zero loss and zero weight, and the negative normaliser divides by the supervised count
        rather than the raw negative count. Annotated centres are never masked. `None` (the default) supervises
        every voxel, which is the objective of yesterday, unchanged. Trainers derive the threshold from the
        pipeline's own inference operating point, `TrackerConfig.threshold` (0.97) — the response at which the
        shipped tracker would already have called the voxel a cell — never from a sweep of this loss.
        """
        return BalancedBCE.per_frame(logits, centres, neg_weight, ignore_above).mean()

    @staticmethod
    def per_frame(
        logits: Float[Tensor, "b z y x"],
        centres: list[Int[Tensor, "n 3"]],
        neg_weight: float,
        ignore_above: float | None = None,
    ) -> Float[Tensor, " b"]:
        """Each frame's balanced BCE, ONE scalar per frame — what `of` means over.

        The batched training step weights each pair (its draw weight, its reliability multiplier) before summing,
        so it needs the per-frame vector rather than the batch mean; `of` is exactly this averaged. Same weights,
        same masking, same numbers — only the final reduction differs (sum-per-frame here, mean there).
        """
        target = BalancedBCE._positives(logits, centres)
        supervised = BalancedBCE._supervised(logits, target, ignore_above)
        weight = BalancedBCE._weights(target, supervised, neg_weight)
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none").sum(dim=(1, 2, 3))

    @staticmethod
    def ignored_fraction(
        logits: Float[Tensor, "b z y x"], centres: list[Int[Tensor, "n 3"]], ignore_above: float
    ) -> float:
        """Share of voxels the same rule would leave unsupervised — the masking's own diagnostic, one number.

        Reported rather than inferred because it going to zero as the model sharpens IS the self-balancing
        property: a detector whose confident responses are all annotated has nothing left to mask.
        """
        target = BalancedBCE._positives(logits, centres)
        return 1.0 - float(BalancedBCE._supervised(logits, target, ignore_above).to(torch.float32).mean())

    @staticmethod
    def _positives(logits: Float[Tensor, "b z y x"], centres: list[Int[Tensor, "n 3"]]) -> Float[Tensor, "b z y x"]:
        """The target map — one at each in-bounds annotated centre, zero everywhere else."""
        target = torch.zeros_like(logits)
        bounds = torch.tensor(logits.shape[1:], device=logits.device)
        for index, coords in enumerate(centres):
            on_device = coords.to(logits.device)
            inside = (on_device >= 0).all(dim=1) & (on_device < bounds).all(dim=1)
            kept = on_device[inside]
            target[index, kept[:, 0], kept[:, 1], kept[:, 2]] = 1.0
        return target

    @staticmethod
    def _supervised(
        logits: Float[Tensor, "b z y x"], target: Float[Tensor, "b z y x"], ignore_above: float | None
    ) -> Bool[Tensor, "b z y x"]:
        """Which voxels the loss may hold an opinion on — every one, unless a threshold excuses the undecidable."""
        annotated = target == 1.0
        if ignore_above is None:
            return torch.ones_like(annotated)
        return annotated | (logits.detach().sigmoid() <= ignore_above)

    @staticmethod
    def _weights(
        target: Float[Tensor, "b z y x"], supervised: Bool[Tensor, "b z y x"], neg_weight: float
    ) -> Float[Tensor, "b z y x"]:
        """Per-frame class weights over the SUPERVISED voxels — masking removes terms, it does not rescale them."""
        weight = torch.empty_like(target)
        for index in range(target.shape[0]):
            n_pos = target[index].sum().clamp(min=1)
            n_neg = (supervised[index].sum().to(n_pos.dtype) - n_pos).clamp(min=1)
            weight[index] = torch.where(target[index] == 1.0, 1.0 / n_pos, neg_weight / n_neg)
        return weight * supervised.to(weight.dtype)

"""Focal BCE over softmaxed link logits — the association objective, named by its form, not its use.

`SoftmaxFocalBCE` soft-maxes the source→target logits along a normalisation axis, then applies focal binary
cross-entropy over the rows and columns the annotation touches. The edge finetuner and the joint trainer both
pick it for links; neither owns it.

WHICH AXIS IS NORMALISED IS THE WHOLE DESIGN. Over SOURCES it states "each target has one parent" — the
frontier's form, and the one the inference scorer reads. That axis alone leaves two measured defects:

  * a pair carrying ONE source is degenerate. `softmax(dim=0)` of a single row is identically 1.0, so the
    gradient is exactly zero while the loss reads a constant 66.67 (a clamped BCE(1, 0) on every rival).
    28.8% of our real training pairs are that shape, and 61.7% carry two sources or fewer.
  * a source can only be discouraged from preferring a RIVAL TARGET when some other supervised source claims
    that rival's column. Every dense mislink we have measured picks an UNANNOTATED detection, which no source
    claims — so the axis the linker actually chooses along has never been supervised.

Normalising over TARGETS as well states "each source has one child", is a real distribution even for a single
source, and costs no parameter and no scale (a slack row would need one: our edge logits sit near -11, so a
zero-logit slack would take essentially all the mass).
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

_FOCAL_POWER = 2.0
# A softmax over one element is 1.0 whatever the logits — an axis this short constrains nothing.
_ONE_CANDIDATE = 1

SOURCE_AXIS = 0
"""Normalise across sources: one parent per target — the frontier's form, and what the scorer reads."""

TARGET_AXIS = 1
"""Normalise across targets: one child per source — the axis the linker chooses along."""


class SoftmaxFocalBCE:
    """Focal BCE over logits soft-maxed along one or both axes of the source→target matrix."""

    @staticmethod
    def of(
        logits: Float[Tensor, "s t"],
        target: Float[Tensor, "s t"],
        axes: Sequence[int] = (SOURCE_AXIS,),
        *,
        balanced: bool = False,
    ) -> Float[Tensor, ""]:
        """Mean focal BCE across `axes`, over the rows and columns the annotation touches.

        An axis of LENGTH ONE contributes NOTHING TO THE SUM AND STILL COUNTS IN THE DIVISOR. A softmax across
        a single element is identically 1.0 whatever the logits, so that term asserts nothing and carries no
        gradient — it only contributed a large constant (a clamped BCE(1, 0) on every rival), which is how
        28.8% of our pairs came to report a loss of 66.67 while learning nothing.

        Dividing by the axes ASKED FOR rather than by the ones that happened to bind is the part that is not
        bookkeeping. A pair whose target has one candidate parent carries strictly less evidence about
        association than one whose target has forty competing for it, so it should push proportionally less —
        and this is the only place that difference can be expressed without inventing a weight.

        It arrived by accident and is held on ONE arm: a version of this loss that averaged the live term with
        the dead constant halved the same gradients, and its checkpoint is the best own-weights result of the
        campaign (0.9278 held out against a 0.9060 init, mislinks 56 -> 43), while an arm that renormalised
        onto the surviving axis instead failed to beat its own initialisation. One seed, one window — the
        weighting is kept because it is also the form that needs no constant, not because a single run liked
        it.

        Computed in fp32 with autocast disabled: `binary_cross_entropy` runs on probabilities (not logits) and
        is unsafe under a bf16 autocast, so a joint step's autocast must not reach it.
        """
        active = (target.sum(dim=1) > 0).unsqueeze(1) | (target.sum(dim=0) > 0).unsqueeze(0)
        constraining = [axis for axis in axes if logits.shape[axis] > _ONE_CANDIDATE]
        if not active.any() or not constraining:
            return logits.new_zeros(())
        along = SoftmaxFocalBCE._balanced_along if balanced else SoftmaxFocalBCE._along
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scores = logits.float()
            truth = target.float()
            terms = [along(scores, truth, active, axis) for axis in constraining]
            return torch.stack(terms).sum() / len(axes)

    @staticmethod
    def batched(
        logits: Float[Tensor, "b s t"],
        target: Float[Tensor, "b s t"],
        masks: tuple[Bool[Tensor, "b s"], Bool[Tensor, "b t"]],
        axes: Sequence[int] = (SOURCE_AXIS,),
        *,
        balanced: bool = False,
    ) -> Float[Tensor, " b"]:
        """Per-pair focal BCE over a PADDED batch — element `b` equals `of` on that pair's unpadded `(s, t)`.

        The batched forward emits one padded `(B, _MAX_NODES, _MAX_NODES)` logit block so the whole GPU forward
        is a single static shape; this reads it without the per-pair Python loop that block replaced. The masks
        carry each pair's real node counts, and the reduction reproduces `of` exactly: a pad node contributes
        nothing (excluded from the active cells, and `-inf` before the softmax so it takes no probability), an
        axis with ONE real candidate contributes nothing to the numerator but still counts in the `/ len(axes)`
        divisor, and a pair with no constraining axis reads zero. The per-pair weighting (the draw weight, the
        reliability multiplier) is applied by the caller to this vector, exactly as it was to the loop's scalars.
        """
        source_mask, target_mask = masks
        valid = source_mask[:, :, None] & target_mask[:, None, :]  # (B, S, T) real cells
        touched = target * valid.float()
        active = ((touched.sum(dim=2) > 0)[:, :, None] | (touched.sum(dim=1) > 0)[:, None, :]) & valid
        real_along = {SOURCE_AXIS: source_mask.sum(dim=1), TARGET_AXIS: target_mask.sum(dim=1)}  # per-pair counts
        reduce = SoftmaxFocalBCE._balanced_along_batched if balanced else SoftmaxFocalBCE._along_batched
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scores, truth = logits.float(), target.float()
            total = scores.new_zeros(scores.shape[0])
            for axis in axes:
                keep = real_along[axis] > _ONE_CANDIDATE  # (B,) — a 1-candidate axis constrains nothing
                axis_mask = source_mask if axis == SOURCE_AXIS else target_mask
                term = reduce(scores, truth, active, axis, axis_mask)
                total = total + term * keep.float()
            return total / len(axes)

    @staticmethod
    def _along(
        scores: Float[Tensor, "s t"], truth: Float[Tensor, "s t"], active: Bool[Tensor, "s t"], axis: int
    ) -> Float[Tensor, ""]:
        """Focal BCE over the softmax along one axis, averaged over the active cells."""
        return SoftmaxFocalBCE._focal(scores, truth, axis)[active].mean()

    @staticmethod
    def _along_batched(
        scores: Float[Tensor, "b s t"],
        truth: Float[Tensor, "b s t"],
        active: Bool[Tensor, "b s t"],
        axis: int,
        axis_mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, " b"]:
        """Per-pair mean of the focal BCE over the active cells — the batched `_along`."""
        focal = SoftmaxFocalBCE._focal_batched(scores, truth, axis, axis_mask)
        active_f = active.float()
        return (focal * active_f).sum(dim=(1, 2)) / active_f.sum(dim=(1, 2)).clamp(min=1)

    @staticmethod
    def _balanced_along_batched(
        scores: Float[Tensor, "b s t"],
        truth: Float[Tensor, "b s t"],
        active: Bool[Tensor, "b s t"],
        axis: int,
        axis_mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, " b"]:
        """Per-pair, per-decision class-balanced mean — the batched `_balanced_along`."""
        focal = SoftmaxFocalBCE._focal_batched(scores, truth, axis, axis_mask)
        over = axis + 1  # (B, S, T): SOURCE_AXIS reduces dim 1, TARGET_AXIS dim 2
        positive = SoftmaxFocalBCE._class_mean(focal, (truth > 0) & active, over)
        negative = SoftmaxFocalBCE._class_mean(focal, (truth == 0) & active, over)
        decided = active.any(dim=over)  # (B, D)
        per_decision = ((positive + negative) * decided.float()).sum(dim=1)
        return per_decision / decided.float().sum(dim=1).clamp(min=1)

    @staticmethod
    def _focal_batched(
        scores: Float[Tensor, "b s t"], truth: Float[Tensor, "b s t"], axis: int, axis_mask: Bool[Tensor, "b n"]
    ) -> Float[Tensor, "b s t"]:
        """Per-cell focal BCE over the masked softmax along `axis` — pad nodes on that axis take no probability.

        Only the NORMALISED axis is masked to `-inf`, not every invalid cell: masking the whole valid block would
        make a pad-target column all `-inf` and its softmax `nan`. Masking the axis alone leaves each real column
        a distribution over the real sources (and each real row over the real targets), which is what `of` sees.
        """
        over = axis + 1
        along_pad = ~axis_mask[:, :, None] if axis == SOURCE_AXIS else ~axis_mask[:, None, :]
        probability = torch.softmax(scores.masked_fill(along_pad, float("-inf")), dim=over)
        bce = F.binary_cross_entropy(probability, truth, reduction="none")
        p_t = probability * truth + (1 - probability) * (1 - truth)
        return ((1 - p_t) ** _FOCAL_POWER) * bce

    @staticmethod
    def _balanced_along(
        scores: Float[Tensor, "s t"], truth: Float[Tensor, "s t"], active: Bool[Tensor, "s t"], axis: int
    ) -> Float[Tensor, ""]:
        """The same focal BCE, counted once per DECISION and class-balanced inside it.

        `_along` averages over active CELLS, which leaves the objective unbalanced twice over, and in the same
        two ways `BalancedBCE` already refuses for detection:

          * WITHIN a decision the single true candidate is outnumbered by its rivals — up to 38 to 1 on our
            detected corpus (targets/gap mean 8.3, max 39) — so the gradient is mostly "not that one" and only
            the focal power pushes back, implicitly and by an amount nobody chose.
          * ACROSS decisions a cell mean weighs each decision by HOW MANY ACTIVE CELLS it holds, so a decision
            is heard in proportion to its rival count rather than once. (The active mask is a union of touched
            rows and columns, so widths are often equal and this is the weaker of the two effects — the class
            imbalance above is the one that bites at every width.)

        Here each decision (a column when normalising over sources — "who is this target's parent") contributes
        the mean over its positives plus the mean over its rivals, exactly `BalancedBCE`'s positives-1/n_pos,
        negatives-1/n_neg rule read at the row level, and the decisions are then averaged. So one decision is
        one vote whatever its width, and inside it the true candidate carries the same total weight as the field.
        No constant enters: the balance is the two class counts, which the data supplies.
        """
        focal = SoftmaxFocalBCE._focal(scores, truth, axis)
        # Reduce ALONG the normalised axis: the softmax runs down it, so the competitors of one decision are
        # exactly the entries it spans, and what survives the reduction is one number per decision.
        positive = SoftmaxFocalBCE._class_mean(focal, (truth > 0) & active, axis)
        negative = SoftmaxFocalBCE._class_mean(focal, (truth == 0) & active, axis)
        decided = active.any(dim=axis)
        return (positive + negative)[decided].mean()

    @staticmethod
    def _focal(scores: Float[Tensor, "s t"], truth: Float[Tensor, "s t"], axis: int) -> Float[Tensor, "s t"]:
        """Per-cell focal BCE over the softmax along `axis` — the quantity both reductions differ only in reading."""
        probability = torch.softmax(scores, dim=axis)
        bce = F.binary_cross_entropy(probability, truth, reduction="none")
        p_t = probability * truth + (1 - probability) * (1 - truth)
        return ((1 - p_t) ** _FOCAL_POWER) * bce

    @staticmethod
    def _class_mean(focal: Float[Tensor, "s t"], mask: Bool[Tensor, "s t"], over: int) -> Float[Tensor, " d"]:
        """Each decision's mean over one class — zero where that class is empty, never a division by zero."""
        counted = mask.sum(dim=over)
        return (focal * mask).sum(dim=over) / counted.clamp(min=1)

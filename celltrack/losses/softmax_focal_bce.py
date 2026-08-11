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
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scores = logits.float()
            truth = target.float()
            terms = [SoftmaxFocalBCE._along(scores, truth, active, axis) for axis in constraining]
            return torch.stack(terms).sum() / len(axes)

    @staticmethod
    def _along(
        scores: Float[Tensor, "s t"], truth: Float[Tensor, "s t"], active: Bool[Tensor, "s t"], axis: int
    ) -> Float[Tensor, ""]:
        """Focal BCE over the softmax along one axis, averaged over the active cells."""
        probability = torch.softmax(scores, dim=axis)
        bce = F.binary_cross_entropy(probability, truth, reduction="none")
        p_t = probability * truth + (1 - probability) * (1 - truth)
        return (((1 - p_t) ** _FOCAL_POWER) * bce)[active].mean()

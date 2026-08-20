"""The link objective as a swappable strategy — how the source→target logits are turned into one scalar loss.

`SoftmaxFocalBCE` is the PRIMITIVE (focal BCE over a softmaxed axis); this module is the STRATEGY over it, so
the trainer picks a form by config instead of by an `if` at the call site. Every form takes the same pair of
inputs (a `(s, t)` logit block and its one-hot annotation) and returns the same shape (a scalar `per_pair`, a
`(b,)` vector `batched`), so `TrackingLoss` holds one `AssociationObjective` and never learns which it got.

Two forms exist. `SoftmaxFocal` is the parameter-free frontier objective — source-axis (optionally target-axis
too) focal BCE, exactly the two static calls it replaces, so a run that asks for it is byte-identical to
yesterday's. `SlackRow` is the parental-softmax form (Trackastra's `1 + Σexp`): it appends a "no source parents
this target" row so an unannotated rival's probability mass has somewhere to go OTHER than a real source it does
not belong to. The frontier writes that row's logit as the constant 0, valid because their logits are O(1); OURS
sit near -11 (a source-softmax is column-shift-invariant, so the absolute scale drifted low), where a fixed 0
would take essentially all the mass. So the slack logit here is ONE LEARNED SCALAR, calibrated on the first
batch to the mean real logit and free to track the scale as training moves it — a derived init, not a constant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, SoftmaxFocalBCE

_Masks = tuple[Bool[Tensor, "b s"], Bool[Tensor, "b t"]]


class AssociationObjective(nn.Module, ABC):
    """One link loss over the source→target logits — a scalar per pair, a vector per padded batch.

    An `nn.Module`, not a function, because a form may carry a trained weight (the slack scalar): the trainer
    folds its `.parameters()` into the optimiser and round-trips its `state_dict` in the snapshot, exactly as it
    does the contrastive head. A parameter-free form (`SoftmaxFocal`) holds none, so that plumbing is inert for it.
    """

    @abstractmethod
    def per_pair(self, logits: Float[Tensor, "s t"], target: Float[Tensor, "s t"]) -> Float[Tensor, ""]:
        """The link loss for ONE pair's unpadded `(s, t)` logits against its one-hot annotation."""

    @abstractmethod
    def batched(
        self, logits: Float[Tensor, "b s t"], target: Float[Tensor, "b s t"], masks: _Masks
    ) -> Float[Tensor, " b"]:
        """The link loss per pair over a PADDED batch — element `b` equals `per_pair` on that pair's real block."""

    @staticmethod
    def from_config(*, link_axes: Sequence[int], balanced_links: bool, slack_links: bool) -> AssociationObjective:
        """Pick the form the config asks for — the slack row when asked, else the plain (a)symmetric focal BCE.

        Takes primitives, not the `TrackingLossConfig`, so this module never imports the loss assembly that
        imports it. `slack_links` overrides the axes: the slack row is a statement about PARENTS, defined on the
        source axis alone, so a slack run normalises over sources whatever `symmetric_links` requested.
        """
        if slack_links:
            return SlackRow(balanced=balanced_links)
        return SoftmaxFocal(axes=tuple(link_axes), balanced=balanced_links)


class SoftmaxFocal(AssociationObjective):
    """The frontier objective, unchanged: focal BCE over the softmax along the asked axes. Holds no parameters."""

    def __init__(self, axes: Sequence[int], *, balanced: bool) -> None:
        super().__init__()
        self._axes = tuple(axes)
        self._balanced = balanced

    @override
    def per_pair(self, logits: Float[Tensor, "s t"], target: Float[Tensor, "s t"]) -> Float[Tensor, ""]:
        return SoftmaxFocalBCE.of(logits, target, self._axes, balanced=self._balanced)

    @override
    def batched(
        self, logits: Float[Tensor, "b s t"], target: Float[Tensor, "b s t"], masks: _Masks
    ) -> Float[Tensor, " b"]:
        return SoftmaxFocalBCE.batched(logits, target, masks, self._axes, balanced=self._balanced)


class SlackRow(AssociationObjective):
    """Parental softmax: a learned "no parent" row appended to the sources before the source-axis focal BCE.

    For each target column the softmax now runs over the real sources PLUS one slack source whose logit is a
    single learned scalar. A target with no true parent (every unannotated rival) can place its mass on the slack
    instead of being forced onto a real source — which is precisely the dense confusor, where the head ranks an
    unannotated near-neighbour above the true farther successor. It also un-degenerates the single-source pair:
    one real row softmaxes to 1.0 with zero gradient, but one real row PLUS the slack is a real two-way decision.

    The slack scalar is created eagerly (so the optimiser owns it from the first step) and CALIBRATED lazily to
    the mean real logit on the first batch — our logits sit near -11, and a fixed 0 (the frontier's constant)
    would swamp them. A registered flag makes the calibration one-shot, so a restored run keeps its trained value.
    """

    def __init__(self, *, balanced: bool) -> None:
        super().__init__()
        self._balanced = balanced
        self.slack_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("_calibrated", torch.zeros((), dtype=torch.bool))

    @override
    def per_pair(self, logits: Float[Tensor, "s t"], target: Float[Tensor, "s t"]) -> Float[Tensor, ""]:
        self._calibrate(logits.detach())
        _, width = logits.shape
        row = self.slack_logit.view(1, 1).expand(1, width)
        logits_aug = torch.cat([logits, row], dim=0)
        target_aug = torch.cat([target, target.new_zeros(1, width)], dim=0)
        return SoftmaxFocalBCE.of(logits_aug, target_aug, (SOURCE_AXIS,), balanced=self._balanced)

    @override
    def batched(
        self, logits: Float[Tensor, "b s t"], target: Float[Tensor, "b s t"], masks: _Masks
    ) -> Float[Tensor, " b"]:
        source_mask, target_mask = masks
        valid = source_mask[:, :, None] & target_mask[:, None, :]
        self._calibrate(logits.detach()[valid])
        batch, _, width = logits.shape
        row = self.slack_logit.view(1, 1, 1).expand(batch, 1, width)
        logits_aug = torch.cat([logits, row], dim=1)
        target_aug = torch.cat([target, target.new_zeros(batch, 1, width)], dim=1)
        source_mask_aug = torch.cat([source_mask, source_mask.new_ones(batch, 1)], dim=1)
        return SoftmaxFocalBCE.batched(
            logits_aug, target_aug, (source_mask_aug, target_mask), (SOURCE_AXIS,), balanced=self._balanced
        )

    def _calibrate(self, real_logits: Float[Tensor, "..."]) -> None:
        """Set the slack logit to the mean real logit ONCE — a derived neutral start, not a hand-picked constant.

        Runs under `no_grad` and fills the existing parameter in place, so the optimiser slot built around it is
        untouched; the flag is a buffer, so it rides the checkpoint and a resumed run never re-calibrates onto
        its own moved scale. A batch with no real cells (all-padded) is skipped — there is nothing to average.
        """
        calibrated = cast(Tensor, self._calibrated)
        if bool(calibrated) or real_logits.numel() == 0:
            return
        with torch.no_grad():
            self.slack_logit.fill_(float(real_logits.mean()))
            calibrated.fill_(True)  # noqa: FBT003 — the buffer's calibrated/not state, not a flag argument

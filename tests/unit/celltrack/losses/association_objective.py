"""Unit tests for the link-objective strategy — the factory, the frontier form's parity, and the slack row.

Equivalence classes: the factory's two branches (and the slack override of the axes); `SoftmaxFocal` reproducing
the primitive it wraps EXACTLY (so an unasked run is byte-identical); and `SlackRow`'s three claims — the scalar
is a real parameter calibrated once to the mean real logit, the slack un-degenerates the single-source pair that
the plain source axis leaves gradient-free, and `batched` equals `per_pair` on the same unpadded block.
"""

import torch

from celltrack.losses.association_objective import AssociationObjective, SlackRow, SoftmaxFocal
from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, TARGET_AXIS, SoftmaxFocalBCE


def _pair(sources: int, targets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A `(s, t)` logit block near our real operating scale (~-11) and a one-hot annotation on the diagonal."""
    torch.manual_seed(0)
    logits = torch.randn(sources, targets) - 11.0
    target = torch.zeros(sources, targets)
    for i in range(min(sources, targets)):
        target[i, i] = 1.0
    return logits, target


def test_from_config():
    """No slack -> the plain focal BCE over the asked axes; slack -> the slack row, source-axis whatever was asked."""
    plain = AssociationObjective.from_config(link_axes=(SOURCE_AXIS,), balanced_links=False, slack_links=False)
    assert isinstance(plain, SoftmaxFocal)

    slack = AssociationObjective.from_config(
        link_axes=(SOURCE_AXIS, TARGET_AXIS), balanced_links=False, slack_links=True
    )
    assert isinstance(slack, SlackRow)


def test_softmax_focal_matches_primitive():
    """The parameter-free form is exactly the two static calls it replaces — the unasked run stays yesterday's."""
    logits, target = _pair(4, 5)
    objective = SoftmaxFocal(axes=(SOURCE_AXIS,), balanced=False)

    assert torch.equal(objective.per_pair(logits, target), SoftmaxFocalBCE.of(logits, target, (SOURCE_AXIS,)))
    assert not list(objective.parameters())  # holds nothing, so the optimiser and clip are unchanged

    batch_logits = logits.unsqueeze(0)
    batch_target = target.unsqueeze(0)
    masks = (torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool))
    expected = SoftmaxFocalBCE.batched(batch_logits, batch_target, masks, (SOURCE_AXIS,))
    assert torch.equal(objective.batched(batch_logits, batch_target, masks), expected)


def test_slack_scalar_calibrates_once():
    """The slack logit is a real parameter, set to the mean real logit on the first call and never again."""
    logits, target = _pair(4, 5)
    objective = SlackRow(balanced=False)

    assert "slack_logit" in dict(objective.named_parameters())  # eager: the optimiser owns it from step one
    assert not bool(objective._calibrated)

    objective.per_pair(logits, target)
    assert bool(objective._calibrated)
    assert torch.isclose(objective.slack_logit.detach(), logits.mean(), atol=1e-5)

    moved = logits + 5.0  # a later, shifted batch must NOT re-calibrate — the scalar is now trained
    objective.per_pair(moved, target)
    assert torch.isclose(objective.slack_logit.detach(), logits.mean(), atol=1e-5)


def test_slack_un_degenerates_single_source():
    """One real source softmaxes to 1.0 with zero gradient; the slack row makes it a real two-way decision."""
    logits, target = _pair(1, 4)
    logits = logits.clone().requires_grad_(True)

    plain = SoftmaxFocalBCE.of(logits, target, (SOURCE_AXIS,))
    assert float(plain) == 0.0  # a length-1 source axis constrains nothing — the measured degeneracy

    objective = SlackRow(balanced=False)
    loss = objective.per_pair(logits, target)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0


def test_slack_batched_matches_per_pair():
    """Element `b` of the slack `batched` equals `per_pair` on that pair's unpadded block — the batched contract."""
    logits, target = _pair(4, 5)
    per = SlackRow(balanced=False)
    per.per_pair(logits, target)  # calibrate on the same block
    expected = per.per_pair(logits, target)

    batched_objective = SlackRow(balanced=False)
    masks = (torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool))
    got = batched_objective.batched(logits.unsqueeze(0), target.unsqueeze(0), masks)
    assert torch.isclose(got[0], expected, atol=1e-5)

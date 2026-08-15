"""Unit test for the softmax-focal BCE link loss — and for the axis that decides what it can teach."""

import pytest
import torch

from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, TARGET_AXIS, SoftmaxFocalBCE

_BOTH = (SOURCE_AXIS, TARGET_AXIS)


def _padded(pairs: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad a ragged list of (logits, target) pairs to one (B, maxS, maxT) block with the source/target masks."""
    max_s = max(logits.shape[0] for logits, _ in pairs)
    max_t = max(logits.shape[1] for logits, _ in pairs)
    logits = torch.zeros(len(pairs), max_s, max_t)
    target = torch.zeros(len(pairs), max_s, max_t)
    source_mask = torch.zeros(len(pairs), max_s, dtype=torch.bool)
    target_mask = torch.zeros(len(pairs), max_t, dtype=torch.bool)
    for i, (pair_logits, pair_target) in enumerate(pairs):
        s, t = pair_logits.shape
        logits[i, :s, :t] = pair_logits
        target[i, :s, :t] = pair_target
        source_mask[i, :s] = True
        target_mask[i, :t] = True
    return logits, target, source_mask, target_mask


@pytest.mark.parametrize("axes", [(SOURCE_AXIS,), (TARGET_AXIS,), _BOTH])
@pytest.mark.parametrize("balanced", [False, True])
def test_batched(axes: tuple[int, ...], balanced: bool):
    """The padded batched loss returns each pair's scalar IDENTICAL to `of` on its unpadded matrix.

    Covers the equivalence classes that make the padding non-trivial: a lone source (the source axis constrains
    nothing, and over the source axis alone the whole pair reads zero), a wide contested target, and uneven
    node counts padded to a common block — the case the per-pair loop never had to get right.
    """
    torch.manual_seed(0)
    pairs = [
        (torch.randn(1, 3), torch.tensor([[1.0, 0.0, 0.0]])),  # lone source
        (torch.randn(4, 2), torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])),  # contested
        (torch.randn(2, 3), torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])),  # uneven
    ]
    logits, target, source_mask, target_mask = _padded(pairs)

    batched = SoftmaxFocalBCE.batched(logits, target, (source_mask, target_mask), axes, balanced=balanced)

    for i, (pair_logits, pair_target) in enumerate(pairs):
        expected = SoftmaxFocalBCE.of(pair_logits, pair_target, axes, balanced=balanced)
        assert torch.allclose(batched[i], expected, atol=1e-5), f"pair {i}: {batched[i].item()} != {expected.item()}"


def test_softmax_focal_b_c_e_of():
    """A confident-correct link scores below a confident-wrong one; finite + differentiable; empty target is zero."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # source 0 is the parent of target 0
    right = torch.tensor([[6.0, -6.0], [-6.0, -6.0]], requires_grad=True)
    wrong = torch.tensor([[-6.0, 6.0], [6.0, 6.0]])
    lower = SoftmaxFocalBCE.of(right, target)
    assert torch.isfinite(lower) and lower.item() < SoftmaxFocalBCE.of(wrong, target).item()
    lower.backward()
    assert right.grad is not None
    assert SoftmaxFocalBCE.of(torch.zeros(2, 2), torch.zeros(2, 2)).item() == 0.0


def test_softmax_focal_b_c_e_of_is_blind_to_a_lone_source_over_the_source_axis():
    """The defect the target axis exists to fix: one source makes the source-softmax a constant.

    `softmax(dim=0)` of a single row is identically 1.0 whatever the logits, so no logit can change the term.
    28.8% of our real training pairs carry exactly one source, and each reported a loss of 66.67 while
    learning nothing until that axis was dropped as the non-constraint it is.
    """
    lone = SoftmaxFocalBCE.of(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0]]))
    contested = SoftmaxFocalBCE.of(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    assert lone.item() == 0.0  # no constraint to state, and no constant left behind to swamp the live terms
    assert contested.item() > 0.0  # two candidate parents IS a constraint, so the term returns


def test_softmax_focal_b_c_e_of_ranks_a_lone_source_over_the_target_axis():
    """Over targets that same pair is a real distribution: the truth is rewarded, its rivals are pushed down."""
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])

    SoftmaxFocalBCE.of(logits, target, axes=_BOTH).backward()

    assert logits.grad is not None
    truth, *rivals = logits.grad[0].tolist()
    assert truth < 0  # raise the true successor
    assert all(rival > 0 for rival in rivals)  # and lower every rival IN THE SAME ROW


def test_softmax_focal_b_c_e_of_over_both_axes_prefers_a_mutually_consistent_match():
    """Both axes agree only when the link is the best child of its source AND the best parent of its target."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mutual = torch.tensor([[6.0, -6.0], [-6.0, -6.0]])
    contested = torch.tensor([[6.0, -6.0], [6.0, -6.0]])  # source 1 claims target 0 just as hard

    assert (
        SoftmaxFocalBCE.of(mutual, target, axes=_BOTH).item() < SoftmaxFocalBCE.of(contested, target, axes=_BOTH).item()
    )


def test_softmax_focal_b_c_e_of_defaults_to_the_source_axis_alone():
    """The default is the frontier's form, unchanged — a run that does not ask sees identical numbers."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logits = torch.tensor([[2.0, -1.0], [-3.0, 0.5]])

    assert SoftmaxFocalBCE.of(logits, target).item() == SoftmaxFocalBCE.of(logits, target, axes=(SOURCE_AXIS,)).item()


def test_softmax_focal_b_c_e_of_drops_an_axis_that_constrains_nothing():
    """A lone source contributes no source-axis term at all — not a huge constant one.

    Over that axis the softmax is 1.0 regardless of the logits, so the term is a clamped BCE(1, 0) on every
    rival: a large number that carries no gradient and swamps the terms that do. Dropping it is what makes the
    logged loss readable, and it is why the same pair over BOTH axes reports an ordinary value.
    """
    logits = torch.zeros(1, 3)
    target = torch.tensor([[1.0, 0.0, 0.0]])

    assert SoftmaxFocalBCE.of(logits, target).item() == 0.0  # nothing to say, so it says nothing
    assert SoftmaxFocalBCE.of(logits, target, axes=_BOTH).item() < 1.0  # the target axis alone, at half weight


def test_softmax_focal_b_c_e_of_balanced_is_off_by_default():
    """The balanced reduction is opt-in — a run that does not ask sees the cell mean it always saw."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    logits = torch.tensor([[2.0, -1.0], [-3.0, 0.5]])

    assert SoftmaxFocalBCE.of(logits, target).item() == SoftmaxFocalBCE.of(logits, target, balanced=False).item()


def test_softmax_focal_b_c_e_of_balanced_gives_the_truth_the_weight_of_the_whole_field():
    """One true candidate carries as much as all its rivals together, however many of them there are.

    Under the cell mean a wide decision is mostly "not that one": with 39 rivals the truth holds 1/40 of the
    term, so widening the candidate set quietly dilutes the only cell that says what the answer IS. Balanced,
    the positive mean and the negative mean each count once, so the truth's share is 1/2 at every width — the
    same rule `BalancedBCE` already applies to detection, read per decision.
    """
    target = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])  # one parent among five candidates for one target
    logits = torch.zeros(5, 1, requires_grad=True)

    SoftmaxFocalBCE.of(logits, target, balanced=True).backward()

    assert logits.grad is not None
    truth, *rivals = logits.grad[:, 0].tolist()
    assert truth < 0 and all(rival > 0 for rival in rivals)
    assert abs(truth) == pytest.approx(sum(rivals), rel=1e-5)  # the field pushes back exactly as hard as the truth


def test_softmax_focal_b_c_e_of_balanced_is_the_mean_over_decisions():
    """Hand-computed: each column contributes its own positive mean plus its own rival mean, then averaged.

    At uniform logits every cell's focal BCE is 0.25 * -log(0.5) = 0.1733. The left column holds the truth and
    one rival, so it contributes 0.1733 + 0.1733; the right column holds two rivals and no truth, so it
    contributes 0 + 0.1733. The loss is the mean of the two, 0.2599 — where the cell mean returns 0.1733,
    every cell being identical. The gap IS the balance: the decision that has an answer to teach now carries
    more than the one that only has rivals to push down.
    """
    logits = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    cell = 0.25 * -torch.log(torch.tensor(0.5))

    balanced = SoftmaxFocalBCE.of(logits, target, balanced=True)

    assert balanced.item() == pytest.approx(((cell + cell) + cell).item() / 2, rel=1e-5)
    assert SoftmaxFocalBCE.of(logits, target).item() == pytest.approx(cell.item(), rel=1e-5)


def test_softmax_focal_b_c_e_of_weights_a_pair_by_the_constraints_it_can_state():
    """A lone-source pair pushes HALF as hard as a contested one, because only one of two axes binds.

    The divisor is the axes ASKED FOR, not the ones that happened to bind. A target with one candidate parent
    carries less evidence about association than one with several competing for it, and this is where that
    difference is expressed without inventing a weight. Measured, not assumed: the arm trained under this
    weighting is the best own-weights checkpoint of the campaign, and renormalising onto the surviving axis
    instead failed to beat its own initialisation.
    """
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    SoftmaxFocalBCE.of(logits, target, axes=_BOTH).backward()

    contested = torch.zeros(1, 3, requires_grad=True)
    SoftmaxFocalBCE.of(contested, target, axes=(TARGET_AXIS,)).backward()

    assert logits.grad is not None and contested.grad is not None
    assert torch.allclose(logits.grad, contested.grad / 2)  # one binding axis of two asked for

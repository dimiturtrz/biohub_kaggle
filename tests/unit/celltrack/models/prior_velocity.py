"""Unit tests for the prior-velocity feature — the expected incoming displacement and its warm-start widening.

Both are pure tensor code with a closed form, so every case here is hand-computed: a confident previous gap,
an ambiguous one, a partially-matched one, and a first frame with no previous gap at all. The widening is
checked the only way that matters — the widened layer reproducing the original's output exactly.
"""

import torch
from torch import nn

from celltrack.models.prior_velocity import GapHistory, PriorVelocity


def test_velocity():
    """A carried gap normalises its RAW logits over the SOURCES, then takes the expected incoming displacement.

    The softmax lives inside the history, not at its two call sites, so the trainer and the shipped scorer
    cannot normalise a gap differently: this is the one definition both go through.
    """
    previous = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    coordinates = torch.tensor([[0.0, 0.0, 10.0]])
    logits = torch.tensor([[1.5], [1.5]])  # equal logits == an ambiguous gap, softmax to 0.5 each

    velocity = GapHistory(previous, logits).velocity(coordinates)

    expected = PriorVelocity.expected_incoming(coordinates, previous, torch.softmax(logits, dim=0))
    assert torch.equal(velocity, expected)
    assert torch.equal(velocity, torch.tensor([[0.0, 0.0, 9.0]]))  # 10 - (0.5*0 + 0.5*2)


def test_velocity_without_a_previous_gap():
    """An empty history — a video's FIRST gap, or a training pair with no predecessor — yields exact zeros."""
    coordinates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.equal(GapHistory().velocity(coordinates), torch.zeros(2, 3))
    assert torch.equal(GapHistory(coordinates).velocity(coordinates), torch.zeros(2, 3))  # coords without logits


def test_expected_incoming():
    """A confident previous gap gives the node's displacement from its predecessor; an ambiguous one averages."""
    previous = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    coordinates = torch.tensor([[0.0, 0.0, 10.0]])

    confident = PriorVelocity.expected_incoming(coordinates, previous, torch.tensor([[1.0], [0.0]]))
    assert torch.equal(confident, torch.tensor([[0.0, 0.0, 10.0]]))

    ambiguous = PriorVelocity.expected_incoming(coordinates, previous, torch.tensor([[0.5], [0.5]]))
    assert torch.equal(ambiguous, torch.tensor([[0.0, 0.0, 9.0]]))  # 10 - (0.5*0 + 0.5*2)

    opposed = PriorVelocity.expected_incoming(
        torch.tensor([[0.0, 0.0, 5.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        torch.tensor([[0.5], [0.5]]),
    )
    assert torch.equal(opposed, torch.zeros(1, 3))  # two equally-likely, opposite predecessors cancel


def test_expected_incoming_partial_mass():
    """Incoming mass below one shrinks the velocity toward zero instead of extrapolating a confident guess."""
    velocity = PriorVelocity.expected_incoming(
        torch.tensor([[0.0, 0.0, 10.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.4]]),
    )
    assert torch.equal(velocity, torch.tensor([[0.0, 0.0, 4.0]]))


def test_expected_incoming_first_frame():
    """A node with no previous gap — the first frame — takes the zero vector, the same value an unmatched node takes."""
    coordinates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.equal(PriorVelocity.expected_incoming(coordinates), torch.zeros(2, 3))


def test_expected_incoming_shapes():
    """The result is one 3-vector per current node, in the coordinates' dtype whatever the probabilities' is."""
    coordinates = torch.rand(5, 3)
    velocity = PriorVelocity.expected_incoming(coordinates, torch.rand(4, 3), torch.rand(4, 5, dtype=torch.float64))
    assert velocity.shape == (5, PriorVelocity.DIM)
    assert velocity.dtype == coordinates.dtype


def test_widen_linear_inputs():
    """The widened layer reproduces the original exactly on the original inputs, whatever the appended ones are."""
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)
    widened = PriorVelocity.widen_linear_inputs(layer, PriorVelocity.DIM)
    inputs = torch.randn(7, 4)
    appended = torch.randn(7, PriorVelocity.DIM) * 1e3

    assert widened.in_features == 4 + PriorVelocity.DIM
    assert torch.equal(widened(torch.cat([inputs, appended], dim=-1)), layer(inputs))
    assert torch.equal(widened.weight[:, 4:], torch.zeros(3, PriorVelocity.DIM))


def test_widen_linear_inputs_without_bias():
    """A bias-free layer widens to a bias-free layer — the copy carries the original's structure, not a default."""
    layer = nn.Linear(2, 2, bias=False)
    widened = PriorVelocity.widen_linear_inputs(layer, 1)
    assert widened.bias is None
    assert torch.equal(widened(torch.cat([torch.eye(2), torch.ones(2, 1)], dim=-1)), layer(torch.eye(2)))

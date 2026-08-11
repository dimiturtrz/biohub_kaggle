"""Unit tests for the contrastive projection head — its width, its slice, and its two gradient regimes.

The head exists to keep a contrastive gradient off the shared trunk, so the tests that matter are the two
`.grad` cases: detached, a parameter that produced the features must stay untouched; attached, it must not.
"""

import torch
from torch import nn

from celltrack.models.contrastive_projection import ContrastiveProjection, ContrastiveProjectionConfig

_APPEARANCE_DIM = 6
_TRAILING_DIM = 4  # stands in for the sinusoidal position embed (and the prior-velocity block behind it)


def _trunk() -> nn.Linear:
    """A stand-in backbone: the one parameter every node feature in these tests was produced by."""
    torch.manual_seed(0)
    return nn.Linear(3, _APPEARANCE_DIM + _TRAILING_DIM)


def _head(*, stop_gradient: bool) -> ContrastiveProjection:
    """The head under test at the fixture's appearance width, in one of its two gradient regimes."""
    torch.manual_seed(0)
    return ContrastiveProjection(ContrastiveProjectionConfig(_APPEARANCE_DIM, stop_gradient=stop_gradient))


def test_contrastive_projection_config_embedding_dim():
    """The contrasted width is DERIVED from the appearance block it re-bases, not chosen — they are equal."""
    assert ContrastiveProjectionConfig(32).embedding_dim == 32


def test_contrastive_projection_embedding_dim():
    """The head reports its config's width, so what InfoNCE is told comes from the head that produced the vector."""
    assert _head(stop_gradient=False).embedding_dim == _APPEARANCE_DIM


def test_forward():
    """The output is one embedding per node, exactly as wide as the head reports to InfoNCE."""
    embedding = _head(stop_gradient=False).forward(torch.randn(5, _APPEARANCE_DIM + _TRAILING_DIM))
    assert embedding.shape == (5, _head(stop_gradient=False).embedding_dim)


def test_forward_reads_only_the_leading_appearance_block():
    """Trailing channels — position, then velocity — do not reach the head, whatever their values are."""
    head = _head(stop_gradient=False)
    appearance = torch.randn(3, _APPEARANCE_DIM)
    near = torch.cat([appearance, torch.full((3, _TRAILING_DIM), 5.0)], dim=1)
    far = torch.cat([appearance, torch.full((3, _TRAILING_DIM), -70.0)], dim=1)
    assert torch.equal(head.forward(near), head.forward(far))


def test_forward_detached_leaves_the_trunk_untouched():
    """The strict fix: the contrastive gradient reaches the head's own weights and no backbone weight at all."""
    trunk, head = _trunk(), _head(stop_gradient=True)

    head.forward(trunk(torch.randn(4, 3))).square().sum().backward()

    assert trunk.weight.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters())


def test_forward_attached_reaches_the_trunk():
    """The other regime is one field away: attached, the same backward does move the parameter behind the features."""
    trunk, head = _trunk(), _head(stop_gradient=False)

    head.forward(trunk(torch.randn(4, 3))).square().sum().backward()

    assert trunk.weight.grad is not None
    assert trunk.weight.grad.abs().sum() > 0

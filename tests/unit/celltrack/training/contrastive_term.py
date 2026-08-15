"""Unit tests for the contrastive term's PLACEMENT — which vectors it contrasts, and how wide it says they are.

The width is the part that has already broken once: `InfoNCE.of` takes the LEADING appearance width, and an
earlier version derived it by subtracting a trailing block, which mis-sliced the moment a third block was
appended. So the contract is asserted directly here — whatever the term hands the loss, the width it names
is the true width of that block.
"""

import math

import pytest
import torch
from torch import Tensor, nn

from celltrack.losses.info_nce import InfoNCE
from celltrack.training.contrastive_term import ContrastiveTerm, ContrastiveTermConfig
from celltrack.training.joint_config import ContrastiveSite, LossCfg

_APPEARANCE_DIM = 2
_TRAILING_DIM = 3  # the position embed the term must never contrast


def _term(site: ContrastiveSite) -> ContrastiveTerm:
    """The term at one site, at temperature 1 so every expected value below is a closed form."""
    torch.manual_seed(0)
    loss = LossCfg(contrastive_site=site, temperature=1.0)
    return ContrastiveTerm(ContrastiveTermConfig.from_loss(loss, _APPEARANCE_DIM))


def _features(appearance: list[list[float]]) -> Tensor:
    """Appearance rows with the constant trailing block real node features carry appended."""
    rows = torch.tensor(appearance)
    return torch.cat([rows, torch.full((rows.shape[0], _TRAILING_DIM), 7.0)], dim=1)


def _identity(term: ContrastiveTerm) -> ContrastiveTerm:
    """The same term with its head set to the identity, so the projection is exactly one GELU — hand-computable."""
    assert term.projection is not None
    for layer in (module for module in term.projection.mlp if isinstance(module, nn.Linear)):
        with torch.no_grad():
            layer.weight.copy_(torch.eye(_APPEARANCE_DIM))
            layer.bias.zero_()
    return term


def test_from_loss():
    """The term's config reads exactly three values off the whole objective — temperature, site, appearance width."""
    config = ContrastiveTermConfig.from_loss(LossCfg(contrastive_site=ContrastiveSite.PROJECTION, temperature=0.7), 5)
    assert config == ContrastiveTermConfig(temperature=0.7, site=ContrastiveSite.PROJECTION, appearance_dim=5)


def test_forward():
    """At the FEATURES site nothing is interposed: the term IS InfoNCE over the leading appearance block."""
    source, target = _features([[1.0, 0.0]]), _features([[1.0, 0.0], [0.0, 1.0]])
    edge = torch.tensor([[1.0, 0.0]])

    value = _term(ContrastiveSite.FEATURES).forward(source, target, edge)

    assert torch.allclose(value, InfoNCE.of(source, target, edge, 1.0, _APPEARANCE_DIM))
    assert torch.allclose(value, torch.tensor(math.log(1 + math.exp(-1.0))))  # cosines (1, 0) at temperature 1


def test_forward_through_an_identity_projection():
    """Hand-computed through the head: GELU keeps (1, 0) and (0, 1) orthogonal, so the loss is log(1 + e^-1).

    GELU(1) scales the vector and GELU(0) is exactly zero, and the loss normalises, so the projected cosines
    are the unprojected ones — which makes the value a constant rather than a re-run of the implementation.
    """
    source, target = _features([[1.0, 0.0]]), _features([[1.0, 0.0], [0.0, 1.0]])

    value = _identity(_term(ContrastiveSite.PROJECTION)).forward(source, target, torch.tensor([[1.0, 0.0]]))

    assert torch.allclose(value, torch.tensor(math.log(1 + math.exp(-1.0))))


def test_forward_tells_the_loss_the_true_width_of_what_it_contrasts(monkeypatch: pytest.MonkeyPatch):
    """The width named is the width handed over — the projected embedding's own, not the backbone's."""
    seen: list[tuple[int, int]] = []
    contrast = InfoNCE.of

    def _recording(source: Tensor, target: Tensor, matrix: Tensor, temperature: float, appearance_dim: int) -> Tensor:
        seen.append((source.shape[1], appearance_dim))
        return contrast(source, target, matrix, temperature, appearance_dim)

    monkeypatch.setattr(InfoNCE, "of", _recording)
    term = _term(ContrastiveSite.PROJECTION)
    assert term.projection is not None

    term.forward(_features([[1.0, 0.0]]), _features([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))

    assert seen == [(term.projection.embedding_dim, term.projection.embedding_dim)]


def _backward_through(site: ContrastiveSite) -> nn.Linear:
    """A stand-in backbone whose output the term contrasts, returned after one backward through the term."""
    torch.manual_seed(0)
    trunk = nn.Linear(_APPEARANCE_DIM + _TRAILING_DIM, _APPEARANCE_DIM + _TRAILING_DIM)
    source = trunk(_features([[1.0, 0.0]]))
    target = trunk(_features([[1.0, 0.0], [0.0, 1.0]]))
    _term(site).forward(source, target, torch.tensor([[1.0, 0.0]])).backward()
    return trunk


def test_forward_detached_reaches_no_backbone_parameter():
    """DETACHED_PROJECTION is the strict fix: the term shapes the embedding and leaves the trunk alone."""
    assert _backward_through(ContrastiveSite.DETACHED_PROJECTION).weight.grad is None


def test_forward_attached_reaches_the_backbone():
    """The other two sites are one field away, and both do move the parameter behind the features."""
    for site in (ContrastiveSite.PROJECTION, ContrastiveSite.FEATURES):
        gradient = _backward_through(site).weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


def test_forward_holds_parameters_only_where_it_projects():
    """An unasked run builds no head at all — nothing to optimise, and nothing drawn from the RNG."""
    assert list(_term(ContrastiveSite.FEATURES).parameters()) == []
    assert list(_term(ContrastiveSite.PROJECTION).parameters()) != []

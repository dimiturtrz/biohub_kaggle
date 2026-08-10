"""Unit test for the InfoNCE contrastive term over node appearance features."""

import math

import torch

from celltrack.losses.info_nce import InfoNCE

_POSITION_DIM = 2  # the trailing block these fixtures append to stand in for the sinusoidal position embed


def _features(appearance: list[list[float]], position: float) -> torch.Tensor:
    """Appearance rows with a constant `_POSITION_DIM`-wide position block appended, as node features carry."""
    rows = torch.tensor(appearance)
    return torch.cat([rows, torch.full((rows.shape[0], _POSITION_DIM), position)], dim=1)


def test_info_n_c_e_of():
    """Hand-computed 2x2; perfect ranking → ~0; unannotated rows ignored; divisions averaged; no row → 0 not NaN."""
    source = _features([[1.0, 0.0]], 0.0)
    target = _features([[1.0, 0.0], [0.0, 1.0]], 0.0)
    edge = torch.tensor([[1.0, 0.0]])
    # cosine sims are (1, 0) at temperature 1, so the loss is -log(e / (e + 1)) = log(1 + e^-1).
    assert torch.allclose(
        InfoNCE.of(source, target, edge, 1.0, _POSITION_DIM), torch.tensor(math.log(1 + math.exp(-1.0)))
    )

    sharp = InfoNCE.of(source, target, edge, 0.01, _POSITION_DIM)  # a low temperature on a perfect ranking
    assert 0.0 <= sharp.item() < 1e-6

    # A second source whose row carries no annotated successor must not change the value at all.
    padded_source = _features([[1.0, 0.0], [0.0, 1.0]], 0.0)
    padded_edge = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(
        InfoNCE.of(padded_source, target, padded_edge, 1.0, _POSITION_DIM),
        InfoNCE.of(source, target, edge, 1.0, _POSITION_DIM),
    )

    # A division (one source, two annotated daughters) is the mean of -log p over both positives.
    division = InfoNCE.of(source, target, torch.tensor([[1.0, 1.0]]), 1.0, _POSITION_DIM)
    log_probability = torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)
    assert torch.allclose(division, -log_probability.mean())

    empty = InfoNCE.of(source, target, torch.zeros(1, 2), 1.0, _POSITION_DIM)
    assert empty.item() == 0.0 and not torch.isnan(empty)


def test_info_n_c_e_of_ignores_the_position_block():
    """Cells at different POSITIONS with identical appearance are not separable — the position channels are dropped."""
    source = _features([[1.0, 0.0]], 5.0)
    near = _features([[0.0, 1.0], [1.0, 0.0]], 5.0)  # the wrong target shares the source's position block
    far = _features([[0.0, 1.0], [1.0, 0.0]], -3.0)
    edge = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(
        InfoNCE.of(source, near, edge, 1.0, _POSITION_DIM), InfoNCE.of(source, far, edge, 1.0, _POSITION_DIM)
    )


def test_info_n_c_e_of_is_differentiable():
    """The term reaches the features it contrasts — the point of the loss is a gradient into the backbone."""
    source = _features([[1.0, 0.0]], 0.0).requires_grad_(True)
    target = _features([[1.0, 0.0], [0.0, 1.0]], 0.0)
    InfoNCE.of(source, target, torch.tensor([[1.0, 0.0]]), 0.07, _POSITION_DIM).backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()

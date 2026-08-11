"""Unit tests for the JS-reliability logarithmic opinion pool — reliability, the soft AND, and the re-centring."""

import torch

from celltrack.edges.js_log_pool import JensenShannonLogPool


def _logits(*columns: list[float]) -> torch.Tensor:
    """View logits whose softmax-over-sources is exactly the given per-view source distributions."""
    return torch.log(torch.tensor(columns).unsqueeze(-1))  # (v, s, 1) — one target, several sources


def test_reliability():
    """Every view at the same distribution diverges by zero, so nothing is down-weighted — a plain geometric mean."""
    agreed = torch.tensor([[0.7], [0.3]]).expand(4, 2, 1)
    weights = JensenShannonLogPool.reliability(agreed)
    assert torch.allclose(weights, torch.full((4, 1), 0.25))


def test_reliability_sums_to_one_over_views():
    """The weights are convex — that is what keeps the pooled distribution at one view's temperature."""
    weights = JensenShannonLogPool.reliability(torch.tensor([[0.9, 0.2], [0.1, 0.8]]).unsqueeze(0).repeat(3, 1, 1))
    assert torch.allclose(weights.sum(dim=0), torch.ones(2))


def test_reliability_down_weights_the_dissenting_view():
    """The view furthest from the consensus gets the smallest weight, the agreeing ones share the rest equally."""
    distributions = torch.tensor([[[0.1], [0.9]], [[0.1], [0.9]], [[0.1], [0.9]], [[0.9], [0.1]]])
    weights = JensenShannonLogPool.reliability(distributions)
    assert weights[3, 0] < weights[0, 0]
    assert torch.allclose(weights[:3, 0], weights[0, 0].expand(3))


def test_pool():
    """Re-centring restores the identity view's scale exactly, so the pool is a no-op on unanimous views."""
    identity = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    pooled = JensenShannonLogPool.pool(identity.expand(4, 2, 2))
    assert torch.allclose(pooled, identity, atol=1e-5)


def test_pool_recentres_onto_the_identity_views_mean():
    """Whatever the views disagree about, each target's pooled logits sit on the identity view's offset."""
    views = torch.tensor([[[2.0, -1.0], [0.5, 3.0]], [[0.0, 4.0], [1.0, -2.0]], [[1.0, 1.0], [2.0, 0.0]]])
    pooled = JensenShannonLogPool.pool(views)
    assert torch.allclose(pooled.mean(dim=0), views[0].mean(dim=0), atol=1e-5)


_ADMISSION_THRESHOLD = 0.28


def test_pool_refuses_an_edge_the_arithmetic_mean_admits():
    """The whole point: one optimistic view carries an arithmetic mean over the bar and gets pulled under it here.

    Three views give the candidate source 0.1 of the target's mass; a single dissenting view gives it 0.9. The
    arithmetic mean — a soft OR — lands at 0.30 and admits the edge. The log pool is a soft AND, and the JS
    reliability additionally discounts the dissenter, so the same candidate comes out below the same bar.
    """
    views = _logits([0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1])
    arithmetic = torch.softmax(views, dim=1).mean(dim=0)
    pooled = torch.softmax(JensenShannonLogPool.pool(views), dim=0)
    assert float(arithmetic[0, 0]) > _ADMISSION_THRESHOLD
    assert float(pooled[0, 0]) < _ADMISSION_THRESHOLD


def test_pool_output_is_a_valid_distribution_after_softmax():
    """The pooled logits still normalise over the sources of each target — the head's own normalisation."""
    views = torch.randn(4, 5, 3)
    pooled = torch.softmax(JensenShannonLogPool.pool(views), dim=0)
    assert torch.allclose(pooled.sum(dim=0), torch.ones(3), atol=1e-5)

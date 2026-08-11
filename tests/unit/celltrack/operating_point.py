import pytest

from celltrack.operating_point import TrackerConfig


def test_post_init():
    """An operating point that cannot mean what it says is refused at construction, not obeyed quietly."""
    with pytest.raises(ValueError, match="edge_blend must sum to 1"):
        TrackerConfig(edge_blend=(0.8, 0.8))  # rescales the logits -> shifts the softmax temperature
    with pytest.raises(ValueError, match="threshold must be a probability"):
        TrackerConfig(threshold=5.0)
    with pytest.raises(ValueError, match="min_track_length"):
        TrackerConfig(min_track_length=0)

    assert TrackerConfig().threshold == 0.97  # the LB-measured best, not the round number


def test_shipped():
    """The submitted 0.895 recipe, pinned field by field — this is the contract four modules now mount.

    Pinned rather than smoke-tested because the whole point of the method is that the selector, the kernel and
    the diagnostics agree on ONE operating point. If a field here drifts, the thing that breaks is not this
    test but the comparability of every checkpoint we select and every number we quote, silently.
    """
    shipped = TrackerConfig.shipped()

    assert shipped.threshold == 0.97  # LB ladder 0.99/0.98/0.97 = 0.887/0.891/0.892
    assert (shipped.linker.name, shipped.linker.disappearance_cost) == ("flow", 3.0)  # global solver, its peak
    assert shipped.bidirectional_edges is True  # +0.004 on the LB under a global consumer, -0.0027 under a local one
    assert (shipped.linker.gate_um, shipped.linker.affinity_bonus) == (10.0, 20.0)
    assert (shipped.min_track_length, shipped.smooth_strength) == (6, 0.8)  # both LB-arbitrated, not proxy
    assert shipped.division is None  # divisions stay opt-in: the proxy cannot arbitrate that axis

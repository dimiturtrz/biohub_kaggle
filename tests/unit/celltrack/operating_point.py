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

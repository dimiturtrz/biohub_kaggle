import pytest

from celltrack.training.early_stop import EarlyStop


def test_patience_below_one_is_rejected():
    with pytest.raises(ValueError, match="patience"):
        EarlyStop(patience=0)


def test_update():
    stop = EarlyStop(patience=3)
    assert stop.update(0.5) is True  # first score is a best
    assert stop.update(0.6) is True  # strictly better
    assert stop.update(0.6) is False  # no gain
    assert stop.best == pytest.approx(0.6)


def test_min_delta_gates_tiny_gains():
    stop = EarlyStop(patience=2, min_delta=0.01)
    assert stop.update(0.50) is True
    assert stop.update(0.505) is False  # +0.005 < min_delta -> not an improvement
    assert stop.best == pytest.approx(0.50)


def test_should_stop():
    stop = EarlyStop(patience=2)
    stop.update(0.9)  # best
    assert stop.should_stop is False
    stop.update(0.8)  # miss 1
    assert stop.should_stop is False
    stop.update(0.8)  # miss 2 -> patience reached
    assert stop.should_stop is True


def test_recovering_dip_resets_the_counter():
    stop = EarlyStop(patience=3)
    stop.update(0.90)  # best
    stop.update(0.85)  # dip 1
    stop.update(0.86)  # dip 2
    assert stop.should_stop is False
    assert stop.update(0.95) is True  # recovers to a new best
    assert stop.waited == 0
    assert stop.should_stop is False

import types
from typing import cast

from celltrack.training.tracked_run import TrackedRun
from core.hparams import RunConfig
from core.obs import Obs
from core.tracking import Tracker


class _RecordingBackend:
    """A duck-typed mlflow stand-in that records the metric calls the tracker forwards."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, value, step))


def _run(config: RunConfig, backend: _RecordingBackend | None = None) -> TrackedRun:
    tracker = Tracker._Live(cast(types.ModuleType, backend)) if backend else Tracker._Noop()
    return TrackedRun(config, tracker, Obs.setup())


def _scripted(scores: list[float]):
    """A stateful evaluate() returning the next scripted score each call."""
    it = iter(scores)
    return lambda: {"score": next(it)}


def test_fit():
    config = RunConfig().apply_overrides(["optim.epochs=10", "optim.patience=2", "eval.every=1"])
    saves = []
    best = _run(config).fit(
        train=lambda: {"edge": 0.01},
        evaluate=_scripted([0.5, 0.6, 0.55, 0.54]),  # best at epoch 1, then 2 non-improving -> stop @3
        save=lambda: saves.append(1),
    )
    assert best == 0.6
    assert len(saves) == 2  # saved only on the two improvements


def test_fit_stops_before_the_epoch_ceiling():
    config = RunConfig().apply_overrides(["optim.epochs=50", "optim.patience=1", "eval.every=1"])
    trains = []
    _run(config).fit(
        train=lambda: (trains.append(1), {"edge": 0.0})[1],
        evaluate=_scripted([0.9, 0.8]),  # best @0, miss @1 -> patience 1 exhausted -> stop
        save=lambda: None,
    )
    assert len(trains) == 2  # stopped at epoch 1, far short of 50


def test_fit_evaluates_only_on_scheduled_and_final_epochs():
    config = RunConfig().apply_overrides(["optim.epochs=4", "optim.patience=10", "eval.every=2"])
    evals = []

    def evaluate():
        evals.append(1)
        return {"score": 0.5 + 0.01 * len(evals)}  # strictly improving -> never early-stops

    trains = []
    _run(config).fit(train=lambda: (trains.append(1), {"edge": 0.0})[1], evaluate=evaluate, save=lambda: None)
    assert len(trains) == 4  # every epoch trains
    assert len(evals) == 3  # epochs 0, 2, and the last (3)


def test_fit_logs_metrics_to_the_tracker_per_epoch():
    config = RunConfig().apply_overrides(["optim.epochs=1", "optim.patience=5", "eval.every=1"])
    backend = _RecordingBackend()
    _run(config, backend).fit(
        train=lambda: {"edge": 0.02},
        evaluate=lambda: {"score": 0.7},
        save=lambda: None,
    )
    logged = {(key, step) for key, _, step in backend.metrics}
    assert ("edge", 0) in logged
    assert ("score", 0) in logged
    assert ("best", 0) in logged

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from celltrack.eval import proxy_eval
from celltrack.eval.proxy import CV_MOVIES
from celltrack.eval.proxy_eval import TrackerProxyEval, _Args, _ConfigOverride, _SweepTracking
from celltrack.tracker import CellTracker, TrackerConfig
from core.metrics.score import VideoMetrics
from core.paths import DataRoot
from tests.unit.celltrack.conftest import RecordingMlflow


class _Edges:
    """The edge counts of one movie, exposing only the Jaccard the sweep records."""

    def jaccard(self) -> float:
        return 0.5


class _Metrics:
    """One movie's metrics, exposing only what a tracked sweep cell reads off them."""

    edges = _Edges()

    def total_node_ratio(self) -> float:
        return 0.25


def test_apply():
    """`--set` overrides keep each field's type, and a dotted key reaches the nested (pydantic) linker config."""
    base = TrackerConfig()
    assert _ConfigOverride.apply(base, "smooth_strength=0.5").smooth_strength == 0.5
    assert _ConfigOverride.apply(base, "min_track_length=5").min_track_length == 5  # cast to int, not float
    assert _ConfigOverride.apply(base, "linker.name=flow").linker.name == "flow"  # dotted key updates nested config
    # A dotted key reaches an off-by-default nested config, instantiating it from its declared type first.
    assert _ConfigOverride.apply(base, "reuse.gate_um=7.5").reuse.gate_um == 7.5
    assert _ConfigOverride.apply(base, "detector_blend=0.6").detector_blend == 0.6  # None-default cast via annotation


def test_cast():
    """`cast` parses a raw string into the target type, treating bool as a flag rather than an int subclass."""
    assert _ConfigOverride.cast(int, "5") == 5
    assert _ConfigOverride.cast(float, "0.5") == 0.5
    assert _ConfigOverride.cast(bool, "true") is True
    assert _ConfigOverride.cast(bool, "false") is False  # not int("false"): bool is parsed as a flag


def test_concrete_type():
    """`concrete_type` reads a set field's type from its value, and an unset optional's from its annotation."""
    assert _ConfigOverride.concrete_type(TrackerConfig, "threshold", 0.99) is float
    # detector_blend defaults to None, so its real type lives only in the `float | None` annotation.
    assert _ConfigOverride.concrete_type(TrackerConfig, "detector_blend", None) is float


def test_nested():
    """`nested` returns the live sub-config, instantiating an off-by-default one from its declared type."""
    base = TrackerConfig()
    assert _ConfigOverride.nested(base, "linker") is base.linker  # the already-set linker config, unchanged
    assert isinstance(_ConfigOverride.nested(base, "reuse"), BaseModel)  # reuse is None by default -> instantiated


def test_override_rejects_a_bare_key():
    """An override without `=` is a usage error, caught before it silently no-ops."""
    with pytest.raises(ValueError, match="key=value"):
        _ConfigOverride.apply(TrackerConfig(), "smooth_strength")


@dataclass
class _Score:
    """A stand-in for SplitScore exposing only the `.score` the CLI reads."""

    score: float


class _FakePipeline:
    """A mounted tracker whose score depends on the config, so the grid produces distinct values."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    def with_config(self, config: TrackerConfig) -> "_FakePipeline":
        return _FakePipeline(config)


class _FakeProxy:
    """A loaded proxy scoring a pipeline by its threshold and disappearance, so the wiring runs without data."""

    def score(self, pipeline: _FakePipeline) -> _Score:
        return _Score(pipeline.config.threshold + pipeline.config.linker.disappearance_cost)

    def metrics(self, pipeline: _FakePipeline) -> dict[str, float]:
        return {"movie": pipeline.config.threshold}


class _Root:
    """A data root exposing only `.processed`, which the mocked pipeline mount ignores."""

    def processed(self, key: str) -> Path:
        return Path(key)


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_eval.TestMovieProxy, "load", classmethod(lambda cls, root, stems=(): _FakeProxy()))
    monkeypatch.setattr(
        CellTracker,
        "from_packs",
        classmethod(lambda cls, p1, p2, responses, device, config=None: _FakePipeline(TrackerConfig())),
    )


def test_scores(monkeypatch: pytest.MonkeyPatch):
    """`scores` mounts once and returns the proxy score for every `(threshold, disappearance)` in the grid."""
    _patch(monkeypatch)
    result = TrackerProxyEval("cpu", (0.98, 0.99), (0.0,)).scores(cast(DataRoot, _Root()))
    assert result == {(0.98, 0.0): 0.98, (0.99, 0.0): 0.99}


def test_breakdown(monkeypatch: pytest.MonkeyPatch):
    """`breakdown` mounts the tracker at its first swept threshold and returns each movie's metrics."""
    _patch(monkeypatch)
    result = TrackerProxyEval("cpu", (0.98, 0.99)).breakdown(cast(DataRoot, _Root()))
    assert result == {"movie": 0.98}


def test_by_acquisition(monkeypatch: pytest.MonkeyPatch):
    """`by_acquisition` groups the per-movie metrics by their filename prefix into one split score each."""
    monkeypatch.setattr(proxy_eval.SplitScore, "of", staticmethod(lambda metrics: metrics))
    breakdown = {"44b6_a": "m1", "44b6_b": "m2", "6bba_c": "m3"}
    result = TrackerProxyEval.by_acquisition(cast(dict[str, VideoMetrics], breakdown))
    assert result == {"44b6": ["m1", "m2"], "6bba": ["m3"]}


def test_config_at():
    """`config_at` resolves one grid cell — the swept point with every `--set` override folded on top."""
    config = TrackerConfig()
    resolved = TrackerProxyEval("cpu", overrides=("smooth_strength=0.3",)).config_at(0.97, 2.0)
    assert (resolved.threshold, resolved.linker.disappearance_cost) == (0.97, 2.0)
    assert resolved.smooth_strength == 0.3 != config.smooth_strength


def test_cell(mlflow_backend: RecordingMlflow):
    """A swept cell becomes one run: the resolved config as params, the score and per-movie numbers as metrics."""
    config = TrackerProxyEval("cpu").config_at(0.97, 2.0)
    _SweepTracking("test-4").cell(config, 0.81, cast(dict[str, VideoMetrics], {"44b6_x": _Metrics()}))
    params = mlflow_backend.logged_params()
    assert params["threshold"] == 0.97
    assert params["linker.disappearance_cost"] == 2.0  # the pydantic sub-config is dumped, then flattened
    assert ("metric", "proxy_score", 0.81, None) in mlflow_backend.calls
    assert ("metric", "44b6_x.raw_jaccard", 0.5, None) in mlflow_backend.calls
    assert ("metric", "44b6_x.node_ratio", 0.25, None) in mlflow_backend.calls


def test_from_argv(monkeypatch: pytest.MonkeyPatch):
    """`from_argv` parses the threshold and disappearance ranges and selects the CV proxy under `--cv`."""
    monkeypatch.setattr(sys, "argv", ["proxy_eval", "--threshold", "0.97,0.99", "--disappearance", "0.0,2.0", "--cv"])
    args = _Args.from_argv()
    assert args.thresholds == (0.97, 0.99)
    assert args.disappearance_costs == (0.0, 2.0)
    assert args.stems == CV_MOVIES


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The CLI mounts, sweeps the grid, and logs a score for each cell without raising."""
    _patch(monkeypatch)
    monkeypatch.setattr(proxy_eval.DataRoot, "from_config", staticmethod(lambda config: cast(DataRoot, _Root())))
    monkeypatch.setattr(sys, "argv", ["proxy_eval", "--device", "cpu", "--threshold", "0.98,0.99"])
    proxy_eval.main()

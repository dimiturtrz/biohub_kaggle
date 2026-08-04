import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from celltrack import proxy_eval
from celltrack.champion import ChampionConfig, ChampionPipeline
from celltrack.proxy import CV_MOVIES
from celltrack.proxy_eval import ChampionProxyEval, _Args
from core.paths import DataRoot


@dataclass
class _Score:
    """A stand-in for SplitScore exposing only the `.score` the CLI reads."""

    score: float


class _FakePipeline:
    """A mounted champion whose score depends on the config, so the grid produces distinct values."""

    def __init__(self, config: ChampionConfig) -> None:
        self.config = config

    def with_config(self, config: ChampionConfig) -> "_FakePipeline":
        return _FakePipeline(config)


class _FakeProxy:
    """A loaded proxy scoring a pipeline by its threshold and disappearance, so the wiring runs without data."""

    def score(self, pipeline: _FakePipeline) -> _Score:
        return _Score(pipeline.config.threshold + pipeline.config.disappearance_cost)


class _Root:
    """A data root exposing only `.processed`, which the mocked pipeline mount ignores."""

    def processed(self, key: str) -> Path:
        return Path(key)


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_eval.TestMovieProxy, "load", classmethod(lambda cls, root, stems=(): _FakeProxy()))
    monkeypatch.setattr(
        ChampionPipeline,
        "from_packs",
        classmethod(lambda cls, p1, p2, responses, device, config=None: _FakePipeline(ChampionConfig())),
    )


def test_scores(monkeypatch: pytest.MonkeyPatch):
    """`scores` mounts once and returns the proxy score for every `(threshold, disappearance)` in the grid."""
    _patch(monkeypatch)
    result = ChampionProxyEval("cpu", (0.98, 0.99), (0.0,)).scores(cast(DataRoot, _Root()))
    assert result == {(0.98, 0.0): 0.98, (0.99, 0.0): 0.99}


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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from celltrack import proxy_eval
from celltrack.champion import ChampionConfig, ChampionPipeline
from celltrack.proxy_eval import ChampionProxyEval
from core.paths import DataRoot


@dataclass
class _Score:
    """A stand-in for SplitScore exposing only the `.score` the CLI reads."""

    score: float


class _FakePipeline:
    """A mounted champion whose score depends on the config, so the sweep produces distinct values."""

    def __init__(self, config: ChampionConfig) -> None:
        self.config = config

    def with_config(self, config: ChampionConfig) -> "_FakePipeline":
        return _FakePipeline(config)


class _FakeProxy:
    """A loaded proxy that scores a pipeline by its disappearance cost, so the wiring runs without data."""

    def score(self, pipeline: _FakePipeline) -> _Score:
        return _Score(0.9 + pipeline.config.disappearance_cost)


class _Root:
    """A data root exposing only `.processed`, which the mocked pipeline mount ignores."""

    def processed(self, key: str) -> Path:
        return Path(key)


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_eval.TestMovieProxy, "load", classmethod(lambda cls, root: _FakeProxy()))
    monkeypatch.setattr(
        ChampionPipeline,
        "from_packs",
        classmethod(lambda cls, p1, p2, responses, device, config=None: _FakePipeline(ChampionConfig())),
    )


def test_scores(monkeypatch: pytest.MonkeyPatch):
    """`scores` mounts once and returns the proxy score per swept disappearance cost."""
    _patch(monkeypatch)
    result = ChampionProxyEval("cpu", (0.0, 1.0)).scores(cast(DataRoot, _Root()))
    assert result == {0.0: 0.9, 1.0: 1.9}


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The CLI mounts, sweeps the disappearance costs, and logs a score for each without raising."""
    _patch(monkeypatch)
    monkeypatch.setattr(proxy_eval.DataRoot, "from_config", staticmethod(lambda config: cast(DataRoot, _Root())))
    monkeypatch.setattr(sys, "argv", ["proxy_eval", "--device", "cpu", "--disappearance", "0.0,2.0"])
    proxy_eval.main()

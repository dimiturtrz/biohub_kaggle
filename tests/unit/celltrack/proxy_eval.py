import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from celltrack import proxy_eval
from core.geometry import Spacing
from core.paths import DataRoot


@dataclass
class _Score:
    """A stand-in for SplitScore exposing only the `.score` the CLI reads."""

    score: float


class _FakeProxy:
    """A loaded proxy whose score is fixed, so the CLI wiring can be exercised without data or GPU."""

    spacing = Spacing(z=1.0, y=1.0, x=1.0)

    def score(self, pipeline: object) -> _Score:
        return _Score(0.9)


class _Root:
    """A data root exposing only `.processed`, which the mocked scorer loader ignores."""

    def processed(self, key: str) -> Path:
        return Path(key)


def _patch_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_eval.TestMovieProxy, "load", classmethod(lambda cls, root: _FakeProxy()))
    monkeypatch.setattr(proxy_eval, "_scorers", lambda proc, device: {"single-seed": object(), "dual-seed": object()})


def test_evaluate(monkeypatch: pytest.MonkeyPatch):
    """`evaluate` builds the champion pipeline per scorer and returns each one's proxy score."""
    _patch_loads(monkeypatch)
    result = proxy_eval.evaluate(cast(DataRoot, _Root()), "cpu")
    assert result == {"single-seed": 0.9, "dual-seed": 0.9}


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The CLI mounts, evaluates, and logs a score for each scorer without raising."""
    _patch_loads(monkeypatch)
    monkeypatch.setattr(proxy_eval.DataRoot, "from_config", staticmethod(lambda config: cast(DataRoot, _Root())))
    monkeypatch.setattr(sys, "argv", ["proxy_eval", "--device", "cpu"])
    proxy_eval.main()

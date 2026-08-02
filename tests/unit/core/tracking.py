import types
from pathlib import Path
from typing import cast

import pytest

from core.tracking import Tracker


class _StubBackend:
    """Records the mlflow calls a `Tracker._Live` handle forwards, so each forwarder is unit-testable
    without a real mlflow backend. `_Live` types its sink as the mlflow module (that's what it gets in
    production); the stub is duck-compatible, so tests cast it to satisfy the annotation."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.calls.append(("metric", key, value, step))

    def set_tag(self, key: str, value: str) -> None:
        self.calls.append(("tag", key, value))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("artifact", path))

    def end_run(self) -> None:
        self.calls.append(("end",))

    def live(self) -> Tracker._Live:
        return Tracker._Live(cast(types.ModuleType, self))


def test_metric():
    stub = _StubBackend()
    stub.live().metric("val_score", 0.5, step=3)
    assert ("metric", "val_score", 0.5, 3) in stub.calls


def test_tag():
    stub = _StubBackend()
    stub.live().tag("phase", "test")
    assert ("tag", "phase", "test") in stub.calls


def test_artifact(tmp_path: Path):
    stub = _StubBackend()
    present = tmp_path / "config.json"
    present.write_text("{}", encoding="utf-8")
    stub.live().artifact(present)
    stub.live().artifact(tmp_path / "missing.json")  # non-existent path is not logged
    assert stub.calls == [("artifact", str(present))]


def test_end():
    stub = _StubBackend()
    stub.live().end()
    assert ("end",) in stub.calls


def test_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELLTRACK_NO_MLFLOW", "1")
    handle = Tracker("celltrack", "run", params={"seed": 0}).start()
    assert isinstance(handle, Tracker._Noop)
    handle.metric("v", 0.1)  # no-op, must not raise


def test_track_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CELLTRACK_NO_MLFLOW", "1")
    handle = Tracker("celltrack", "run").track_run(tmp_path)
    assert isinstance(handle, Tracker._Noop)


def test_backend_is_none_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CELLTRACK_NO_MLFLOW", "1")
    assert Tracker._backend() is None


def test_backend_is_the_module_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CELLTRACK_NO_MLFLOW", raising=False)
    assert Tracker._backend() is not None


def test_flat_dots_nested_config():
    flat = Tracker._flat({"optim": {"lr": 1e-3, "epochs": 5}, "seed": 0})
    assert flat == {"optim.lr": 1e-3, "optim.epochs": 5, "seed": 0}

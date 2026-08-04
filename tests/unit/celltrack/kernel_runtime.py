from pathlib import Path

import pytest

from celltrack import kernel_runtime


def test_find(monkeypatch: pytest.MonkeyPatch):
    """`find` returns the first mounted path matching the pattern under /kaggle/input."""
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: ["/kaggle/input/a/celltrack/x.py"])
    assert kernel_runtime.find("celltrack/x.py") == "/kaggle/input/a/celltrack/x.py"


def test_install_wheels(monkeypatch: pytest.MonkeyPatch):
    """`install_wheels` pip-installs every bundled wheel offline and returns their names."""
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: ["/kit/a.whl", "/kit/b.whl"])
    seen: list[list[str]] = []
    monkeypatch.setattr(kernel_runtime.subprocess, "run", lambda command, check: seen.append(command))

    wheels = kernel_runtime.install_wheels(Path("/kit/celltrack_src"))

    assert wheels == ["/kit/a.whl", "/kit/b.whl"]
    assert seen[0][1:5] == ["-m", "pip", "install", "--no-index"]
    assert seen[0][-2:] == ["/kit/a.whl", "/kit/b.whl"]


def test_pilkwang_packs(monkeypatch: pytest.MonkeyPatch):
    """`pilkwang_packs` picks the support-pack-50ep seed and the seed314159 seed out of the mounted configs."""
    configs = [
        "/kaggle/input/seed314159-pack/weights/unet_transformer/split_0/config.json",
        "/kaggle/input/support-pack-50ep/weights/unet_transformer/split_0/config.json",
    ]
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: configs)

    pack1, pack2 = kernel_runtime.pilkwang_packs()

    assert "support-pack" in str(pack1)
    assert "seed314159" in str(pack2)


def test_pack_source(monkeypatch: pytest.MonkeyPatch):
    """`pack_source` resolves the mounted pack's src/ root, three levels above the models package init."""
    monkeypatch.setattr(
        kernel_runtime, "find", lambda pattern: "/kaggle/input/p/repo/src/biohub_tracking/models/__init__.py"
    )
    assert kernel_runtime.pack_source() == Path("/kaggle/input/p/repo/src")


def test_test_videos(monkeypatch: pytest.MonkeyPatch):
    """`test_videos` returns the competition test zarrs as sorted Paths."""
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: ["/in/b.zarr", "/in/a.zarr"])
    assert kernel_runtime.test_videos() == [Path("/in/a.zarr"), Path("/in/b.zarr")]


def test_run_submission(monkeypatch: pytest.MonkeyPatch):
    """`run_submission` runs the predictor over each video and writes the graphs to a submission CSV."""

    class _Graph:
        node_ids = (0, 1)
        edges = ((0, 1),)

    class _Submission:
        def __init__(self, graphs: dict[str, object]) -> None:
            self.graphs = graphs

        def write_csv(self, out: str) -> None:
            written["out"] = out
            written["keys"] = tuple(self.graphs)

    written: dict[str, object] = {}
    monkeypatch.setitem(__import__("sys").modules, "core.data.submission", type("_M", (), {"Submission": _Submission}))
    calls: list[tuple[str, Path]] = []

    def predict(name: str, path: Path) -> _Graph:
        calls.append((name, path))
        return _Graph()

    kernel_runtime.run_submission(predict, [Path("/in/m1.zarr"), Path("/in/m2.zarr")], out="/out/s.csv")

    assert calls == [("m1", Path("/in/m1.zarr")), ("m2", Path("/in/m2.zarr"))]
    assert written == {"out": "/out/s.csv", "keys": ("m1", "m2")}

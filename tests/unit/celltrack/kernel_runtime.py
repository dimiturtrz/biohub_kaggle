from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from celltrack import kernel_runtime
from celltrack.kernel_runtime import KernelRuntime
from core.data.tracks import TrackGraph


def test_find(monkeypatch: pytest.MonkeyPatch):
    """`find` returns the first mounted path matching the pattern under /kaggle/input."""
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: ["/kaggle/input/a/celltrack/x.py"])
    assert KernelRuntime.find("celltrack/x.py") == "/kaggle/input/a/celltrack/x.py"


def test_install_wheels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`install_wheels` pip-installs every bundled wheel offline and returns their names."""
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: ["/kit/a.whl", "/kit/b.whl"])
    seen: list[list[str]] = []
    monkeypatch.setattr(kernel_runtime.subprocess, "run", lambda command, check: seen.append(command))

    wheels = KernelRuntime.install_wheels(tmp_path / "celltrack_src")

    assert wheels == ["/kit/a.whl", "/kit/b.whl"]
    assert seen[0][1:5] == ["-m", "pip", "install", "--no-index"]
    assert seen[0][-2:] == ["/kit/a.whl", "/kit/b.whl"]


def test_association_ranker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The re-ranker artifact is found by its MANIFEST, so the kernel names a dataset and not a mount path."""
    mount = tmp_path / "assoc-ranker"
    monkeypatch.setattr(
        kernel_runtime.glob, "glob", lambda pattern, recursive: [str(mount / "ASSOCIATION_RANKER_MANIFEST.json")]
    )
    assert KernelRuntime.association_ranker() == mount

    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: [])
    assert KernelRuntime.association_ranker() is None  # dataset not attached: run without it, don't crash


def test_pilkwang_packs(monkeypatch: pytest.MonkeyPatch):
    """`pilkwang_packs` picks the support-pack-50ep seed and the seed314159 seed out of the mounted configs."""
    configs = [
        "/kaggle/input/seed314159-pack/weights/unet_transformer/split_0/config.json",
        "/kaggle/input/support-pack-50ep/weights/unet_transformer/split_0/config.json",
    ]
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: configs)

    pack1, pack2 = KernelRuntime.pilkwang_packs()

    assert "support-pack" in str(pack1)
    assert "seed314159" in str(pack2)


def test_pack_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`pack_source` resolves the mounted pack's src/ root, three levels above the models package init."""
    init = tmp_path / "repo/src/biohub_tracking/models/__init__.py"
    monkeypatch.setattr(KernelRuntime, "find", lambda pattern: str(init))
    assert KernelRuntime.pack_source() == tmp_path / "repo/src"


def test_center_prior_pack(monkeypatch: pytest.MonkeyPatch):
    """`center_prior_pack` picks the DeepCenter `best.pt` by its dataset token, skipping the detector packs."""
    found = [
        "/kaggle/input/seed314159-pack/weights/unet_transformer/split_0/best.pt",
        "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/full_frame_center/best.pt",
    ]
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: found)
    assert KernelRuntime.center_prior_pack() == Path(found[1]).parent

    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: found[:1])
    assert KernelRuntime.center_prior_pack() is None  # detector-only mount -> no confirmed recovery


def test_secondary_pack(monkeypatch: pytest.MonkeyPatch):
    """`secondary_pack` picks the seed314159 pack by its dataset token, skipping the support-pack seed."""
    configs = [
        "/kaggle/input/support-pack-50ep/weights/unet_transformer/split_0/config.json",
        "/kaggle/input/seed314159-pack/weights/unet_transformer/split_0/config.json",
    ]
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: configs)
    assert KernelRuntime.secondary_pack() == Path(configs[1]).parent

    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: configs[:1])
    assert KernelRuntime.secondary_pack() is None  # secondary seed not attached -> single-seed path


def test_test_videos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`test_videos` returns the competition test zarrs as sorted Paths."""
    unsorted = [str(tmp_path / "b.zarr"), str(tmp_path / "a.zarr")]
    monkeypatch.setattr(kernel_runtime.glob, "glob", lambda pattern, recursive: unsorted)
    assert KernelRuntime.test_videos() == [tmp_path / "a.zarr", tmp_path / "b.zarr"]


def test_run_submission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`run_submission` runs the predictor over each video and writes the graphs to a submission CSV."""

    class _Graph:
        node_ids = (0, 1)
        edges = ((0, 1),)

    class _Submission:
        def __init__(self, graphs: dict[str, object]) -> None:
            self.graphs = graphs

        def write_csv(self, out: Path) -> None:
            written["out"] = out
            written["keys"] = tuple(self.graphs)

    written: dict[str, object] = {}
    monkeypatch.setitem(__import__("sys").modules, "core.data.submission", type("_M", (), {"Submission": _Submission}))
    calls: list[tuple[str, Path]] = []

    def predict(name: str, path: Path) -> _Graph:
        calls.append((name, path))
        return _Graph()

    videos = [tmp_path / "m1.zarr", tmp_path / "m2.zarr"]
    out = tmp_path / "s.csv"
    KernelRuntime.run_submission(cast(Callable[[str, Path], TrackGraph], predict), videos, out=str(out))

    assert calls == [("m1", videos[0]), ("m2", videos[1])]
    assert written == {"out": out, "keys": ("m1", "m2")}

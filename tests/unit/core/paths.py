from pathlib import Path

import pytest

from core.paths import DataRoot


@pytest.fixture
def config(tmp_path: Path) -> Path:
    written = tmp_path / "paths.yaml"
    written.write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    return written


def test_from_config(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CELLTRACK_DATA", raising=False)  # env override wins when set — resolve from paths.yaml here
    assert DataRoot.from_config(config).raw("x") == tmp_path / "root" / "raw" / "x"


def test_from_config_prefers_the_env_override(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A runner without the dataset points CELLTRACK_DATA at an empty dir so data-gated tests skip."""
    monkeypatch.setenv("CELLTRACK_DATA", str(tmp_path / "elsewhere"))
    assert DataRoot.from_config(config).raw("x") == tmp_path / "elsewhere" / "raw" / "x"


def test_raw(tmp_path: Path):
    """`raw` never creates anything — the download is read-only."""
    assert not DataRoot(tmp_path).raw("dataset").exists()


def test_processed(tmp_path: Path):
    destination = DataRoot(tmp_path).processed("dataset")
    assert destination.is_dir()


def test_videos(tmp_path: Path):
    split = tmp_path / "raw" / "biohub_cell_tracking" / "train"
    for name in ("b.zarr", "a.zarr", "a.geff"):
        (split / name).mkdir(parents=True)
    assert [video.name for video in DataRoot(tmp_path).videos("train")] == ["a.zarr", "b.zarr"]


def test_videos_of_a_missing_split_is_empty(tmp_path: Path):
    assert DataRoot(tmp_path).videos("train") == []


def test_track_store(tmp_path: Path):
    assert DataRoot(tmp_path).track_store(Path("x/aaaa_1.zarr")).name == "aaaa_1.geff"


def test_synthetic(tmp_path: Path):
    """A GENERATED corpus is neither downloaded nor derived, so it gets its own top-level area under the root."""
    assert DataRoot(tmp_path).synthetic("biohub_synthetic") == tmp_path / "synthetic" / "biohub_synthetic"

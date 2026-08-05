from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.paths import DataRoot
from tools import dataset_stats
from tools.dataset_stats import (
    BlobScaleConfig,
    BlobScaleSurvey,
    DatasetSurvey,
    load_annotated_video,
)


@pytest.fixture
def root_with_video(video_store: Path, geff_store: Path) -> tuple[DataRoot, Path]:
    """A data root whose train split is the one synthetic video/annotation pair from conftest."""
    root = video_store.parent
    split = root / "raw" / "biohub_cell_tracking" / "train"
    split.mkdir(parents=True)
    for store in (video_store, geff_store):
        store.rename(split / store.name)
    return DataRoot(root), split / video_store.name


def test_load_annotated_video(root_with_video: tuple[DataRoot, Path]):
    """The shared loader returns the video and its GEFF annotation as a matched pair."""
    root, video = root_with_video
    opened, tracks = load_annotated_video(root, video)
    assert isinstance(opened, CellVideo)
    assert isinstance(tracks, AnnotatedTracks)
    assert len(tracks.graph.node_ids) == 3


def test_summarise(root_with_video: tuple[DataRoot, Path]):
    """Motion is reported in micrometres, so the 4x anisotropy has to be applied before the norm."""
    root, video = root_with_video
    summary = DatasetSurvey(root).summarise(video)
    assert summary.annotated_nodes == 3
    assert summary.estimated_nodes == 30
    assert summary.annotated_fraction == 0.1
    assert summary.divisions == 1
    assert summary.median_motion_um == pytest.approx(1.9617, abs=1e-3)
    assert summary.beyond_cutoff_share == 0.0


def test_run(root_with_video: tuple[DataRoot, Path]):
    """A split reduces to one row per video, with the acquisition prefix split out."""
    root, _ = root_with_video
    table = DatasetSurvey(root).run("train", 1)
    assert table["dataset"].to_list() == ["aaaa_00000001"]
    assert table["prefix"].to_list() == ["aaaa"]
    assert table["annotated_nodes"].to_list() == [3]
    assert table["divisions"].to_list() == [1]


def test_peak_sigma_at():
    """A broader blob peaks at a larger LoG sigma than a tight one — the scale it recovers."""
    config = BlobScaleConfig(sigmas_um=np.array([1.0, 3.0, 6.0]))
    survey = BlobScaleSurvey(DataRoot(Path("unused")), config)
    isotropic = Spacing(z=1.0, y=1.0, x=1.0)
    centre = np.array([13, 13, 13])

    tight = np.zeros((27, 27, 27), dtype=np.float32)
    tight[tuple(centre)] = 1.0
    broad = gaussian_filter(tight, sigma=3.0)

    tight_sigma = survey.peak_sigma_at(tight, isotropic, centre)
    broad_sigma = survey.peak_sigma_at(broad, isotropic, centre)
    assert tight_sigma in config.sigmas_um
    assert broad_sigma >= tight_sigma


def test_measure(video_store: Path, in_bounds_tracks: AnnotatedTracks, monkeypatch: pytest.MonkeyPatch):
    """Every sampled ground-truth centre yields one blob-scale reading drawn from the sweep."""
    cell = CellVideo.from_ome_zarr(video_store)
    monkeypatch.setattr(dataset_stats, "load_annotated_video", lambda _root, _video: (cell, in_bounds_tracks))
    config = BlobScaleConfig()
    survey = BlobScaleSurvey(DataRoot(Path("unused")), config)

    peaks = survey.measure([Path("dummy.zarr")], np.random.default_rng(0))

    assert peaks.shape == (len(in_bounds_tracks.graph.node_ids),)
    assert set(peaks.tolist()) <= set(config.sigmas_um.tolist())

from pathlib import Path

import pytest

from celltrack.explore import DatasetSurvey
from core.paths import DataRoot


@pytest.fixture
def survey(video_store: Path, geff_store: Path) -> DatasetSurvey:
    """A root whose train split is the one synthetic video/annotation pair from conftest."""
    root = video_store.parent
    split = root / "raw" / "biohub_cell_tracking" / "train"
    split.mkdir(parents=True)
    for store in (video_store, geff_store):
        store.rename(split / store.name)
    return DatasetSurvey(DataRoot(root))


def test_summarise(survey: DatasetSurvey, tmp_path: Path):
    """Motion is reported in micrometres, so the 4x anisotropy has to be applied before the norm."""
    summary = survey.summarise(tmp_path / "raw" / "biohub_cell_tracking" / "train" / "aaaa_00000001.zarr")
    assert summary.annotated_nodes == 3
    assert summary.estimated_nodes == 30
    assert summary.annotated_fraction == 0.1
    assert summary.divisions == 1
    assert summary.median_motion_um == pytest.approx(1.9617, abs=1e-3)
    assert summary.beyond_cutoff_share == 0.0


def test_run(survey: DatasetSurvey):
    table = survey.run("train", 1)
    assert table["dataset"].to_list() == ["aaaa_00000001"]
    assert table["prefix"].to_list() == ["aaaa"]
    assert table["annotated_nodes"].to_list() == [3]
    assert table["divisions"].to_list() == [1]

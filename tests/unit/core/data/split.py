from pathlib import Path

import pytest

from core.data.split import AcquisitionFolds, VideoSplit

FOLDS = 5
VIDEOS = [Path(f"{prefix}_{index:04x}.zarr") for prefix in ("44b6", "6bba") for index in range(20)]


def test_prefix_of():
    assert AcquisitionFolds.prefix_of(Path("44b6_0113de3b.zarr")) == "44b6"


def test_by_prefix():
    grouped = AcquisitionFolds.by_prefix(VIDEOS)
    assert sorted(grouped) == ["44b6", "6bba"]
    assert len(grouped["44b6"]) == len(grouped["6bba"]) == 20


def test_rank_of():
    """Rank comes from the name, so a fold is reproducible without a seed or a stored id list."""
    assert AcquisitionFolds.rank_of(Path("44b6_0113de3b.zarr")) == AcquisitionFolds.rank_of(
        Path("elsewhere/44b6_0113de3b.zarr")
    )


def test_split():
    split = AcquisitionFolds(folds=FOLDS).split(VIDEOS, index=0)
    assert len(split.validation) == len(VIDEOS) // FOLDS
    assert len(split.train) + len(split.validation) == len(VIDEOS)


def test_split_holds_out_the_validation_videos():
    """The held-out videos are never fitted on — that is the whole point of the partition."""
    split = AcquisitionFolds(folds=FOLDS).split(VIDEOS, index=2)
    assert not set(split.train) & set(split.validation)


def test_split_covers_every_video_exactly_once_across_folds():
    folds = AcquisitionFolds(folds=FOLDS)
    held_out = [video for index in range(FOLDS) for video in folds.split(VIDEOS, index).validation]
    assert sorted(held_out) == sorted(VIDEOS)


def test_split_puts_both_acquisitions_in_every_fold():
    """The test split is two videos of each prefix, so a fold that holds only one would not mirror it."""
    folds = AcquisitionFolds(folds=FOLDS)
    for index in range(FOLDS):
        assert sorted(folds.split(VIDEOS, index).validation_by_prefix()) == ["44b6", "6bba"]


def test_split_is_deterministic():
    """Membership follows from the names, so the order the videos are listed in cannot change a fold."""
    folds = AcquisitionFolds(folds=FOLDS)
    shuffled = folds.split(list(reversed(VIDEOS)), index=1)
    assert set(folds.split(VIDEOS, index=1).validation) == set(shuffled.validation)


def test_split_rejects_a_fold_outside_the_range():
    with pytest.raises(ValueError, match="outside"):
        AcquisitionFolds(folds=FOLDS).split(VIDEOS, index=FOLDS)


def test_validation_by_prefix():
    split = VideoSplit(train=(), validation=(Path("44b6_a.zarr"), Path("6bba_b.zarr"), Path("44b6_c.zarr")))
    assert split.validation_by_prefix() == {
        "44b6": (Path("44b6_a.zarr"), Path("44b6_c.zarr")),
        "6bba": (Path("6bba_b.zarr"),),
    }

from pathlib import Path

from core.data.split import Acquisitions, StratifiedSplit

VIDEOS = [Path(f"{prefix}_{index:04x}.zarr") for prefix in ("44b6", "6bba") for index in range(20)]


def test_prefix_of():
    assert Acquisitions.prefix_of(Path("44b6_0113de3b.zarr")) == "44b6"


def test_by_prefix():
    grouped = Acquisitions.by_prefix(VIDEOS)
    assert sorted(grouped) == ["44b6", "6bba"]
    assert len(grouped["44b6"]) == len(grouped["6bba"]) == 20


def test_rank_of():
    """Rank comes from the name, so any ordering is reproducible without a seed or a stored id list."""
    assert Acquisitions.rank_of(Path("44b6_0113de3b.zarr")) == Acquisitions.rank_of(
        Path("elsewhere/44b6_0113de3b.zarr")
    )


def test_partition():
    """Each acquisition contributes its test, then validation, then the rest to train — counts add up."""
    split = StratifiedSplit(test_per_prefix=2, validation_per_prefix=1).partition(VIDEOS)
    assert len(split.test) == 4
    assert len(split.validation) == 2
    assert len(split.train) == len(VIDEOS) - 6
    assert not (set(split.train) & set(split.validation)) and not (set(split.train) & set(split.test))


def test_partition_is_stratified_and_deterministic():
    """Both acquisitions appear in every partition, and the split follows the names, not the listing order."""
    stratified = StratifiedSplit(test_per_prefix=2, validation_per_prefix=1)
    split = stratified.partition(VIDEOS)
    assert sorted(Acquisitions.by_prefix(split.test)) == ["44b6", "6bba"]
    assert set(stratified.partition(list(reversed(VIDEOS))).test) == set(split.test)

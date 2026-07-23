"""The dataset facts the split and the metric arguments rest on, checked against the real download.

These assertions read the competition data, so they are skipped wherever it is absent (CI, a fresh
clone). They exist because the numbers are load-bearing: the two acquisitions differ by an order of
magnitude in annotation density, which is why validation is stratified by prefix rather than pooled.
"""

from pathlib import Path

import pytest

from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.paths import DataRoot

CONFIG = Path(__file__).parents[2] / "paths.yaml"
TRAIN_VIDEOS = 199
TEST_VIDEOS = 4
PREFIX_COUNTS = {"44b6": 71, "6bba": 128}
ANNOTATED_NODES = 133318


def data_root() -> DataRoot:
    if not CONFIG.exists():
        pytest.skip(f"no {CONFIG.name}; the competition data is not configured on this machine")
    root = DataRoot.from_config(CONFIG)
    if not root.videos("train"):
        pytest.skip("the competition data is not downloaded on this machine")
    return root


def test_the_train_split_holds_both_acquisitions():
    videos = data_root().videos("train")
    assert len(videos) == TRAIN_VIDEOS
    assert {prefix: len(group) for prefix, group in AcquisitionFolds.by_prefix(videos).items()} == PREFIX_COUNTS


def test_the_test_split_holds_both_acquisitions():
    """Acquisition identity is present on both sides of the official split, so validation must be too."""
    videos = data_root().videos("test")
    assert len(videos) == TEST_VIDEOS
    assert sorted(AcquisitionFolds.by_prefix(videos)) == sorted(PREFIX_COUNTS)


def test_the_annotation_totals_are_what_the_split_assumes():
    root = data_root()
    tracks = [AnnotatedTracks.from_geff(root.track_store(video)) for video in root.videos("train")]
    assert sum(len(entry.graph.node_ids) for entry in tracks) == ANNOTATED_NODES

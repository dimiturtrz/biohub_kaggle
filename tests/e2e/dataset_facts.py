"""The dataset facts the split and the metric arguments rest on, checked against the real download.

These assertions read the competition data, so they are skipped wherever it is absent (CI, a fresh
clone). They exist because the numbers are load-bearing: the two acquisitions differ by an order of
magnitude in annotation density, which is why validation is stratified by prefix rather than pooled.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from celltrack.eval.bracket import Ceiling, ValidationFold
from core.data.split import AcquisitionFolds
from core.data.submission import Submission
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.paths import DataRoot

CEILING_FLOOR = 0.97

CONFIG = Path(__file__).parents[2] / "paths.yaml"
TRAIN_VIDEOS = 199
TEST_VIDEOS = 4
PREFIX_COUNTS = {"44b6": 71, "6bba": 128}
ANNOTATED_NODES = 133318
_SAMPLE_SUBMISSION = "sample_submission.csv"


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


def test_the_linking_ceiling_is_near_perfect():
    """Linking ground-truth nodes recovers almost every edge, so detection, not linking, is the budget."""
    fold = ValidationFold.load(data_root(), fold=0)
    ceiling = Ceiling.with_gate(fold.spacing, gate_um=15.0).over(fold.annotations)
    assert ceiling.edge_jaccard() > CEILING_FLOOR


def test_our_submission_matches_the_official_format():
    """Our flattened submission is column- and dtype-identical to the competition's sample — the convention holds."""
    root = data_root()
    sample_path = root.videos("test")[0].parent.parent / _SAMPLE_SUBMISSION
    if not sample_path.exists():
        pytest.skip("sample_submission.csv is not present")
    sample = pl.read_csv(sample_path)
    graph = TrackGraph(
        node_ids=np.array([0, 1, 2]),
        coordinates=np.array([[t, 0, 0, 0] for t in range(3)]),
        edges=np.array([[0, 1], [1, 2]]),
    )
    ours = Submission(graphs={"44b6_0113de3b": graph}).to_frame()
    assert ours.columns == sample.columns
    assert ours.dtypes == sample.dtypes

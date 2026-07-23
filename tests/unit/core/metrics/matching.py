import numpy as np

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching

ANISOTROPIC = Spacing(z=4.0, y=1.0, x=1.0)


def graph_of(coordinates: list[list[int]], edges: list[list[int]] | None = None) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates),
        edges=np.array(edges if edges else [], dtype=np.int64).reshape(-1, 2),
    )


def test_is_matched():
    assert NodeMatching(gt_rows=np.array([2, UNMATCHED])).is_matched().tolist() == [True, False]


def test_matched_count():
    assert NodeMatching(gt_rows=np.array([2, UNMATCHED, 0])).matched_count() == 2


def test_match():
    truth = graph_of([[0, 0, 0, 0], [1, 0, 0, 0]])
    prediction = graph_of([[0, 0, 1, 0], [1, 0, 0, 2]])
    matching = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0)).match(prediction, truth)
    assert matching.gt_rows.tolist() == [0, 1]


def test_match_respects_the_distance_cutoff():
    """A predicted cell further than the cutoff has no partner, however isolated it is."""
    truth = graph_of([[0, 0, 0, 0]])
    prediction = graph_of([[0, 0, 100, 0]])
    matching = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0)).match(prediction, truth)
    assert matching.gt_rows.tolist() == [UNMATCHED]


def test_match_measures_distance_in_micrometres():
    """Five voxels apart in z is 20 um under 4 um voxels — outside the cutoff, unlike the same gap in y."""
    truth = graph_of([[0, 0, 0, 0]])
    matcher = DistanceMatcher(spacing=ANISOTROPIC)
    assert matcher.match(graph_of([[0, 5, 0, 0]]), truth).gt_rows.tolist() == [UNMATCHED]
    assert matcher.match(graph_of([[0, 0, 5, 0]]), truth).gt_rows.tolist() == [0]


def test_match_pairs_within_a_timepoint_only():
    """Cells at different timepoints never pair, however close they sit."""
    truth = graph_of([[0, 0, 0, 0]])
    prediction = graph_of([[1, 0, 0, 0]])
    matching = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0)).match(prediction, truth)
    assert matching.gt_rows.tolist() == [UNMATCHED]


def test_match_is_one_to_one():
    """Two predictions on one annotated cell: the assignment keeps one, leaving the other unmatched."""
    truth = graph_of([[0, 0, 0, 0]])
    prediction = graph_of([[0, 0, 0, 0], [0, 0, 1, 0]])
    matching = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0)).match(prediction, truth)
    assert sorted(matching.gt_rows.tolist()) == [UNMATCHED, 0]
    assert matching.matched_count() == 1

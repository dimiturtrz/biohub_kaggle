import math

import numpy as np

from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing
from core.metrics.divisions import DivisionCounts
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics

MATCHER = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0))
PERFECT = EdgeCounts(tp=10, fp=0, fn=0)
NO_DIVISIONS = DivisionCounts(tp=0, fp=0, fn=0)


def metrics(edges: EdgeCounts, predicted: int, estimated: int, divisions: DivisionCounts = NO_DIVISIONS):
    return VideoMetrics(edges=edges, divisions=divisions, predicted_nodes=predicted, estimated_nodes=estimated)


def test_total_node_ratio():
    assert metrics(PERFECT, predicted=110, estimated=100).total_node_ratio() == 0.1


def test_total_node_ratio_is_undefined_without_an_estimate():
    ratio = metrics(PERFECT, predicted=110, estimated=0).total_node_ratio()
    assert math.isnan(ratio)


def test_adjusted_edge_jaccard():
    """Ten percent too many nodes costs one percent of a perfect Jaccard."""
    assert metrics(PERFECT, predicted=110, estimated=100).adjusted_edge_jaccard() == 0.99


def test_adjusted_edge_jaccard_rewards_predicting_too_few():
    """The adjustment is signed, so undershooting the estimate scales the Jaccard above itself."""
    assert metrics(PERFECT, predicted=90, estimated=100).adjusted_edge_jaccard() == 1.01


def test_adjusted_edge_jaccard_never_goes_negative():
    assert metrics(PERFECT, predicted=1100, estimated=100).adjusted_edge_jaccard() == 0.0


def test_video_metrics_of():
    coordinates = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    graph = TrackGraph(node_ids=np.array([0, 1]), coordinates=coordinates, edges=np.array([[0, 1]]))
    truth = AnnotatedTracks(graph=graph, estimated_node_count=20)
    sample = VideoMetrics.of(graph, truth, MATCHER)
    assert sample.edges == EdgeCounts(tp=1, fp=0, fn=0)
    assert sample.predicted_nodes == 2
    assert sample.adjusted_edge_jaccard() == 1.09


def test_split_score_of():
    """Videos are weighted by how many edge events they carry, not counted equally."""
    dense = metrics(EdgeCounts(tp=90, fp=10, fn=0), predicted=100, estimated=100)
    sparse = metrics(EdgeCounts(tp=0, fp=0, fn=1), predicted=100, estimated=100)
    assert SplitScore.of([dense, sparse]).adjusted_edge_jaccard == 0.9 * 100 / 101


def test_split_score_of_drops_the_division_term_when_no_divisions_exist():
    sample = metrics(PERFECT, predicted=100, estimated=100)
    split = SplitScore.of([sample])
    assert math.isnan(split.division_jaccard)
    assert split.score == 1.0


def test_split_score_of_weights_divisions_at_one_tenth():
    sample = metrics(PERFECT, predicted=100, estimated=100, divisions=DivisionCounts(tp=1, fp=1, fn=0))
    assert SplitScore.of([sample]).score == 1.05

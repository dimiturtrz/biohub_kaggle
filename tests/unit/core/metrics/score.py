import math

import numpy as np
import pytest

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


def test_clamped_edge_jaccard():
    """Over-detection is penalised identically — the clamp only ever touches the bonus half of the factor."""
    over = metrics(PERFECT, predicted=110, estimated=100)
    assert over.clamped_edge_jaccard() == 0.99
    assert over.clamped_edge_jaccard() == over.adjusted_edge_jaccard()


def test_clamped_edge_jaccard_pays_nothing_for_under_detection():
    """Undershooting earns the neutral factor 1.0, so the faithful score sits strictly above the clamped one."""
    under = metrics(PERFECT, predicted=90, estimated=100)
    assert under.clamped_edge_jaccard() == 1.0
    assert under.adjusted_edge_jaccard() == 1.01


def test_clamped_edge_jaccard_is_undefined_without_an_estimate():
    assert math.isnan(metrics(PERFECT, predicted=110, estimated=0).clamped_edge_jaccard())


def test_the_two_scores_can_rank_two_candidates_differently():
    """The whole point: the faithful metric prefers the model that finds FEWER cells; the clamped one does not.

    A detects every node it should (ratio 0) at Jaccard 0.88. B trims 30% of the nodes, losing links for a
    Jaccard of 0.86, and is handed a 1.03 factor for it — 0.8858, which beats A. Under the clamp B keeps only
    its 0.86 and A wins. On the hidden, densely annotated evaluation B's missing cells cost real Jaccard, so
    the clamped ordering is the one that transfers.
    """
    honest = metrics(EdgeCounts(tp=88, fp=12, fn=0), predicted=100, estimated=100)
    trimmed = metrics(EdgeCounts(tp=86, fp=14, fn=0), predicted=70, estimated=100)

    assert trimmed.adjusted_edge_jaccard() > honest.adjusted_edge_jaccard()
    assert trimmed.clamped_edge_jaccard() < honest.clamped_edge_jaccard()
    assert SplitScore.of([trimmed]).score > SplitScore.of([honest]).score
    assert SplitScore.of([trimmed]).selection_score < SplitScore.of([honest]).selection_score


def test_the_faithful_metric_is_pinned():
    """The competition number must never drift behind the selection variant — pin it on a known input."""
    sample = metrics(EdgeCounts(tp=86, fp=14, fn=0), predicted=70, estimated=100)
    assert sample.adjusted_edge_jaccard() == pytest.approx(0.8858)
    assert SplitScore.of([sample]).score == pytest.approx(0.8858)
    assert SplitScore.of([sample]).adjusted_edge_jaccard == pytest.approx(0.8858)
    divided = metrics(PERFECT, predicted=110, estimated=100, divisions=DivisionCounts(tp=1, fp=1, fn=0))
    assert SplitScore.of([divided]).score == pytest.approx(0.99 + 0.1 * 0.5)


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


def test_split_score_of_aggregates_the_clamped_form_the_same_way():
    """`selection_score` is the same weighted mean and the same division term over the clamped per-video number."""
    under = metrics(PERFECT, predicted=80, estimated=100, divisions=DivisionCounts(tp=1, fp=1, fn=0))
    split = SplitScore.of([under])
    assert split.adjusted_edge_jaccard == pytest.approx(1.02)
    assert split.clamped_edge_jaccard == 1.0
    assert split.score == pytest.approx(1.07)
    assert split.selection_score == pytest.approx(1.05)

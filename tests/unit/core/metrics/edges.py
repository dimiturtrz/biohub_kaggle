import math

import numpy as np

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher

UNIT = Spacing(z=1.0, y=1.0, x=1.0)
MATCHER = DistanceMatcher(spacing=UNIT)

TRACK = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
LINKS = [[0, 1], [1, 2]]


def graph_of(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)),
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def counted(prediction: TrackGraph, truth: TrackGraph) -> EdgeCounts:
    return EdgeCounts.of(prediction, truth, MATCHER.match(prediction, truth))


def test_jaccard():
    assert EdgeCounts(tp=2, fp=1, fn=1).jaccard() == 0.5


def test_jaccard_is_undefined_without_events():
    assert math.isnan(EdgeCounts(tp=0, fp=0, fn=0).jaccard())


def test_weight():
    assert EdgeCounts(tp=2, fp=1, fn=1).weight() == 4


def test_edge_counts_of():
    truth = graph_of(TRACK, LINKS)
    assert counted(truth, truth) == EdgeCounts(tp=2, fp=0, fn=0)


def test_edge_counts_of_without_a_prediction():
    truth = graph_of(TRACK, LINKS)
    empty = graph_of([], [])
    assert counted(empty, truth) == EdgeCounts(tp=0, fp=0, fn=2)


def test_edge_counts_of_ignores_links_in_unannotated_tissue():
    """Predicted links whose endpoints match nothing are invisible to the metric — recall there is free."""
    truth = graph_of(TRACK, LINKS)
    prediction = graph_of([*TRACK, [0, 500, 0, 0], [1, 500, 0, 0]], [*LINKS, [3, 4]])
    assert counted(prediction, truth) == EdgeCounts(tp=2, fp=0, fn=0)


def test_edge_counts_of_charges_a_wrong_link_between_annotated_cells():
    """Linking two annotated cells that the annotation says are not connected is a false positive."""
    truth = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 9, 0]], [[0, 1]])
    prediction = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 9, 0]], [[0, 2]])
    assert counted(prediction, truth) == EdgeCounts(tp=0, fp=1, fn=1)


def test_edge_counts_of_collapses_duplicate_links():
    truth = graph_of(TRACK, LINKS)
    prediction = graph_of(TRACK, [*LINKS, *LINKS])
    assert counted(prediction, truth) == EdgeCounts(tp=2, fp=0, fn=0)


def test_edge_counts_of_drops_links_that_do_not_span_one_timepoint():
    """Backward links and links that skip a frame are not links this metric recognises."""
    truth = graph_of(TRACK, LINKS)
    prediction = graph_of(TRACK, [*LINKS, [1, 0], [0, 2]])
    assert counted(prediction, truth) == EdgeCounts(tp=2, fp=0, fn=0)


def test_edge_counts_of_caps_a_node_at_two_children():
    """A cell has at most two daughters, so further outgoing links are not evaluable predictions."""
    truth = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 3, 0], [1, 0, 6, 0]], [[0, 1]])
    prediction = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 3, 0], [1, 0, 6, 0]], [[0, 2], [0, 3], [0, 1]])
    assert counted(prediction, truth) == EdgeCounts(tp=0, fp=2, fn=1)

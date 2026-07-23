import math

import numpy as np

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.divisions import DivisionCounts, DivisionScoring
from core.metrics.matching import DistanceMatcher

MATCHER = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0))

# grandparent -> divider -> two daughters -> two grandchildren, the daughters far enough apart to tell apart
DIVIDING_NODES = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 9, 0], [2, 0, -9, 0], [3, 0, 9, 0], [3, 0, -9, 0]]
DIVIDING_EDGES = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]


def graph_of(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)),
        coordinates=np.array(coordinates),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def dividing() -> TrackGraph:
    return graph_of(DIVIDING_NODES, DIVIDING_EDGES)


def test_jaccard():
    assert DivisionCounts(tp=1, fp=1, fn=2).jaccard() == 0.25


def test_jaccard_is_undefined_without_divisions():
    assert math.isnan(DivisionCounts(tp=0, fp=0, fn=0).jaccard())


def test_counts():
    truth = dividing()
    assert DivisionScoring.of(truth, truth, MATCHER).counts() == DivisionCounts(tp=1, fp=0, fn=0)


def test_counts_misses_an_unforked_prediction():
    """A prediction that follows only one daughter never forks, so the division is a false negative."""
    prediction = graph_of([*DIVIDING_NODES[:3], DIVIDING_NODES[4]], [[0, 1], [1, 2], [2, 3]])
    assert DivisionScoring.of(prediction, dividing(), MATCHER).counts() == DivisionCounts(tp=0, fp=0, fn=1)


def test_counts_forgives_a_fork_one_timepoint_late():
    """When a cell visibly splits is subjective, so the window accepts a fork a timepoint off."""
    prediction = graph_of(DIVIDING_NODES, [[0, 1], [1, 2], [2, 4], [2, 5]])
    assert DivisionScoring.of(prediction, dividing(), MATCHER).counts() == DivisionCounts(tp=1, fp=0, fn=0)


def test_counts_charges_a_fork_on_an_annotated_cell_that_does_not_divide():
    """The annotation continues through this cell without splitting, so the extra branch is spurious."""
    truth = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], [[0, 1], [1, 2]])
    prediction = graph_of([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 30, 0]], [[0, 1], [1, 2], [1, 3]])
    assert DivisionScoring.of(prediction, truth, MATCHER).counts() == DivisionCounts(tp=0, fp=1, fn=0)


def test_division_scoring_of():
    """Both graphs are indexed once up front, so every division window is scored against the same structure."""
    scoring = DivisionScoring.of(dividing(), dividing(), MATCHER)
    assert scoring.annotated.dividing_rows().tolist() == [1]
    assert scoring.predicted.successors[1] == (2, 3)

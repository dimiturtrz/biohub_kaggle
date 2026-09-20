import math

import numpy as np

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.divisions import DivisionCounts, DivisionReach, DivisionScoring
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


def test_total():
    """The ruler-wide split is the per-movie ones summed, stage by stage."""
    parts = [DivisionReach(1, 2, 3, 4, 5, 6, 7), DivisionReach(10, 20, 30, 40, 50, 60, 70)]
    assert DivisionReach.total(parts) == DivisionReach(11, 22, 33, 44, 55, 66, 77)


def test_missed():
    assert (
        DivisionReach(
            recovered=1, nodes_missing=2, no_fork=3, fork_rejected=4, unproposable=1, no_kept_child=1, spurious=5
        ).missed()
        == 9
    )


def test_selection_bound():
    """Both node-bearing stages are recoverable by a decision over nodes we already predicted."""
    reach = DivisionReach(
        recovered=1, nodes_missing=5, no_fork=3, fork_rejected=2, unproposable=1, no_kept_child=1, spurious=4
    )
    assert reach.selection_bound() == 5


def test_reach():
    truth = dividing()
    assert DivisionScoring.of(truth, truth, MATCHER).reach() == DivisionReach(
        recovered=1, nodes_missing=0, no_fork=0, fork_rejected=0, unproposable=0, no_kept_child=0, spurious=0
    )


def test_reach_counts_an_orphan_daughter_as_still_proposable():
    """The second lineage is present but unparented, so the orphan-pool recovery stage could still reach it."""
    prediction = graph_of(DIVIDING_NODES, [[0, 1], [1, 2], [2, 4], [3, 5]])
    reach = DivisionScoring.of(prediction, dividing(), MATCHER).reach()
    assert (reach.no_fork, reach.unproposable) == (1, 0)


def test_reach_counts_a_claimed_daughter_as_unproposable():
    """Another track already parents the second daughter, so no orphan-pool proposal can name it."""
    prediction = graph_of([*DIVIDING_NODES, [1, 0, -9, 0]], [[0, 1], [1, 2], [2, 4], [6, 3], [3, 5]])
    reach = DivisionScoring.of(prediction, dividing(), MATCHER).reach()
    assert (reach.no_fork, reach.unproposable) == (1, 1)


def test_reach_counts_a_childless_parent_as_having_no_kept_child():
    """The linker ended the parent's track, so the stage has no kept child to add a second daughter beside."""
    prediction = graph_of(DIVIDING_NODES, [[2, 4], [3, 5]])
    reach = DivisionScoring.of(prediction, dividing(), MATCHER).reach()
    assert (reach.no_fork, reach.no_kept_child) == (1, 1)


def test_reach_blames_the_detector_when_a_daughter_lineage_is_absent():
    """One daughter never appears in the prediction, so no linking decision could have recovered the division."""
    prediction = graph_of([*DIVIDING_NODES[:3], DIVIDING_NODES[4]], [[0, 1], [1, 2], [2, 3]])
    assert DivisionScoring.of(prediction, dividing(), MATCHER).reach().nodes_missing == 1


def test_reach_blames_selection_when_the_nodes_are_there_but_nothing_forks():
    """Both daughters are predicted and merely linked into one track — a fork is a decision away."""
    prediction = graph_of(DIVIDING_NODES, [[0, 1], [1, 2], [2, 4], [3, 5]])
    reach = DivisionScoring.of(prediction, dividing(), MATCHER).reach()
    assert (reach.nodes_missing, reach.no_fork) == (0, 1)


def test_division_scoring_of():
    """Both graphs are indexed once up front, so every division window is scored against the same structure."""
    scoring = DivisionScoring.of(dividing(), dividing(), MATCHER)
    assert scoring.annotated.dividing_rows().tolist() == [1]
    assert scoring.predicted.successors[1] == (2, 3)

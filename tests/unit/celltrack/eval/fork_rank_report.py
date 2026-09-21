"""Equivalence classes of a fork's grading: resolved by the annotation, unresolved, or a real division."""

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from celltrack.eval.fork_rank_report import (
    Branch,
    BranchAccuracy,
    Branching,
    ForkGrading,
    LabelledFork,
    MatchedPair,
    MovieGraph,
    RankReport,
)
from celltrack.postproc.fork_ranking import ForkCandidate


class _Keys:
    """The handful of `tracksdata.DEFAULT_ATTR_KEYS` names this module reads."""

    NODE_ID = "node_id"
    MATCHED_NODE_ID = "matched_node_id"
    EDGE_SOURCE = "source"
    EDGE_TARGET = "target"
    MATCHED_EDGE_MASK = "matched_mask"


class _FakeGraph:
    """A duck-typed stand-in for a matched tracksdata graph: columns in, columns out."""

    def __init__(
        self,
        nodes: Mapping[str, Sequence[float]],
        edges: Mapping[str, Sequence[float]],
        children: Mapping[int, Sequence[int]],
    ):
        self._nodes = nodes
        self._edges = edges
        self._children = children

    def node_ids(self) -> list[int]:
        return [int(node) for node in self._nodes["node_id"]]

    def node_attrs(self, attr_keys: list[str]) -> Mapping[str, Sequence[float]]:
        return {key: self._nodes[key] for key in attr_keys}

    def edge_attrs(self) -> Mapping[str, Sequence[float]]:
        return self._edges

    def successors(self, node: int) -> list[int]:
        return list(self._children.get(int(node), ()))

    def out_degree(self, nodes: list[int]) -> list[int]:
        return [len(self._children.get(int(node), ())) for node in nodes]


def _movie_graph() -> MovieGraph:
    return MovieGraph(
        index_of={10: 0, 11: 1, 12: 2},
        times={10: 0, 11: 1, 12: 1},
        positions_um=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 4.0]]),
        successors={10: [11, 12]},
        probability={(10, 11): 0.9, (10, 12): 0.2},
        gt_edges=frozenset({(10, 12)}),
        annotated_of={10: 5, 12: 7},
        gt_children={5: frozenset({7})},
        gt_out_degree={5: 1},
    )


def _branch(probability: float, *, gt: bool, frames: int = 1, distance: float = 1.0) -> Branch:
    return Branch(probability=probability, frames_ahead=frames, distance_um=distance, is_gt_edge=gt)


def _fork(probability: float, *, true: bool, charged: bool) -> LabelledFork:
    candidate = ForkCandidate(
        parent=0,
        kept=1,
        child=2,
        probability=probability,
        positions_um=np.zeros((3, 3)),
        kept_frames=1,
        child_frames=1,
    )
    return LabelledFork(candidate=candidate, is_true=true, is_charged=charged)


def test_truth_index():
    branching = Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True)))

    assert branching.truth_index() == 1


def test_truth_index_is_none_when_no_branch_is_annotated():
    branching = Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=False)))

    assert branching.truth_index() is None


def test_truth_index_is_none_at_a_real_division_because_cutting_to_one_is_wrong_either_way():
    branching = Branching(branches=(_branch(0.9, gt=True), _branch(0.2, gt=True)))

    assert branching.truth_index() is None


def test_branch_accuracy_of():
    resolved_against = Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True)))
    resolved_for = Branching(branches=(_branch(0.8, gt=True), _branch(0.1, gt=False)))
    unresolved = Branching(branches=(_branch(0.7, gt=False), _branch(0.6, gt=False)))

    graded = BranchAccuracy.of(
        "probability", lambda branch: branch.probability, [resolved_against, resolved_for, unresolved]
    )

    assert graded.decided == 2
    assert graded.correct == 1


def test_lost_edges():
    resolved_against = Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True)))
    resolved_for = Branching(branches=(_branch(0.8, gt=True), _branch(0.1, gt=False)))

    graded = BranchAccuracy.of("probability", lambda branch: branch.probability, [resolved_against, resolved_for])

    assert graded.lost_edges() == 1


def test_accuracy():
    graded = BranchAccuracy.of("probability", lambda branch: branch.probability, [])

    assert graded.accuracy() != graded.accuracy()


def test_branch_accuracy_of_lets_a_future_reading_cue_beat_affinity():
    fork = Branching(branches=(_branch(0.9, gt=False, frames=1), _branch(0.2, gt=True, frames=40)))

    by_probability = BranchAccuracy.of("probability", lambda branch: branch.probability, [fork])
    by_survival = BranchAccuracy.of("survival", lambda branch: float(branch.frames_ahead), [fork])

    assert by_probability.correct == 0
    assert by_survival.correct == 1


def test_division_jaccard():
    assert RankReport.division_jaccard(kept=6, admitted_false=10, total_true=11) == 6 / 21


def test_best_budget():
    report = RankReport(name="cue", total=100, true_ranks=(0, 90), charged_above=(0, 60))

    budget, value = report.best_budget(total_true=11)

    assert budget == 1
    assert value == 1 / 11


def test_rank_report_of():
    forks = [
        _fork(0.9, true=False, charged=True),
        _fork(0.8, true=False, charged=False),
        _fork(0.7, true=True, charged=True),
    ]

    report = RankReport.of("probability", _ProbabilityByCandidate(), forks)

    assert report.total == 3
    assert report.true_ranks == (2,)
    assert report.charged_above == (1,)


def test_worst_rank():
    report = RankReport(name="cue", total=100, true_ranks=(0, 90), charged_above=(0, 60))

    assert report.worst_rank() == 90


def test_kept_at():
    report = RankReport(name="cue", total=100, true_ranks=(0, 90), charged_above=(0, 60))

    assert report.kept_at(90) == 1
    assert report.kept_at(91) == 2


class _ProbabilityByCandidate:
    """The cheapest `ForkRanking`: likeliest fork first, no geometry read."""

    def cost(self, candidate: ForkCandidate) -> float:
        return -candidate.probability


def _grading() -> ForkGrading:
    return ForkGrading(
        forks=(_fork(0.9, true=True, charged=True), _fork(0.8, true=False, charged=True)),
        branchings=(
            Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True))),
            Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=False))),
        ),
    )


def test_total_true():
    assert _grading().total_true() == 1


def test_charged_false():
    assert _grading().charged_false() == 1


def test_resolved():
    assert _grading().resolved() == 1


def test_gradings():
    grading = ForkGrading(
        forks=(_fork(0.9, true=True, charged=True),),
        branchings=(Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True))),),
    )

    assert [graded.name for graded in grading.gradings()][0] == "probability"
    assert len(grading.gradings()) == 5


def test_reports():
    grading = ForkGrading(
        forks=(_fork(0.9, true=True, charged=True),),
        branchings=(Branching(branches=(_branch(0.9, gt=False), _branch(0.2, gt=True))),),
    )

    assert [report.name for report in grading.reports()][0] == "probability"
    assert len(grading.reports()) == 7


def test_frames_ahead():
    ahead = _movie_graph().frames_ahead()

    assert ahead[11] == 1
    assert ahead[10] == 2


def test_forks():
    labelled, branchings = _movie_graph().forks()

    assert len(labelled) == 1
    assert labelled[0].is_charged
    assert not labelled[0].is_true
    assert branchings[0].truth_index() == 1


def test_forks_label_a_fork_true_when_its_annotation_divides():
    dividing = dataclasses.replace(_movie_graph(), gt_out_degree={5: 2})

    labelled, _ = dividing.forks()

    assert labelled[0].is_true


def test_forks_keep_the_likeliest_branch_in_the_candidate():
    labelled, _ = _movie_graph().forks()

    assert labelled[0].candidate.kept == 1
    assert labelled[0].candidate.child == 2


def test_movie_graph():
    predicted = _FakeGraph(
        nodes={
            "node_id": [10, 11, 12],
            "matched_node_id": [5, -1, 7],
            "t": [0, 1, 1],
            "z": [0, 0, 0],
            "y": [0, 0, 0],
            "x": [0, 1, 4],
        },
        edges={"source": [10, 10], "target": [11, 12], "edge_prob": [0.9, 0.2], "matched_mask": [0, 1]},
        children={10: [11, 12]},
    )
    truth = _FakeGraph(nodes={"node_id": [5, 7]}, edges={}, children={5: [7]})

    graph = MatchedPair(predicted=predicted, truth=truth, keys=_Keys(), scale=(1.0, 1.0, 1.0, 1.0)).movie_graph()

    assert graph.annotated_of == {10: 5, 12: 7}
    assert graph.gt_edges == frozenset({(10, 12)})
    assert graph.gt_out_degree == {5: 1, 7: 0}
    assert graph.times[11] == 1
    assert graph.forks()[1][0].truth_index() == 1

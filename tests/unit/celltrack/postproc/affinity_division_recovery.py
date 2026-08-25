import math
from typing import cast

import numpy as np
import pytest
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity
from celltrack.postproc.affinity_division_recovery import (
    AffinityDivisionConfig,
    AffinityDivisionRecovery,
    GeometricCandidacy,
    RunnerUpCandidacy,
    _FrameProposals,
)
from celltrack.postproc.fork_ranking import (
    ForkCandidate,
    GeometryRanking,
    ProbabilityRanking,
    SplitSymmetryRanking,
    _Gap,
)
from core.data.tracks import TrackGraph
from core.geometry import Spacing


class FakeAffinity:
    """An `EdgeAffinity` from an explicit per-gap matrix, rows the t sources, columns the t+1 targets."""

    def __init__(self, by_timepoint: dict[int, Float[np.ndarray, "s t"]]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        return self._by_timepoint.get(timepoint)


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def recovery(matrix: list[list[float]], **overrides: object) -> AffinityDivisionRecovery:
    settings: dict[str, object] = {
        "ranking": ProbabilityRanking(),
        "candidacy": RunnerUpCandidacy(min_kept_prob=0.5, min_second_prob=0.1),
        "parent_gate_um": 5.0,
        "sister_gate_um": 5.0,
        "existing_child_gate_um": math.inf,
        "max_added_forks": 1000,
    }
    return AffinityDivisionRecovery(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        affinity=FakeAffinity({0: np.array(matrix, dtype=np.float32)}),
        **(settings | overrides),  # pyrefly: ignore[bad-argument-type]
    )


def kernel_recovery(by_timepoint: dict[int, list[list[float]]], **overrides: object) -> AffinityDivisionRecovery:
    """The 0.926-kernel replica: geometric candidacy, `d(p,c)+0.15*d(c1,c)` score, kernel gates, `kernel_faithful`."""
    settings: dict[str, object] = {
        "ranking": GeometryRanking(sister_weight=0.15),
        "candidacy": GeometricCandidacy(),
        "parent_gate_um": 8.0,
        "sister_gate_um": 11.0,
        "existing_child_gate_um": 10.0,
        "max_added_forks": None,
        "c3_divergence_um": 2.25,
        "kernel_faithful": True,
    }
    return AffinityDivisionRecovery(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        affinity=FakeAffinity({t: np.array(matrix, dtype=np.float32) for t, matrix in by_timepoint.items()}),
        **(settings | overrides),  # pyrefly: ignore[bad-argument-type]
    )


# One parent (row 0) at t=0; two targets at t=1 — the kept child (row 1) and an orphan runner-up (row 2).
_MITOSIS = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2]]
_KEPT_TOP = [[0.7, 0.2]]


def test_build():
    """`AffinityDivisionConfig.build` binds its gates and the per-video affinity into the recovery stage."""
    affinity = cast(EdgeAffinity, FakeAffinity({}))
    stage = AffinityDivisionConfig(min_second_prob=0.6, parent_gate_um=4.0).build(
        Spacing(z=1.0, y=1.0, x=1.0), affinity, 6, 10.0
    )

    assert isinstance(stage, AffinityDivisionRecovery)
    assert stage.affinity is affinity  # the video's edge-head probabilities are threaded through
    assert stage.candidacy == RunnerUpCandidacy(min_kept_prob=0.0, min_second_prob=0.6)  # the floor rides the candidacy
    assert stage.parent_gate_um == 4.0


def test_parent_gate():
    """Unset, the mother-to-daughter gate IS the linker's — a fork is one frame's travel, like any other link."""
    assert AffinityDivisionConfig().parent_gate(10.0) == 10.0
    assert AffinityDivisionConfig(parent_gate_um=4.7).parent_gate(10.0) == 4.7


def test_sister_gate():
    """Unset, the daughter-to-daughter gate is twice the parent gate — the triangle inequality, not a threshold."""
    assert AffinityDivisionConfig().sister_gate(10.0) == 20.0
    assert AffinityDivisionConfig(parent_gate_um=7.0).sister_gate(10.0) == 14.0  # what a submission set by hand
    assert AffinityDivisionConfig(sister_gate_um=7.2).sister_gate(10.0) == 7.2


def test_existing_child_gate():
    """Unset, the mother's-own-link gate is unbounded — and says so with None, which JSON can carry.

    It used to say it with `math.inf`. Pydantic serialises infinity to JSON `null`, so a config holding it
    dumped and then failed to validate back (`float` rejects None) — a latent break in the run's provenance
    that stayed invisible while no shipped config carried a division block.
    """
    assert AffinityDivisionConfig().existing_child_gate() == math.inf
    assert AffinityDivisionConfig(existing_child_gate_um=7.8).existing_child_gate() == 7.8

    config = AffinityDivisionConfig()
    assert AffinityDivisionConfig.model_validate_json(config.model_dump_json()) == config


def test_transform():
    """The parent's top target is its kept child and its second is an unparented cell — a fork is proposed."""
    forked = recovery(_KEPT_TOP).transform(graph(_MITOSIS, [[0, 1]]))
    assert forked.edges.tolist() == [[0, 10], [0, 20]]
    assert forked.division_parents().tolist() == [0]


def test_rejects_when_runner_up_is_below_the_floor():
    """A second target the head barely scores is not a daughter — the probability floor rejects it."""
    assert recovery([[0.7, 0.05]]).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_when_kept_child_is_not_the_top_target():
    """If the head's first choice is not the linker's kept child the parent is not a confident divider."""
    assert recovery([[0.2, 0.7]]).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_when_kept_child_probability_is_low():
    """A parent whose own primary link the head is unsure of does not earn a second daughter."""
    low_kept = recovery([[0.4, 0.35]])  # helper's runner-up candidacy floors the kept link at 0.5
    assert low_kept.transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_a_runner_up_that_is_already_parented():
    """Only an unparented cell can be a missing second daughter — a linked target is someone's child already."""
    # Row 3 (t=0, far) already links to row 2, so the runner-up is not an orphan.
    coordinates = [*_MITOSIS, [0, 0, 0, 20]]
    two_source_top = [[0.7, 0.2], [0.1, 0.6]]
    forked = recovery(two_source_top).transform(graph(coordinates, [[0, 1], [3, 2]]))
    assert forked.edges.tolist() == [[0, 10], [30, 20]]


def test_rejects_a_runner_up_beyond_the_sister_gate():
    """The two daughters must sit close — a runner-up far from the kept child fails the sister gate."""
    far_sister = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 4]]
    assert recovery(_KEPT_TOP, sister_gate_um=2.0).transform(graph(far_sister, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_a_runner_up_beyond_the_parent_gate():
    """A candidate too far from the mother is an appearance, not a division."""
    far_parent = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 4]]
    gated = recovery(_KEPT_TOP, parent_gate_um=2.0, sister_gate_um=10.0)
    assert gated.transform(graph(far_parent, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_honours_the_global_cap():
    """A zero budget adds nothing however confident the head is — the cap bounds speculation.

    A three-node fixture reaches this state on the DERIVED budget too (0.113% of three nodes rounds to zero),
    which is the shipped behaviour for a tiny movie: too few cells to expect a division is too few to buy one.
    """
    assert recovery(_KEPT_TOP, max_added_forks=0).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_a_parent_whose_existing_child_is_far():
    """A mother already reaching a long way to her kept child is not a divider — the existing-child gate."""
    gated = recovery(_KEPT_TOP, existing_child_gate_um=0.5)
    assert gated.transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_frames_ahead_counts_a_daughters_whole_future():
    """The candidate carries how long each daughter's track runs — read off the linked graph, not walked twice."""
    # Parent 0 at t=0; kept child 1 and orphan 2 at t=1; the orphan continues to 3 at t=2.
    coordinates = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2], [2, 0, 0, 2]]
    forked = recovery(_KEPT_TOP).transform(graph(coordinates, [[0, 1], [2, 3]]))

    assert forked.edges.tolist() == [[0, 10], [20, 30], [0, 20]]


def test_budget():
    """The edge fraction and the absolute ceiling, whichever binds first."""
    stage = recovery(_KEPT_TOP, max_added_forks=100)

    assert stage.budget(73_697) == 100  # an explicit ceiling overrides the derivation, which is what an arm sets


def test_geometry_ranking_survives_the_cap_a_probability_ranking_evicts():
    """The whole fix: under a budget of one, geometry keeps the clean split the head scores lowest.

    Parent 0 forks to an orphan on the OPPOSITE side of its kept child at low probability; parent 3 forks to
    a same-side orphan the head likes far more. Probability admits the crowd's favourite, geometry the split.
    """
    coordinates = [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, -1],  # parent 0's low-probability, opposite-side candidate
        [0, 0, 0, 10],
        [1, 0, 0, 11],
        [1, 0, 0, 12],  # parent 3's confident, same-side candidate
    ]
    edges = [[0, 1], [3, 4]]
    matrix = [[0.9, 0.2, 0.0, 0.0], [0.0, 0.0, 0.8, 0.7]]
    one_fork = {"max_added_forks": 1}  # helper candidacy already floors the runner-up at 0.1

    by_probability = recovery(matrix, **one_fork).transform(graph(coordinates, edges))
    by_geometry = recovery(matrix, ranking=GeometryRanking(sister_weight=0.15), **one_fork)
    by_symmetry = recovery(matrix, ranking=SplitSymmetryRanking(parent_gate_um=5.0), **one_fork)

    assert by_probability.division_parents().tolist() == [30]
    assert by_geometry.transform(graph(coordinates, edges)).division_parents().tolist() == [0]
    assert by_symmetry.transform(graph(coordinates, edges)).division_parents().tolist() == [0]


def test_admits_a_contested_daughter_only_once():
    """Two mothers naming the same orphan is not two divisions — the better-ranked one claims it."""
    coordinates = [
        [0, 0, 0, 0],
        [0, 0, 0, 4],  # the second mother, further from the contested cell
        [1, 0, 0, 1],
        [1, 0, 0, 5],
        [1, 0, 0, -1],  # contested: the runner-up of both mothers
    ]
    edges = [[0, 2], [1, 3]]
    matrix = [[0.9, 0.0, 0.4], [0.0, 0.9, 0.4]]

    forked = recovery(matrix, ranking=GeometryRanking(sister_weight=0.15), parent_gate_um=6.0, sister_gate_um=7.0)

    assert forked.transform(graph(coordinates, edges)).division_parents().tolist() == [0]


def test_rejects_an_unknown_ranking():
    """A misspelled `--set division.ranking=…` is a typo, not a silent fallback to the default."""
    affinity = cast(EdgeAffinity, FakeAffinity({}))
    with pytest.raises(ValueError, match="nearest"):
        AffinityDivisionConfig(ranking="nearest").build(Spacing(z=1.0, y=1.0, x=1.0), affinity, 6, 10.0)


def test_rejects_an_unknown_candidacy():
    """A misspelled `--set division.candidacy=…` is a typo, not a silent fallback to the head-gated default."""
    affinity = cast(EdgeAffinity, FakeAffinity({}))
    with pytest.raises(ValueError, match="unknown division candidacy"):
        AffinityDivisionConfig(candidacy="head").build(Spacing(z=1.0, y=1.0, x=1.0), affinity, 6, 10.0)


def test_geometric_candidacy_recovers_a_sister_the_head_ranks_below_a_false_orphan():
    """The candidacy difference that motivates the arm: a true sister buried below a false orphan by the head.

    Row 1 is the kept child, row 2 a FALSE orphan the head ranks second but which sits beyond the parent gate,
    row 3 the TRUE sister the head ranks third but which is within every gate. Head-gated `RunnerUpCandidacy`
    proposes only the head's second choice (row 2), which the gate then rejects — no fork. `GeometricCandidacy`
    proposes every orphan, so the true sister reaches the gates and is recovered.
    """
    coordinates = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 9], [1, 0, 0, 2]]
    head = [[0.7, 0.2, 0.1]]
    head_gated = recovery(head).transform(graph(coordinates, [[0, 1]]))
    assert head_gated.edges.tolist() == [[0, 10]]
    geometric = recovery(head, candidacy=GeometricCandidacy()).transform(graph(coordinates, [[0, 1]]))
    assert geometric.edges.tolist() == [[0, 10], [0, 30]]


def _one_gap() -> _Gap:
    """A single frame gap: parent row 0, its kept child row 1, one orphan row 2, one parented non-child row 3."""
    return _Gap(
        targets=np.array([1, 2, 3]),
        orphan=np.array([False, True, False]),
        positions_um=np.zeros((4, 3)),
        frames_ahead=np.array([9, 8, 7, 6]),
        successor=np.array([-1, -1, -1, -1]),
    )


def test_runner_up_candidacy_candidates():
    """Head-gated: the top target must be the kept child, and only its orphan runner-up is proposed."""
    gap = _one_gap()
    probabilities = np.array([0.9, 0.5, 0.1])  # top=kept (row1), runner-up=orphan row2, row3 last

    proposed = RunnerUpCandidacy().candidates(parent=0, kept=1, probabilities=probabilities, gap=gap)

    assert [(candidate.parent, candidate.child) for candidate in proposed] == [(0, 2)]


def test_runner_up_candidacy_candidates_declines_when_top_is_not_the_kept_child():
    """If the head's most confident target is not the kept child, this is no division — nothing is proposed."""
    gap = _one_gap()
    probabilities = np.array([0.3, 0.9, 0.1])  # top target is row2, not the kept child row1

    assert RunnerUpCandidacy().candidates(parent=0, kept=1, probabilities=probabilities, gap=gap) == []


def test_geometric_candidacy_candidates():
    """Geometric: every unparented target that is not the kept child is proposed, head opinion aside."""
    gap = _one_gap()
    probabilities = np.array([0.3, 0.9, 0.1])  # order is irrelevant to geometric candidacy

    proposed = GeometricCandidacy().candidates(parent=0, kept=1, probabilities=probabilities, gap=gap)

    assert [(candidate.parent, candidate.child) for candidate in proposed] == [(0, 2)]


def test_budget_derives_the_ceiling_from_the_measured_division_rate():
    """With no explicit ceiling the budget IS the expected number of true divisions, per movie.

    0.113% per node-observation is our own corpus's rate (151 over 133,318 annotated nodes), not a published
    one — 2D nuclei report ~0.33% and developmental 3D ~1.1%, three to ten times higher, because a division
    rate is a property of the observation window as much as the biology. Importing one would over-budget
    every movie by that factor, which is exactly the scale error that made the frontier's evidence bonus a
    no-op when we ported it.
    """
    derived = recovery(_KEPT_TOP, max_added_forks=None)

    assert derived.budget(73_697) == 83  # the dense movie's detections -> ~83 expected divisions
    assert derived.budget(10_703) == 12  # a sparse movie budgets proportionally less


def test_budget_folds_in_the_edge_fraction_ceiling():
    """The frontier's global fork cap (~0.375% of edges) is folded into the budget beside the node-rate one."""
    capped = recovery(_KEPT_TOP, max_added_forks=1000, max_fork_edge_fraction=0.00375)

    assert capped.budget(1000, 100_000) == 375  # edge fraction is the smaller bound here
    assert capped.budget(1000, 100) == 0  # 0.375% of 100 edges rounds to nothing
    assert recovery(_KEPT_TOP, max_added_forks=1000).budget(1000, 100_000) == 1000  # no fraction -> node budget


# A true division: parent (row 0) at t=0, kept child (row 1) and orphan runner-up (row 2) at t=1, and both
# daughters continue to t=2 (rows 3, 4). The head ranks the kept child top and the orphan second.
def _diverging_graph(child_t2_x: int):
    coordinates = [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 3], [2, 0, 0, 0], [2, 0, 0, child_t2_x]]
    return graph(coordinates, [[0, 1], [1, 3], [2, 4]])


def test_c3_divergence_gate_admits_a_diverging_pair_and_rejects_a_static_one():
    """A fork survives the C3 gate only when its daughters go on to move apart by at least the threshold."""
    diverging = recovery(_KEPT_TOP, require_c3_divergence=True).transform(_diverging_graph(child_t2_x=6))
    assert [0, 20] in diverging.edges.tolist()  # sep grows 3 -> 6 (+3um >= 2.25) -> fork kept

    static = recovery(_KEPT_TOP, require_c3_divergence=True).transform(_diverging_graph(child_t2_x=3))
    assert [0, 20] not in static.edges.tolist()  # sep 3 -> 3 (no growth) -> fork rejected

    off = recovery(_KEPT_TOP).transform(_diverging_graph(child_t2_x=3))
    assert [0, 20] in off.edges.tolist()  # gate off -> the static pair is still forked


# A parent (row 0) whose orphan runner-up (row 2) is NOT its kept child's nearest cell: a decoy orphan
# (row 3) sits closer to the kept child (row 1) than the true candidate does.
_INTERLOPER = [[0, 0, 0, 6], [1, 0, 0, 4], [1, 0, 0, 8], [1, 0, 0, 5]]
_KEPT_TOP_WITH_DECOY = [[0.7, 0.2, 0.1]]


def test_mutual_nearest_gate_rejects_a_fork_whose_daughters_are_not_each_others_nearest():
    """The kept child's nearest orphan is the decoy, not the proposed daughter, so the mutual-nearest gate fails."""
    guarded = recovery(_KEPT_TOP_WITH_DECOY, require_mutual_nearest=True).transform(graph(_INTERLOPER, [[0, 1]]))
    assert guarded.edges.tolist() == [[0, 10]]  # runner-up not mutual-nearest with the kept child -> no fork

    off = recovery(_KEPT_TOP_WITH_DECOY).transform(graph(_INTERLOPER, [[0, 1]]))
    assert off.edges.tolist() == [[0, 10], [0, 20]]  # gate off -> the runner-up is forked despite the decoy


def test_mutual_nearest_gate_admits_a_clean_pair():
    """With no closer orphan, the kept child and the proposed daughter ARE each other's nearest — the fork holds."""
    clean = [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 2]]
    forked = recovery(_KEPT_TOP, require_mutual_nearest=True).transform(graph(clean, [[0, 1]]))
    assert forked.edges.tolist() == [[0, 10], [0, 20]]


# A canonical division that passes EVERY kernel gate at once: a mid-track parent (row 1, predecessor row 0) at
# t=1, its kept child (row 2) and one orphan sister (row 3) at t=2, each continuing to a single successor at t=3
# (rows 4, 5) that move apart. Rows 4/5 sit +/-3 so the separation grows 2 -> 6 (>= 2.25). The affinity matrix
# is keyed at the parent's frame (t=1); geometric candidacy ignores its values.
_KERNEL_DIVISION = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 1], [2, 0, 0, -1], [3, 0, 0, 3], [3, 0, 0, -3]]
_KERNEL_EDGES = [[0, 1], [1, 2], [2, 4], [3, 5]]
_KERNEL_AFFINITY = {1: [[0.5, 0.5]]}


def test_kernel_faithful_c1_rejects_a_track_start_parent():
    """C1: the kernel forks only off a MID-track parent. The only variable is whether the parent has a predecessor.

    C1 cannot be unconditional — the OFF-path fixtures fork off track-start parents and expect it — so it is
    gated behind `kernel_faithful`. It only ever REMOVES forks; here it removes the one off a start parent.
    """
    mid_track = kernel_recovery(_KERNEL_AFFINITY).transform(graph(_KERNEL_DIVISION, _KERNEL_EDGES))
    assert [10, 30] in mid_track.edges.tolist()  # parent has predecessor row 0 -> mid-track -> fork kept

    start_parent = kernel_recovery(_KERNEL_AFFINITY).transform(graph(_KERNEL_DIVISION, _KERNEL_EDGES[1:]))
    assert [10, 30] not in start_parent.edges.tolist()  # drop the predecessor edge -> track start -> C1 rejects


def test_kernel_faithful_c3_requires_single_successors_at_t_plus_two():
    """C3: both daughters must continue at EXACTLY t+2 as single-successor nodes, else there is no divergence."""
    diverging = kernel_recovery(_KERNEL_AFFINITY).transform(graph(_KERNEL_DIVISION, _KERNEL_EDGES))
    assert [10, 30] in diverging.edges.tolist()  # single successors at t+2, separation grows -> fork kept

    # The kept child's successor sits at t=4, not t+2 (a gap-spanning link) -> no readable divergence -> rejected.
    gap_spanning = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 1], [2, 0, 0, -1], [4, 0, 0, 3], [3, 0, 0, -3]]
    spanned = kernel_recovery(_KERNEL_AFFINITY).transform(graph(gap_spanning, _KERNEL_EDGES))
    assert [10, 30] not in spanned.edges.tolist()

    # The candidate (row 3) has TWO successors (rows 5, 6) -> not a single-successor node -> C3 rejects.
    multi = [*_KERNEL_DIVISION, [3, 0, 0, -4]]
    multi_succ = kernel_recovery(_KERNEL_AFFINITY).transform(graph(multi, [*_KERNEL_EDGES, [3, 6]]))
    assert [10, 30] not in multi_succ.edges.tolist()


# A fork the kernel's ONE-directional C2 admits but our SYMMETRIC gate rejects: the candidate (row 3) IS the kept
# child's nearest orphan, but the candidate's own nearest cell is a decoy orphan (row 4), not the kept child.
_C2_DIVERGENCE = [
    [0, 0, 0, 1],  # 0 predecessor of the parent
    [1, 0, 0, 1],  # 1 mid-track parent
    [2, 0, 0, 0],  # 2 kept child
    [2, 0, 0, 2],  # 3 candidate: the kept child's nearest orphan
    [2, 0, 0, 3],  # 4 decoy orphan: closer to the candidate than the kept child is
    [3, 0, 0, -2],  # 5 kept child's successor
    [3, 0, 0, 4],  # 6 candidate's successor (they diverge)
]
_C2_EDGES = [[0, 1], [1, 2], [2, 5], [3, 6]]
_C2_AFFINITY = {1: [[0.5, 0.5, 0.5]]}


def test_kernel_faithful_c2_is_one_directional_where_ours_is_symmetric():
    """C2 one-directional accepts the kernel-set fork; our symmetric mutual-nearest rejects the very same one."""
    kernel = kernel_recovery(_C2_AFFINITY).transform(graph(_C2_DIVERGENCE, _C2_EDGES))
    assert [10, 30] in kernel.edges.tolist()  # candidate IS the kept child's nearest orphan -> one-directional admits

    symmetric = AffinityDivisionRecovery(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        affinity=FakeAffinity({1: np.array([[0.5, 0.5, 0.5]], dtype=np.float32)}),
        ranking=GeometryRanking(sister_weight=0.15),
        candidacy=GeometricCandidacy(),
        parent_gate_um=8.0,
        sister_gate_um=11.0,
        existing_child_gate_um=10.0,
        max_added_forks=1000,
        require_mutual_nearest=True,  # our symmetric variant, kernel_faithful off
    )
    assert [10, 30] not in symmetric.transform(graph(_C2_DIVERGENCE, _C2_EDGES)).edges.tolist()


def test_kernel_faithful_two_cap_binds_per_frame():
    """The per-frame cap `round(single_child_parents * 0.0076)` (min one) admits one fork per frame under a
    generous global cap — four valid candidates across two frames yield two forks, the cheaper in each frame."""
    positions = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],  # frame 1 cand a: parent 0, kept 1, child 2 -> cost 1.0
            [0, 0, 0],
            [0, 0, 2],
            [0, 0, 2],  # frame 1 cand b: parent 3, kept 4, child 5 -> cost 2.0
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 3],  # frame 2 cand a: parent 6, kept 7, child 8 -> cost 3.3
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],  # frame 2 cand b: parent 9, kept 10, child 11 -> cost 1.0
        ],
        dtype=float,
    )

    def fork(parent: int, kept: int, child: int) -> ForkCandidate:
        return ForkCandidate(
            parent=parent,
            kept=kept,
            child=child,
            probability=0.5,
            positions_um=positions,
            kept_frames=1,
            child_frames=1,
        )

    frame_one = _FrameProposals([fork(0, 1, 2), fork(3, 4, 5)], single_child_parents=2)  # frame_cap -> 1
    frame_two = _FrameProposals([fork(6, 7, 8), fork(9, 10, 11)], single_child_parents=2)  # frame_cap -> 1

    admitted, cap = kernel_recovery({})._admitted_two_cap([frame_one, frame_two], edge_count=100_000)

    assert cap == round(100_000 * 0.00375)  # global cap 375 is generous -> the per-frame cap is what binds
    assert sorted(fork.child for fork in admitted) == [2, 11]  # one per frame: the cheaper split each


def test_build_kernel_faithful_replicates_the_kernel_recipe():
    """`kernel_faithful=True` builds the kernel replica, ignoring the configured stage's ranking/candidacy/gates."""
    affinity = cast(EdgeAffinity, FakeAffinity({}))
    stage = AffinityDivisionConfig(kernel_faithful=True, ranking="probability", parent_gate_um=4.0).build(
        Spacing(z=1.0, y=1.0, x=1.0), affinity, 6, 10.0
    )

    assert stage.kernel_faithful is True
    assert stage.candidacy == GeometricCandidacy()
    assert stage.ranking == GeometryRanking(sister_weight=0.15)  # the kernel's score, not the configured `probability`
    assert (stage.parent_gate_um, stage.sister_gate_um, stage.existing_child_gate_um) == (8.0, 11.0, 10.0)
    assert stage.c3_divergence_um == 2.25

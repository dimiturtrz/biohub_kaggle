import math
from typing import cast

import numpy as np
import pytest
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity
from celltrack.postproc.affinity_division_recovery import (
    AffinityDivisionConfig,
    AffinityDivisionRecovery,
    ForkCandidate,
    GeometryRanking,
    ProbabilityRanking,
    SplitSymmetryRanking,
    SurvivingDaughterRanking,
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
        "min_second_prob": 0.1,
        "parent_gate_um": 5.0,
        "sister_gate_um": 5.0,
        "existing_child_gate_um": math.inf,
        "max_added_forks": 1000,
        "min_kept_prob": 0.5,
    }
    return AffinityDivisionRecovery(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        affinity=FakeAffinity({0: np.array(matrix, dtype=np.float32)}),
        **(settings | overrides),  # pyrefly: ignore[bad-argument-type]
    )


def candidate(kept: list[float], child: list[float], probability: float = 0.5, child_frames: int = 1) -> ForkCandidate:
    """A fork whose mother sits at the origin, with both daughters placed in micrometres around her."""
    return ForkCandidate(
        parent=0,
        kept=1,
        child=2,
        probability=probability,
        positions_um=np.array([[0.0, 0.0, 0.0], kept, child]),
        kept_frames=1,
        child_frames=child_frames,
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
    assert (stage.min_second_prob, stage.parent_gate_um) == (0.6, 4.0)


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
    low_kept = recovery([[0.4, 0.35]], min_kept_prob=0.5)
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


def test_parent_distance_um():
    """Mother to the proposed new daughter, in micrometres."""
    assert candidate([1.0, 0.0, 0.0], [0.0, 3.0, 4.0]).parent_distance_um() == 5.0


def test_sister_distance_um():
    """The separation between the kept child and the candidate."""
    assert candidate([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).sister_distance_um() == 4.0


def test_existing_child_distance_um():
    """Mother to the child she already keeps."""
    assert candidate([0.0, 0.0, 3.0], [1.0, 0.0, 0.0]).existing_child_distance_um() == 3.0


def test_daughter_angle_cosine():
    """-1 when the daughters leave on opposite sides, 0 at a right angle, 1 when they leave together."""
    assert candidate([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).daughter_angle_cosine() == -1.0
    assert candidate([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]).daughter_angle_cosine() == 0.0
    assert candidate([1.0, 0.0, 0.0], [3.0, 0.0, 0.0]).daughter_angle_cosine() == 1.0


def test_split_imbalance():
    """Zero when the daughters balance about the mother, one when they leave in the same direction."""
    assert candidate([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).split_imbalance() == 0.0
    assert candidate([1.0, 0.0, 0.0], [3.0, 0.0, 0.0]).split_imbalance() == 1.0
    # Equal displacements at a right angle: the half-angle form, cos(45 degrees).
    assert math.isclose(candidate([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).split_imbalance(), 0.5 * math.sqrt(2))


def test_probability_ranking_cost():
    """The head's most confident second daughter sorts first, so its cost is the negated probability."""
    assert ProbabilityRanking().cost(candidate([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], probability=0.3)) == -0.3


def test_geometry_ranking_cost():
    """The frontier's `parent_dist + w * sister_dist`, so a tight pair close to the mother sorts first."""
    fork = candidate([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert GeometryRanking(sister_weight=0.15).cost(fork) == 1.0 + 0.15 * 2.0


def test_split_symmetry_ranking_cost():
    """Relative reach from the mother plus centroid imbalance — an opposite, balanced split scores lowest."""
    ranking = SplitSymmetryRanking(parent_gate_um=4.0)
    opposite = ranking.cost(candidate([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]))
    aligned = ranking.cost(candidate([2.0, 0.0, 0.0], [2.0, 0.0, 0.0]))

    assert opposite == 0.5  # imbalance 0: the daughters' centre of mass is the mother
    assert aligned == 1.5  # same reach, but both daughters left together
    assert opposite < aligned


def test_surviving_daughter_ranking_cost():
    """A daughter that dies at the fork pays the full penalty; one that reaches the length rule pays none."""
    ranking = SurvivingDaughterRanking(SplitSymmetryRanking(parent_gate_um=4.0), min_track_length=6)
    split = ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0])
    survives = ranking.cost(candidate(*split, child_frames=6))
    dies = ranking.cost(candidate(*split, child_frames=1))
    halfway = ranking.cost(candidate(*split, child_frames=3))

    assert survives == 0.5  # exactly the base ranking's cost: a surviving daughter is not penalised at all
    assert dies == 1.5  # the shortfall spans the whole reachable range
    assert survives < halfway < dies


def test_surviving_daughter_ranking_cost_does_not_reward_outliving_the_rule():
    """The penalty saturates at the length rule, so a long track cannot buy its way past a bad split."""
    ranking = SurvivingDaughterRanking(SplitSymmetryRanking(parent_gate_um=4.0), min_track_length=6)
    split = ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0])

    assert ranking.cost(candidate(*split, child_frames=60)) == ranking.cost(candidate(*split, child_frames=6))


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
    one_fork = {"max_added_forks": 1, "min_second_prob": 0.1}

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

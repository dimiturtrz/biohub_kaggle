import math

import numpy as np
import pytest

from celltrack.postproc.fork_ranking import (
    ForkCandidate,
    GeometryRanking,
    ProbabilityRanking,
    SplitSymmetryRanking,
    SurvivingDaughterRanking,
    _Gap,
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


def _one_gap() -> _Gap:
    """A single frame gap: parent row 0, its kept child row 1, one orphan row 2, one parented non-child row 3."""
    return _Gap(
        targets=np.array([1, 2, 3]),
        orphan=np.array([False, True, False]),
        positions_um=np.zeros((4, 3)),
        frames_ahead=np.array([9, 8, 7, 6]),
        successor=np.array([-1, -1, -1, -1]),
        claimed_by=np.array([0, -1, 3]),
        claim_probability=np.array([0.9, 0.0, 0.4]),
    )


def test_taking():
    """A fork declared as displacing names the source row it takes its daughter from, and changes nothing else."""
    fork = ForkCandidate.at_gap(parent=0, kept=1, child=2, probability=0.5, gap=_one_gap())
    taken = fork.taking(3)
    assert (fork.displaced, taken.displaced) == (-1, 3)
    assert taken.child == fork.child


def test_at_gap():
    """The fork factory reads each daughter's surviving-frame count out of the gap it is built over."""
    gap = _one_gap()

    fork = ForkCandidate.at_gap(parent=0, kept=1, child=2, probability=0.5, gap=gap)

    assert (fork.parent, fork.kept, fork.child) == (0, 1, 2)
    assert (fork.kept_frames, fork.child_frames) == (8, 7)  # frames_ahead[1], frames_ahead[2]


def _fork_with_futures(positions: list[list[float]], kept_successor: int, child_successor: int) -> ForkCandidate:
    """A fork whose two daughters (rows 1, 2) continue to the t+2 rows named, positions in micrometres."""
    return ForkCandidate(
        parent=0,
        kept=1,
        child=2,
        probability=0.5,
        positions_um=np.array(positions, dtype=float),
        kept_frames=1,
        child_frames=1,
        kept_successor=kept_successor,
        child_successor=child_successor,
    )


def test_divergence_um():
    """C3: the daughters sit 2um apart at t+1 and 6um apart at t+2, so the separation grows by 4um."""
    fork = _fork_with_futures(
        [[0, 0, 0], [0, 0, 1], [0, 0, 3], [0, 0, 0], [0, 0, 6]], kept_successor=3, child_successor=4
    )
    assert fork.divergence_um() == pytest.approx(4.0)  # sep t+2 (6) minus sep t+1 (2)


def test_divergence_um_is_none_when_a_daughter_track_ends():
    """No t+2 node for a daughter means there is no growth to read — the C3 gate treats it as unproven."""
    fork = _fork_with_futures([[0, 0, 0], [0, 0, 1], [0, 0, 3]], kept_successor=3, child_successor=-1)
    assert fork.divergence_um() is None

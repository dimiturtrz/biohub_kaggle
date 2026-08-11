import numpy as np

from celltrack.linkers.motion_linking import VELOCITY_DAMPING
from celltrack.linkers.motion_prediction import MotionPrediction

# Three frames on the x axis: one cell per frame at 0, 4, 8 — a constant +4 um step.
_TIMEPOINTS = np.array([0, 1, 2], dtype=np.int64)
_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0]], dtype=np.float64)


class _Affinity:
    """A fixed per-gap probability matrix, indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def test_distances():
    """A confident +4 um previous step moves the source half a step forward, so the true successor prices nearer.

    The source at x=4 arrived from x=0 with probability 1, so its expected incoming step is +4 and its predicted
    position is `4 + 0.5*4 = 6`. The target at x=8 is 4 um away raw and 2 um away predicted — the whole point of
    the term, and the number that pins the damping to the motion linker's argued half step.
    """
    term = MotionPrediction(affinity=_Affinity({0: np.array([[1.0]], dtype=np.float64)}))

    predicted = term.distances(1, np.array([1]), np.array([2]), _TIMEPOINTS, _POSITIONS)

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 4.0]]


def test_distances_shrink_the_step_when_the_previous_gap_was_ambiguous():
    """Unnormalised by design: half the incoming mass predicts half the step, not a confident wrong direction.

    `PriorVelocity.expected_incoming` weights the previous displacement by its probability without renormalising,
    so a source the head only half-believes in gets a +2 step rather than +4, and the priced distance degrades
    toward the raw one instead of overshooting.
    """
    term = MotionPrediction(affinity=_Affinity({0: np.array([[0.5]], dtype=np.float64)}))

    predicted = term.distances(1, np.array([1]), np.array([2]), _TIMEPOINTS, _POSITIONS)

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 2.0]]


def test_distances_without_a_previous_frame_are_the_raw_ones():
    """The video's first gap has no history, so the prediction is exactly the source's own position."""
    term = MotionPrediction(affinity=_Affinity({0: np.array([[1.0]], dtype=np.float64)}))

    assert term.distances(0, np.array([0]), np.array([1]), _TIMEPOINTS, _POSITIONS).tolist() == [[4.0]]


def test_distances_without_a_scored_previous_gap_are_the_raw_ones():
    """A previous frame the affinity does not score leaves the pair on raw distance — no invented velocity."""
    term = MotionPrediction(affinity=_Affinity({}))

    assert term.distances(1, np.array([1]), np.array([2]), _TIMEPOINTS, _POSITIONS).tolist() == [[4.0]]

import numpy as np

from celltrack.linkers.frame_gap import FrameGap
from celltrack.linkers.motion_linking import VELOCITY_DAMPING
from celltrack.linkers.motion_prediction import MotionPrediction

# Three frames on the x axis: one cell per frame at 0, 4, 8 — a constant +4 um step.
_TIMEPOINTS = np.array([0, 1, 2], dtype=np.int64)
_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0]], dtype=np.float64)


def _gap(timepoint: int, source: int, target: int, timepoints: np.ndarray, positions_um: np.ndarray) -> FrameGap:
    """One single-source, single-target gap over the given frames — the bundle `MotionPrediction.distances` reads."""
    return FrameGap(
        timepoint=timepoint,
        sources=np.array([source]),
        targets=np.array([target]),
        positions_um=positions_um,
        timepoints=timepoints,
    )


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

    predicted = term.distances(_gap(1, 1, 2, _TIMEPOINTS, _POSITIONS))

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 4.0]]


def test_distances_shrink_the_step_when_the_previous_gap_was_ambiguous():
    """Unnormalised by design: half the incoming mass predicts half the step, not a confident wrong direction.

    `PriorVelocity.expected_incoming` weights the previous displacement by its probability without renormalising,
    so a source the head only half-believes in gets a +2 step rather than +4, and the priced distance degrades
    toward the raw one instead of overshooting.
    """
    term = MotionPrediction(affinity=_Affinity({0: np.array([[0.5]], dtype=np.float64)}))

    predicted = term.distances(_gap(1, 1, 2, _TIMEPOINTS, _POSITIONS))

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 2.0]]


def test_distances_without_a_previous_frame_are_the_raw_ones():
    """The video's first gap has no history, so the prediction is exactly the source's own position."""
    term = MotionPrediction(affinity=_Affinity({0: np.array([[1.0]], dtype=np.float64)}))

    assert term.distances(_gap(0, 0, 1, _TIMEPOINTS, _POSITIONS)).tolist() == [[4.0]]


def test_distances_without_a_scored_previous_gap_are_the_raw_ones():
    """A previous frame the affinity does not score leaves the pair on raw distance — no invented velocity."""
    term = MotionPrediction(affinity=_Affinity({}))

    assert term.distances(_gap(1, 1, 2, _TIMEPOINTS, _POSITIONS)).tolist() == [[4.0]]


# Four frames on the x axis at 0, 4, 8, 12 — a constant +4 step, one cell per frame linked with probability 1.
_FOUR_TIMEPOINTS = np.array([0, 1, 2, 3], dtype=np.int64)
_FOUR_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0], [0.0, 0.0, 12.0]], dtype=np.float64)
_CONFIDENT_CHAIN = {0: np.array([[1.0]]), 1: np.array([[1.0]]), 2: np.array([[1.0]])}


def test_history_one_is_the_single_step_form():
    """History 1 reads only the previous gap — byte-identical to the original single-step velocity."""
    single = MotionPrediction(affinity=_Affinity(_CONFIDENT_CHAIN), history=1)

    predicted = single.distances(_gap(2, 2, 3, _FOUR_TIMEPOINTS, _FOUR_POSITIONS))

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 4.0]]  # one +4 step, same as history absent


def test_history_averages_the_velocity_over_several_gaps():
    """A multi-gap history averages the past steps — on a constant-velocity chain the average is the same +4.

    The source at x=8 arrived +4 from x=4, which arrived +4 from x=0. Averaging both gaps still gives +4, so a
    clean trajectory is unchanged; the point is that on a NOISY chain the average beats down the per-step noise,
    which a single step cannot. Verified here on the noiseless case so the chaining arithmetic is pinned exactly.
    """
    multi = MotionPrediction(affinity=_Affinity(_CONFIDENT_CHAIN), history=3)

    predicted = multi.distances(_gap(2, 2, 3, _FOUR_TIMEPOINTS, _FOUR_POSITIONS))

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 4.0]]  # averaged +4 and +4 → +4, chain arithmetic holds


def test_history_beyond_the_available_frames_uses_what_exists():
    """Asking for more history than the video has does not error — it averages the gaps that exist."""
    multi = MotionPrediction(affinity=_Affinity(_CONFIDENT_CHAIN), history=9)

    predicted = multi.distances(_gap(1, 1, 2, _FOUR_TIMEPOINTS, _FOUR_POSITIONS))

    assert predicted.tolist() == [[4.0 - VELOCITY_DAMPING * 4.0]]  # only one prior gap exists at t=1

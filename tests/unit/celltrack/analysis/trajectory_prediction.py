"""Unit tests for the oracle trajectory-prediction criterion — pinned on constructed motion.

The module's claim is arithmetic about geometry: advance a source by its velocity, then compare distances from
the predicted point. The fixtures build the two cases the measurement must tell apart — a persistently-moving
cell whose velocity DOES point at the true successor, and a stationary one whose velocity cannot help.
"""

import numpy as np
import pytest

from celltrack.analysis.localisation import CandidatePairs
from celltrack.analysis.trajectory_prediction import PredictedStep, TrajectoryDiagnosis
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, NodeMatching

_UNIT = Spacing(z=1.0, y=1.0, x=1.0)


def _diagnosis(
    prediction: TrackGraph,
    truth: TrackGraph,
    gt_rows: list[int],
) -> TrajectoryDiagnosis:
    """A diagnosis with the drift fields empty — the raw-velocity path uses none."""
    return TrajectoryDiagnosis(
        prediction=prediction,
        truth=truth,
        matching=NodeMatching(gt_rows=np.asarray(gt_rows, dtype=np.int64)),
        spacing=_UNIT,
        fields={},
    )


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph from explicit `(t, z, y, x)` coordinates and edges over consecutive node ids."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_of():
    """A source heading straight at its true successor: prediction lands on it, reversing the raw distance.

    The source moved +4 in x last frame and keeps going: the true successor is at x=8 (far, +8 from source at
    x=4) while a decoy sits at x=3 (near). Raw distance prefers the decoy; advancing the source by its velocity
    to x=8 puts the prediction exactly on the true successor, so the criterion rescues the case.
    """
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [2, 0, 0, 8], [2, 0, 0, 3]], [])  # prev, source, true, decoy
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])  # source's GT incoming step: +4 in x
    diagnosis = _diagnosis(prediction, truth, gt_rows=[0, 1, UNMATCHED, UNMATCHED])
    pairs = CandidatePairs(source=np.array([1]), true_target=np.array([2]), rival=np.array([3]))

    step = PredictedStep.of(diagnosis, pairs, field_by_gap=None)

    assert step.count() == 1
    assert step.raw_gap_um[0] > 0.0  # raw distance: decoy nearer
    assert step.predicted_gap_um[0] < 0.0  # prediction: true nearer
    assert step.rescued() == 1
    assert step.speed_um[0] == pytest.approx(4.0)
    assert step.alignment[0] == pytest.approx(1.0)  # velocity points straight at the true successor


def test_predicted_step_of_cannot_help_a_stationary_source():
    """A source with no prior motion: the prediction is the source itself, so it reproduces raw distance.

    This is the measured mislink case — velocity near zero and uncorrelated with the true step. The predicted
    point cannot move toward a far true successor, so a decoy that was nearer stays nearer.
    """
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 8], [2, 0, 0, 3]], [])
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 1]])  # incoming step is zero — the source did not move
    diagnosis = _diagnosis(prediction, truth, gt_rows=[0, 1, UNMATCHED, UNMATCHED])
    pairs = CandidatePairs(source=np.array([1]), true_target=np.array([2]), rival=np.array([3]))

    step = PredictedStep.of(diagnosis, pairs, field_by_gap=None)

    assert step.raw_gap_um[0] == pytest.approx(step.predicted_gap_um[0])  # no velocity → prediction == raw
    assert step.rescued() == 0
    assert step.speed_um[0] == pytest.approx(0.0)


def test_predicted_step_of_skips_a_source_without_a_predecessor():
    """A source with no ground-truth incoming edge carries no velocity and is left out, not predicted from zero."""
    prediction = _graph([[1, 0, 0, 4], [2, 0, 0, 8], [2, 0, 0, 3]], [])
    truth = _graph([[1, 0, 0, 4]], [])  # the source node has no predecessor
    diagnosis = _diagnosis(prediction, truth, gt_rows=[0, UNMATCHED, UNMATCHED])
    pairs = CandidatePairs(source=np.array([0]), true_target=np.array([1]), rival=np.array([2]))

    assert PredictedStep.of(diagnosis, pairs, field_by_gap=None).count() == 0


def test_rescued():
    """Rescued counts wrong→right flips, broken counts right→wrong — the two ways prediction changes an order."""
    step = PredictedStep(
        predicted_gap_um=np.array([-1.0, 1.0, -1.0]),
        raw_gap_um=np.array([1.0, -1.0, -1.0]),
        speed_um=np.array([1.0, 1.0, 1.0]),
        alignment=np.array([1.0, -1.0, 1.0]),
    )

    assert step.rescued() == 1  # row 0: raw wrong (+), predicted right (−)


def test_broken():
    """Broken counts a CORRECT ordering that prediction reverses — right (−) turned wrong (+)."""
    step = PredictedStep(
        predicted_gap_um=np.array([1.0, -1.0]),
        raw_gap_um=np.array([-1.0, -1.0]),
        speed_um=np.array([1.0, 1.0]),
        alignment=np.array([-1.0, 1.0]),
    )

    assert step.broken() == 1  # row 0: raw right (−), predicted wrong (+)


def test_prefers_true():
    """The share where the predicted point sits nearer the true successor — negative gap is a true-nearer case."""
    step = PredictedStep(
        predicted_gap_um=np.array([-1.0, -1.0, 2.0]),
        raw_gap_um=np.array([0.0, 0.0, 0.0]),
        speed_um=np.array([1.0, 1.0, 1.0]),
        alignment=np.array([1.0, 1.0, 1.0]),
    )

    assert step.prefers_true() == pytest.approx(2 / 3)


def test_predicted_step_of_alignment_reads_a_reversal():
    """A velocity pointing AWAY from the true successor reads a negative alignment — the mislink signature.

    The source moved +4 in x but its annotated successor is at x=0 (behind it): the cell reversed relative to
    its prior motion, so the alignment is −1. This is the −0.37 the dense mislinks actually show, in miniature.
    """
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [2, 0, 0, 0], [2, 0, 0, 5]], [])
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    diagnosis = _diagnosis(prediction, truth, gt_rows=[0, 1, UNMATCHED, UNMATCHED])
    pairs = CandidatePairs(source=np.array([1]), true_target=np.array([2]), rival=np.array([3]))

    step = PredictedStep.of(diagnosis, pairs, field_by_gap=None)

    assert step.alignment[0] == pytest.approx(-1.0)  # moved +x, successor is at −x → reversal


def test_count():
    """Count is the number of triples that carried a ground-truth velocity."""
    step = PredictedStep(
        predicted_gap_um=np.array([1.0, 2.0]),
        raw_gap_um=np.array([1.0, 2.0]),
        speed_um=np.array([1.0, 1.0]),
        alignment=np.array([0.0, 0.0]),
    )
    assert step.count() == 2

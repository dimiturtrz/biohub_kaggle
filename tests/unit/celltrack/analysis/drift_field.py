"""Unit tests for the displacement-voting drift field — pinned against fields whose answer is known.

The estimator's whole claim is that it recovers shared motion from mostly-WRONG pairs, so the fixtures build
exactly that situation: a known displacement applied to a known set of points, with the wrong pairings present
and outnumbering the right ones. A test that fed it only true pairs would prove nothing about the method.
"""

import numpy as np
import pytest

from celltrack.analysis.drift_field import (
    AnnotatedResidual,
    CandidateReordering,
    DisplacementVotes,
    DriftField,
    FieldSettings,
    SyntheticRecovery,
)
from celltrack.analysis.localisation import CandidatePairs
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_UNIT = Spacing(z=1.0, y=1.0, x=1.0)
_SIGMA = 8.0


def _drifting_cloud(shift: tuple[float, float, float], count: int = 40, seed: int = 0):
    """Two frames of a cloud translated by `shift`, as `(positions_um, timepoints)`.

    The cells are spread over tens of micrometres so many pairs fall inside the gate, which means most of the
    candidate pairs are WRONG — the condition the estimator is built for.
    """
    generator = np.random.default_rng(seed)
    first = generator.uniform(0.0, 30.0, size=(count, 3))
    positions = np.concatenate([first, first + np.asarray(shift)])
    timepoints = np.concatenate([np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)])
    return positions, timepoints


def test_displacement_votes_of():
    """Every in-gate cross-frame pair votes, so the votes far outnumber the cells — wrong pairs included."""
    positions, timepoints = _drifting_cloud((1.0, 0.0, 0.0), count=20)

    votes = DisplacementVotes.of(positions, timepoints, timepoint=0, gate_um=10.0)

    assert votes.count() > 20  # more votes than true pairs, i.e. the wrong pairings are present
    assert votes.origins_um.shape[1] == 3
    assert np.all(np.linalg.norm(votes.displacements_um, axis=1) <= 10.0)


def test_displacement_votes_count():
    """An empty frame pair casts no votes rather than raising."""
    positions, timepoints = _drifting_cloud((1.0, 0.0, 0.0))

    assert DisplacementVotes.of(positions, timepoints, timepoint=5).count() == 0


def test_drift_field_of():
    """A uniform translation is recovered from mostly-wrong pairs — the estimator's core claim.

    The true pairs all agree on one displacement while the wrong ones scatter, so the vote histogram peaks at
    the true shift. Recovery to well inside one vote bin is what distinguishes a mode-finder from an average,
    which would be dragged toward zero by the majority of wrong pairs.
    """
    shift = (2.0, 1.0, 0.0)
    positions, timepoints = _drifting_cloud(shift, count=60)
    votes = DisplacementVotes.of(positions, timepoints, timepoint=0)
    anchors = DriftField.anchor_grid(positions, spacing_um=15.0)

    field = DriftField.of(votes, FieldSettings(anchors, _SIGMA))

    assert np.allclose(field.global_drift(), shift, atol=0.5)
    assert field.sharpness.max() > 0.0


def test_anchor_grid():
    """The grid spans the detections at the requested spacing."""
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    anchors = DriftField.anchor_grid(positions, spacing_um=5.0)

    assert anchors.shape[1] == 3
    assert anchors.min() == pytest.approx(0.0)
    assert anchors.max() >= 10.0


def test_at():
    """The field reads back continuously between anchors, and an empty query is empty rather than an error."""
    shift = (2.0, 1.0, 0.0)
    positions, timepoints = _drifting_cloud(shift, count=60)
    field = DriftField.of(DisplacementVotes.of(positions, timepoints, 0), FieldSettings.over(positions, _SIGMA, 15.0))

    sampled = field.at(np.array([[5.0, 5.0, 5.0], [20.0, 20.0, 20.0]]))

    assert sampled.shape == (2, 3)
    assert np.allclose(sampled, shift, atol=0.6)
    assert field.at(np.zeros((0, 3))).shape == (0, 3)


def test_global_drift():
    """The uniform component is the field's own volume average — no separate estimator behind it."""
    field = DriftField(
        anchors_um=np.zeros((2, 3)),
        vectors_um=np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        sharpness=np.ones(2),
        sigma_um=_SIGMA,
    )

    assert field.global_drift() == pytest.approx([2.0, 0.0, 0.0])


def test_synthetic_recovery_of():
    """An imposed shear is recovered differentially, so whatever real drift the data has cancels out."""
    positions, timepoints = _drifting_cloud((1.0, 0.0, 0.0), count=80)
    anchors = DriftField.anchor_grid(positions, spacing_um=15.0)

    recovery = SyntheticRecovery.of(positions, timepoints, 0, FieldSettings(anchors, _SIGMA), amplitude_um=3.0)

    assert recovery.imposed_um.shape == anchors.shape
    assert np.abs(recovery.imposed_um).max() > 0.0  # the shear is not a constant the difference would cancel


def test_error_um():
    """Recovery error is per anchor, so a field that fails only in one region is visible rather than averaged."""
    recovery = SyntheticRecovery(
        imposed_um=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        recovered_um=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    assert recovery.error_um() == pytest.approx([0.0, 2.0])


def _annotated(step: tuple[float, float, float]) -> TrackGraph:
    """One annotated cell taking a single step of `step` voxels between frames 0 and 1."""
    return TrackGraph(
        node_ids=np.arange(2, dtype=np.int64),
        coordinates=np.asarray([[0, 5, 5, 5], [1, 5 + step[0], 5 + step[1], 5 + step[2]]], dtype=np.int64),
        edges=np.asarray([[0, 1]], dtype=np.int64),
    )


def _uniform_field(vector: tuple[float, float, float]) -> DriftField:
    """A field that returns `vector` everywhere."""
    return DriftField(
        anchors_um=np.zeros((1, 3)),
        vectors_um=np.asarray([vector], dtype=np.float64),
        sharpness=np.ones(1),
        sigma_um=_SIGMA,
    )


def test_annotated_residual_of():
    """Annotated steps are scored before and after the field, on truth the field never saw."""
    residual = AnnotatedResidual.of(_annotated((2, 0, 0)), _UNIT, {0: _uniform_field((2.0, 0.0, 0.0))})

    assert residual.raw_um == pytest.approx([2.0])
    assert residual.residual_um == pytest.approx([0.0])


def test_explained():
    """A field matching the motion explains it; one pointing the wrong way reads NEGATIVE, not zero.

    The negative branch is the over-correction guard: a field flexible enough to subtract motion the cell did
    not share makes the residual LONGER than the raw step, and that has to be visible rather than clipped.
    """
    matched = AnnotatedResidual.of(_annotated((2, 0, 0)), _UNIT, {0: _uniform_field((2.0, 0.0, 0.0))})
    opposed = AnnotatedResidual.of(_annotated((2, 0, 0)), _UNIT, {0: _uniform_field((-2.0, 0.0, 0.0))})

    assert matched.explained() == pytest.approx(1.0)
    assert opposed.explained() < 0.0


def _triple(true_target: tuple[int, int, int], rival: tuple[int, int, int]) -> tuple[TrackGraph, CandidatePairs]:
    """A source at the origin with a true successor and a rival in the next frame."""
    prediction = TrackGraph(
        node_ids=np.arange(3, dtype=np.int64),
        coordinates=np.asarray([[0, 0, 0, 0], [1, *true_target], [1, *rival]], dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    pairs = CandidatePairs(
        source=np.array([0], dtype=np.int64),
        true_target=np.array([1], dtype=np.int64),
        rival=np.array([2], dtype=np.int64),
    )
    return prediction, pairs


def test_candidate_reordering_of():
    """The gap is `d(true) - d(rival)`, positive when the rival is the nearer — the mislink's own signature."""
    prediction, pairs = _triple(true_target=(0, 0, 6), rival=(0, 0, 2))

    reordering = CandidateReordering.of(prediction, pairs, _UNIT, {0: _uniform_field((0.0, 0.0, 0.0))})

    assert reordering.count() == 1
    assert reordering.raw_gap_um == pytest.approx([4.0])


def test_rescued():
    """A field pushing the source toward the true successor flips the preference — what a rescue IS."""
    prediction, pairs = _triple(true_target=(0, 0, 6), rival=(0, 0, -2))

    reordering = CandidateReordering.of(prediction, pairs, _UNIT, {0: _uniform_field((0.0, 0.0, 4.0))})

    assert reordering.raw_gap_um[0] > 0.0  # the rival started nearer
    assert reordering.rescued() == 1


def test_broken():
    """The same correction can reorder a CORRECT edge the wrong way, and that has to be counted too."""
    prediction, pairs = _triple(true_target=(0, 0, 2), rival=(0, 0, -6))

    reordering = CandidateReordering.of(prediction, pairs, _UNIT, {0: _uniform_field((0.0, 0.0, -4.0))})

    assert reordering.raw_gap_um[0] < 0.0  # the true successor started nearer
    assert reordering.broken() == 1


def test_candidate_reordering_count():
    """A gap with no estimated field contributes no verdict rather than a silent zero correction."""
    prediction, pairs = _triple(true_target=(0, 0, 6), rival=(0, 0, 2))

    assert CandidateReordering.of(prediction, pairs, _UNIT, {}).count() == 0


def test_over():
    """Settings build their own anchor grid from the detections they will be estimated over."""
    positions, _ = _drifting_cloud((1.0, 0.0, 0.0), count=20)

    settings = FieldSettings.over(positions, sigma_um=_SIGMA, spacing_um=15.0)

    assert settings.sigma_um == _SIGMA
    assert settings.device == "cpu"  # the measured-faster default at this problem size
    assert len(settings.anchors_um) > 1

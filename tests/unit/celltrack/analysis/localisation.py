import numpy as np
import pytest

from celltrack.analysis.localisation import (
    AnnotatedStepByFate,
    CandidatePairs,
    CrowdedNodes,
    DeltaCollisions,
    EndpointSides,
    LocalisationOffsets,
    MatchingInverse,
    MislinkEndpoints,
    OffsetSummary,
    RivalDistanceGap,
)
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, NodeMatching

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph over consecutively-numbered node ids, `t, z, y, x` per node."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_matching_inverse_of() -> None:
    matching = NodeMatching(gt_rows=np.array([2, UNMATCHED, 0], dtype=np.int64))
    assert MatchingInverse.of(matching, truth_count=3).rows.tolist() == [2, UNMATCHED, 0]


def test_localisation_offsets_of() -> None:
    prediction = _graph([[0, 1, 4, 0]], [])
    truth = _graph([[0, 0, 0, 0]], [])
    matching = NodeMatching(gt_rows=np.array([0], dtype=np.int64))

    offsets = LocalisationOffsets.of(prediction, truth, matching, _SPACING)

    assert np.allclose(offsets.offsets[0], [1.625, 1.625, 0.0])


def test_localisation_offsets_of_skips_unmatched_detections() -> None:
    prediction = _graph([[0, 0, 0, 0], [0, 0, 0, 8]], [])
    truth = _graph([[0, 0, 0, 0]], [])
    matching = NodeMatching(gt_rows=np.array([0, UNMATCHED], dtype=np.int64))

    offsets = LocalisationOffsets.of(prediction, truth, matching, _SPACING)

    assert offsets.prediction_rows.tolist() == [0]


def test_select() -> None:
    offsets = LocalisationOffsets(
        prediction_rows=np.array([0, 3, 7], dtype=np.int64),
        offsets=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    selected = offsets.select(np.array([3, 7, 99], dtype=np.int64))

    assert selected.prediction_rows.tolist() == [3, 7]


def test_summary() -> None:
    offsets = LocalisationOffsets(
        prediction_rows=np.array([0, 1], dtype=np.int64),
        offsets=np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]]),
    )

    assert offsets.summary().count == 2
    assert offsets.summary().median_um == 2.0


def test_offset_summary_of() -> None:
    summary = OffsetSummary.of(np.array([[0.0, -1.0, 0.0], [0.0, -3.0, 0.0]]))

    assert summary.count == 2
    assert summary.median_um == 2.0
    assert summary.iqr_um == (1.5, 2.5)
    assert summary.axis_median_abs_um[1] == 2.0
    assert summary.axis_median_signed_um[1] == -2.0


def test_offset_summary_of_an_empty_cut_is_nan_not_zero() -> None:
    summary = OffsetSummary.of(np.empty((0, 3)))

    assert summary.count == 0
    assert np.isnan(summary.median_um)
    assert np.isnan(summary.axis_median_abs_um[0])


def test_mislink_endpoints_of() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 40], [0, 0, 0, 40]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 40], [0, 0, 0, 40]], [[0, 2]])
    matching = NodeMatching(gt_rows=np.array([0, 1, 2, 3], dtype=np.int64))

    assert MislinkEndpoints.of(prediction, truth, matching).rows.tolist() == [0, 1]


def test_within_read_out_grid() -> None:
    offsets = LocalisationOffsets(
        prediction_rows=np.array([0, 1], dtype=np.int64),
        # (z, y, x) in um: the first is two full-resolution voxels out in y — the half-block a 4x read-out
        # can round by; the second is three, which rounding alone cannot reach.
        offsets=np.array([[0.0, 2 * 0.40625, 0.0], [0.0, 3 * 0.40625, 0.0]]),
    )

    assert offsets.within_read_out_grid(_SPACING, (1, 4, 4)) == (1.0, 0.5, 1.0)


def test_within_read_out_grid_admits_no_z_error_when_z_is_not_downsampled() -> None:
    offsets = LocalisationOffsets(
        prediction_rows=np.array([0, 1], dtype=np.int64),
        offsets=np.array([[0.0, 0.0, 0.0], [1.625, 0.0, 0.0]]),
    )

    assert offsets.within_read_out_grid(_SPACING, (1, 4, 4))[0] == 0.5


def test_crowded_nodes_of() -> None:
    truth = _graph([[0, 0, 0, 0], [0, 0, 0, 100]], [])
    # Row 0's annotated cell has a second detection 4 um away; row 1's has only its own.
    prediction = _graph([[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 0, 10]], [])
    matching = NodeMatching(gt_rows=np.array([0, 1, UNMATCHED], dtype=np.int64))

    crowded = CrowdedNodes.of(prediction, truth, matching, _SPACING, radius_um=7.0)

    assert crowded.rows.tolist() == [0]


def test_crowding_is_counted_within_a_frame_not_across_frames() -> None:
    truth = _graph([[0, 0, 0, 0]], [])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [])
    matching = NodeMatching(gt_rows=np.array([0, UNMATCHED], dtype=np.int64))

    assert CrowdedNodes.of(prediction, truth, matching, _SPACING, radius_um=7.0).rows.tolist() == []


def test_mislinked() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 40]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 40]], [[0, 2]])
    matching = NodeMatching(gt_rows=np.array([0, 1, 2], dtype=np.int64))

    pairs = CandidatePairs.mislinked(prediction, truth, matching)

    assert (pairs.source.tolist(), pairs.true_target.tolist(), pairs.rival.tolist()) == ([0], [1], [2])


def test_correct() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 12], [1, 0, 0, 8]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 1, UNMATCHED, UNMATCHED], dtype=np.int64))

    pairs = CandidatePairs.correct(prediction, truth, matching, _SPACING, gate_um=10.0)

    assert pairs.rival.tolist() == [3]


def test_correct_drops_a_source_with_no_rival_inside_the_gate() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 200]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 1, UNMATCHED], dtype=np.int64))

    assert CandidatePairs.correct(prediction, truth, matching, _SPACING, gate_um=10.0).source.tolist() == []


def _pairs(true_target: int, rival: int) -> CandidatePairs:
    """One triple: prediction row 0 as the source, against the named true target and rival."""
    return CandidatePairs(
        source=np.array([0], dtype=np.int64),
        true_target=np.array([true_target], dtype=np.int64),
        rival=np.array([rival], dtype=np.int64),
    )


# Sources at x=0; on the detection lattice (1,4,4) the y/x columns divide by four, so rows 1 and 2 share a
# lattice cell and row 3 sits one lattice voxel further out in x.
_LATTICE_GRAPH = _graph([[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 4], [1, 0, 0, 8]], [])


def test_delta_collisions_of() -> None:
    collisions = DeltaCollisions.of(_LATTICE_GRAPH, _pairs(true_target=1, rival=3), (1, 4, 4))

    assert collisions.differences.tolist() == [[0, 0, -1]]


def test_lattice_separations() -> None:
    collisions = DeltaCollisions.of(_LATTICE_GRAPH, _pairs(true_target=1, rival=3), (1, 4, 4))

    assert collisions.lattice_separations().tolist() == [1]


def test_identical() -> None:
    collisions = DeltaCollisions.of(_LATTICE_GRAPH, _pairs(true_target=1, rival=2), (1, 4, 4))

    assert collisions.identical() == 1


def test_single_voxel() -> None:
    collisions = DeltaCollisions.of(_LATTICE_GRAPH, _pairs(true_target=1, rival=3), (1, 4, 4))

    assert collisions.single_voxel() == 1


def test_rival_distance_gap_of() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 20], [1, 0, 0, 8]], [])

    gaps = RivalDistanceGap.of(prediction, _pairs(true_target=1, rival=2), _SPACING)

    assert gaps.gaps_um[0] == (20 - 8) * 0.40625


def test_rival_nearer() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 20], [1, 0, 0, 8]], [])

    assert RivalDistanceGap.of(prediction, _pairs(true_target=1, rival=2), _SPACING).rival_nearer() == 1.0
    assert RivalDistanceGap.of(prediction, _pairs(true_target=2, rival=1), _SPACING).rival_nearer() == 0.0


def test_invertible() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 20], [1, 0, 0, 8]], [])

    gaps = RivalDistanceGap.of(prediction, _pairs(true_target=1, rival=2), _SPACING)

    assert gaps.invertible(noise_um=1.0) == 0.0
    assert gaps.invertible(noise_um=10.0) == 1.0


def test_mislink_endpoints_of_a_reproduced_edge_is_empty() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 1], dtype=np.int64))

    assert MislinkEndpoints.of(prediction, truth, matching).rows.tolist() == []


def test_endpoint_sides_of() -> None:
    """The triple's offsets are reported per side, and an unannotated rival comes back as an EMPTY summary.

    The empty rival column is the point rather than an omission: every real mislink's chosen partner is an
    unannotated detection, so there is no annotated centre to subtract. A non-empty rival column would mean the
    wrong partner was an annotated cell after all, which contradicts the standing measurement.
    """
    offsets = LocalisationOffsets(
        prediction_rows=np.array([0, 1], dtype=np.int64),
        offsets=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]),
    )

    sides = EndpointSides.of(offsets, _pairs(true_target=1, rival=2))

    assert (sides.source.count, sides.source.median_um) == (1, 1.0)
    assert (sides.true_target.count, sides.true_target.median_um) == (1, 3.0)
    assert sides.rival.count == 0
    assert np.isnan(sides.rival.median_um)


def test_annotated_step_by_fate_of() -> None:
    """The step between ANNOTATED centres, split by fate — the quantity no detection touches.

    This is the arbiter for whether a mislinked edge's cell really travelled far, or whether the apparent
    distance was manufactured by two mislocalised endpoints. Measuring it off the annotation is what makes it
    independent of the read-out, so the fixture puts the annotated source and target 40 voxels apart while the
    tracker links elsewhere.
    """
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 40], [1, 0, 0, 4], [0, 0, 80, 0], [1, 0, 80, 4]], [[0, 1], [3, 4]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 40], [1, 0, 0, 4], [0, 0, 80, 0], [1, 0, 80, 4]], [[0, 2], [3, 4]])
    matching = NodeMatching(gt_rows=np.array([0, 1, 2, 3, 4], dtype=np.int64))

    steps = AnnotatedStepByFate.of(truth, prediction, matching, _SPACING)

    assert steps.mislinked_um == pytest.approx([40 * 0.40625])
    assert steps.correct_um == pytest.approx([4 * 0.40625])

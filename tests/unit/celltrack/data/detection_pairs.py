"""Unit tests for the detected-neighbourhood corpus — the negatives are the reason it exists."""

import numpy as np

from celltrack.data.detection_pairs import DetectionPairs
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)
# Wide enough that these tiny fixtures are ungated unless a test is ABOUT the gate.
_GATE_UM = 100.0


class _Frames:
    """A frame source that is never read — these tests are about the matrix, not the volumes."""

    def frame(self, timepoint: int, downsample: tuple[int, int, int]):
        raise AssertionError("the corpus must not read frames while it is being built")


def _graph(rows: list[tuple[int, int, int, int]], edges: list[tuple[int, int]]) -> TrackGraph:
    """A graph from (id, t, y, x) rows at z=0, with edges given as node-id pairs."""
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    coordinates = np.array([[row[1], 0, row[2], row[3]] for row in rows], dtype=np.int64)
    return TrackGraph(node_ids=ids, coordinates=coordinates, edges=np.array(edges, dtype=np.int64).reshape(-1, 2))


def test_of():
    """A decidable source is supervised against EVERY detection of the next frame, rivals included.

    The unannotated detection at (0, 30) is the whole point: on an annotated-node corpus it would not appear
    in the matrix at all, and here it is an explicit zero — a negative the loss can learn to reject.
    """
    detections = _graph([(1, 0, 0, 0), (2, 1, 0, 2), (3, 1, 0, 30)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 2)], [(10, 11)])

    pairs = DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, _GATE_UM)

    assert len(pairs) == 1
    assert pairs[0].timepoint == 0
    assert pairs[0].source_centres.shape == (1, 3)  # the one decidable source
    assert pairs[0].target_centres.shape == (2, 3)  # BOTH detections are candidates, annotated or not
    assert pairs[0].edge_matrix.tolist() == [[1.0, 0.0]]  # the rival is a supervised negative


def test_of_drops_a_source_whose_successor_was_not_detected():
    """If the true successor has no detection, the row is UNDECIDABLE — no column of it is the answer.

    Supervising it would teach the head that every candidate is wrong when one of them may well be the cell,
    merely detected off-centre; that is the case this corpus must refuse rather than guess at.
    """
    detections = _graph([(1, 0, 0, 0), (2, 1, 0, 30)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 2)], [(10, 11)])

    assert DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, _GATE_UM) == []


def test_of_drops_an_unmatched_source():
    """A detection matching no annotated cell has no known successor, so it cannot supervise a row."""
    detections = _graph([(1, 0, 0, 50), (2, 1, 0, 2)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 2)], [(10, 11)])

    assert DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, _GATE_UM) == []


def test_of_drops_a_dividing_parent():
    """A fork has two successors: this corpus prices one-to-one association, so the parent is left out.

    Picking a winner would silently supervise half a division as a negative. Divisions are recovered by a
    different stage entirely, and this corpus must not quietly take a position on them.
    """
    detections = _graph([(1, 0, 0, 0), (2, 1, 0, 2), (3, 1, 0, 4)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 2), (12, 1, 0, 4)], [(10, 11), (10, 12)])

    assert DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, _GATE_UM) == []


def test_of_carries_the_previous_frame_as_history():
    """Velocity reads the DETECTED previous frame, the same population it will read at inference."""
    detections = _graph([(1, 0, 0, 0), (2, 1, 0, 1), (3, 2, 0, 2)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 1), (12, 2, 0, 2)], [(10, 11), (11, 12)])

    pairs = DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, _GATE_UM)

    assert [pair.timepoint for pair in pairs] == [0, 1]
    assert pairs[0].previous_centres is None  # the first frame has no history
    assert pairs[1].previous_centres is not None
    assert pairs[1].previous_centres.tolist() == [[0, 0, 0]]


def test_of_gates_the_candidates_but_never_the_true_successor():
    """Out-of-gate rivals are dropped; a true step LONGER than the gate is kept, because that is the hard case.

    Dropping it would delete exactly the positives the linker gets wrong while keeping their near rivals — the
    corpus would then teach that the near cell is always right, which is the failure it exists to correct.
    """
    detections = _graph([(1, 0, 0, 0), (2, 1, 0, 9), (3, 1, 0, 40)], [])
    truth = _graph([(10, 0, 0, 0), (11, 1, 0, 9)], [(10, 11)])

    pairs = DetectionPairs.of(detections, truth, _Frames(), _ISOTROPIC, 5.0)

    assert pairs[0].target_centres.shape == (1, 3)  # the 40um rival is outside the gate and dropped
    assert pairs[0].edge_matrix.tolist() == [[1.0]]  # the 9um TRUE successor survives its own gate

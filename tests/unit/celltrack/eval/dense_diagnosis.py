import math

import numpy as np

from celltrack.eval.dense_diagnosis import DenseFateDiagnosis, Fate, MislinkSignal
from core.data.tracks import TrackGraph
from core.metrics.matching import NodeMatching


class _Affinity:
    """A fixed per-gap probability matrix indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph whose node ids are its row indices, so edge ids and rows coincide for the fixture."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_dense_fate_diagnosis_of():
    """Each ground-truth edge lands in the fate its endpoints' matched detections earned — one edge per class.

    Truth: four cells at t0 (rows 0-3) each continuing to a t1 node (rows 4-7), plus a decoy t1 node (row 8);
    four annotated edges 0->4, 1->5, 2->6, 3->7. Under the identity matching (predicted row i stands in for
    truth row i) the predicted links 0->4, 2->8, 3->6 make one edge of each detected fate:
      - 0->4 reproduced                                    -> CORRECT
      - 1->5: source 1 links to nothing                    -> SKIP
      - 2->6: source 2 links to decoy 8, and 6 was claimed by 3->6  -> MISLINK_CONFLICT
      - 3->7: source 3 links to 6, and target 7 sits unclaimed       -> MISLINK_FREE
    """
    truth = _graph(
        [
            [0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 9, 0], [0, 0, 9, 9], [1, 0, 0, 0],
            [1, 0, 0, 9], [1, 0, 9, 0], [1, 0, 9, 9], [1, 0, 5, 5],
        ],
        [[0, 4], [1, 5], [2, 6], [3, 7]],
    )
    prediction = _graph(truth.coordinates.tolist(), [[0, 4], [2, 8], [3, 6]])
    matching = NodeMatching(gt_rows=np.arange(9, dtype=np.int64))  # prediction row i stands in for truth row i

    counts = DenseFateDiagnosis.of(prediction, truth, matching).counts()

    assert counts[Fate.CORRECT] == 1
    assert counts[Fate.SKIP] == 1
    assert counts[Fate.MISLINK_CONFLICT] == 1
    assert counts[Fate.MISLINK_FREE] == 1
    assert counts[Fate.ENDPOINT_MISSING] == 0


def test_of_endpoint_missing_when_a_detection_is_absent():
    """An annotated edge whose endpoint never matched a detection is detection-limited, not a link error."""
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 1]])
    prediction = _graph([[0, 0, 0, 0]], [])  # only the source is detected
    matching = NodeMatching(gt_rows=np.array([0], dtype=np.int64))  # predicted row 0 -> truth row 0; truth row 1 absent

    counts = DenseFateDiagnosis.of(prediction, truth, matching).counts()
    assert counts[Fate.ENDPOINT_MISSING] == 1


def test_counts():
    """`counts` reports every fate, defaulting absent ones to zero so the decomposition is total over the family."""
    diagnosis = DenseFateDiagnosis(fates=np.array([Fate.CORRECT, Fate.CORRECT, Fate.SKIP], dtype=np.int64))
    counts = diagnosis.counts()
    assert counts == {
        Fate.CORRECT: 2,
        Fate.SKIP: 1,
        Fate.MISLINK_CONFLICT: 0,
        Fate.MISLINK_FREE: 0,
        Fate.ENDPOINT_MISSING: 0,
    }


def test_mislink_signal_of():
    """For a mislinked edge, `MislinkSignal.of` reads the affinity's true-successor and chosen-neighbour P.

    Truth continues cell 0 (t0) to cell 2 (t1); the tracker instead links 0→1 (the wrong neighbour). The
    affinity at gap 0 scores the chosen neighbour 1 at 0.7 and the true successor 2 at 0.2 — an inverted
    ranking, which is what a hard-negative retrain (not the distance/cost lever) would target.
    """
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 0, 9]], [[0, 2]])
    prediction = _graph(truth.coordinates.tolist(), [[0, 1]])  # chose neighbour 1, not true successor 2
    matching = NodeMatching(gt_rows=np.arange(3, dtype=np.int64))
    affinity = _Affinity({0: np.array([[0.7, 0.2]], dtype=np.float64)})  # source 0 -> [target 1, target 2]

    signal = MislinkSignal.of(prediction, truth, matching, affinity)

    assert signal.p_true.tolist() == [0.2]
    assert signal.p_chosen.tolist() == [0.7]


def test_inverted_fraction():
    """`inverted_fraction` is the share of mislinks whose chosen neighbour outscored the true successor."""
    signal = MislinkSignal(p_true=np.array([0.1, 0.8, 0.3]), p_chosen=np.array([0.6, 0.2, 0.9]))
    assert signal.inverted_fraction() == 2 / 3  # edges 0 and 2 are inverted; edge 1 is not


def test_means():
    """`means` averages the true- and chosen-target probabilities, and is NaN when there are no mislinks."""
    signal = MislinkSignal(p_true=np.array([0.2, 0.4]), p_chosen=np.array([0.6, 0.8]))
    assert signal.means() == (0.30000000000000004, 0.7)
    empty = MislinkSignal(p_true=np.array([]), p_chosen=np.array([]))
    assert all(math.isnan(value) for value in empty.means())  # both NaN on empty input

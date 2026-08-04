import numpy as np

from celltrack.dense_diagnosis import DenseFateDiagnosis, Fate
from core.data.tracks import TrackGraph
from core.metrics.matching import NodeMatching


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph whose node ids are its row indices, so edge ids and rows coincide for the fixture."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_of():
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
        [[0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 9, 0], [0, 0, 9, 9], [1, 0, 0, 0], [1, 0, 0, 9], [1, 0, 9, 0], [1, 0, 9, 9], [1, 0, 5, 5]],
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
    assert counts == {Fate.CORRECT: 2, Fate.SKIP: 1, Fate.MISLINK_CONFLICT: 0, Fate.MISLINK_FREE: 0, Fate.ENDPOINT_MISSING: 0}

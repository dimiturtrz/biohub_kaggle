import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack.eval.dense_diagnosis import AffinityIndex, DenseDiagnosis, DenseFateDiagnosis, Fate, MislinkSignal
from celltrack.linkers.linkers import LinkerConfig
from celltrack.tracker import CellTracker, TrackerConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import NodeMatching
from core.paths import DataRoot


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


class _FakeScorer:
    """The tracker's edge scorer, exposing only the `affinities` diagnose reads — a fixed per-gap probability."""

    def __init__(self, affinity: _Affinity) -> None:
        self._affinity = affinity

    def affinities(self, path: Path, prediction: TrackGraph, device: str) -> _Affinity:
        return self._affinity


class _FakeTracker:
    """A tracker returning a canned prediction and affinity, so `diagnose`'s wiring runs without real inference."""

    def __init__(self, prediction: TrackGraph, affinity: _Affinity) -> None:
        self._prediction = prediction
        self.edge_scorer = _FakeScorer(affinity)

    def run(self, name: str, path: Path) -> TrackGraph:
        return self._prediction


def test_diagnose():
    """`diagnose` runs the tracker, matches, and decomposes both the fates and the mislink affinity signal.

    Truth links cell 0 (t0) to cell 2 (t1); the tracker mislinks 0->1 with target 2 left unclaimed (MISLINK_FREE),
    and the affinity scores the chosen neighbour above the true successor — the read-out diagnose returns.
    """
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 0, 9]], [[0, 2]])
    prediction = _graph(truth.coordinates.tolist(), [[0, 1]])
    affinity = _Affinity({0: np.array([[0.7, 0.2]], dtype=np.float64)})  # source 0 -> [target 1, target 2]
    tracker = _FakeTracker(prediction, affinity)

    fate, signal = DenseDiagnosis.diagnose(
        cast(CellTracker, tracker), Path("v.zarr"), truth, Spacing(1.0, 1.0, 1.0), "cpu"
    )

    assert fate.counts()[Fate.MISLINK_FREE] == 1
    assert signal.p_true.tolist() == [0.2]
    assert signal.p_chosen.tolist() == [0.7]


def test_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`configured` mounts the same reference packs at an ARBITRARY config — how a swept arm gets decomposed."""
    mounted: dict[str, object] = {}
    monkeypatch.setattr(
        CellTracker,
        "from_packs",
        classmethod(
            lambda cls, pack1, pack2, responses, device, config=None: mounted.update(pack1=pack1, config=config)
        ),
    )
    arm = TrackerConfig(linker=LinkerConfig(name="flow", evidence_ramp=0.5))

    DenseDiagnosis.configured(DataRoot(tmp_path), "cpu", arm)

    assert cast(Path, mounted["pack1"]).match("*/reference/pilkwang/split_0")
    assert cast(TrackerConfig, mounted["config"]) == arm


def test_shipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`shipped` mounts the reference packs off the data root, at the shipped threshold unless one is given."""
    mounted: dict[str, object] = {}
    monkeypatch.setattr(
        CellTracker,
        "from_packs",
        classmethod(
            lambda cls, pack1, pack2, responses, device, config=None: mounted.update(pack1=pack1, config=config)
        ),
    )

    DenseDiagnosis.shipped(DataRoot(tmp_path), "cpu", threshold=0.5)

    assert cast(Path, mounted["pack1"]).match("*/reference/pilkwang/split_0")
    assert cast(TrackerConfig, mounted["config"]).threshold == 0.5
    DenseDiagnosis.shipped(DataRoot(tmp_path), "cpu")
    assert cast(TrackerConfig, mounted["config"]).threshold == TrackerConfig().threshold


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
            [0, 0, 0, 0],
            [0, 0, 0, 9],
            [0, 0, 9, 0],
            [0, 0, 9, 9],
            [1, 0, 0, 0],
            [1, 0, 0, 9],
            [1, 0, 9, 0],
            [1, 0, 9, 9],
            [1, 0, 5, 5],
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

    signal = MislinkSignal.of(prediction, truth, matching, AffinityIndex.of(prediction, affinity))

    assert signal.p_true.tolist() == [0.2]
    assert signal.p_chosen.tolist() == [0.7]


def test_mislink_signal_of_reads_a_post_processed_graph_through_the_detection_row_space():
    """The affinity is scored over the DETECTIONS; a pruned/bridged graph is looked up by node ID, not by row.

    Detections carry an extra t0 node (id 10) the short-track filter dropped, so the surviving source sits at
    detection row 1 — row 0 of the finished graph. Reading the finished graph's rows into the matrix would take
    the DROPPED node's probabilities (true 0.5 vs chosen 0.4, not inverted); the ID lookup takes detection row
    1, whose true successor scores 0.2 against the chosen neighbour's 0.9. A bridge's invented node (99) sits
    in the finished graph without an affinity slot, and indexing it is what the ID lookup avoids.
    """
    detections = TrackGraph(
        node_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 3], [1, 0, 0, 9]], dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    prediction = TrackGraph(
        node_ids=np.array([11, 12, 13, 99], dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 6], [1, 0, 0, 3], [1, 0, 0, 9], [1, 0, 0, 5]], dtype=np.int64),
        edges=np.array([[11, 12], [11, 99]], dtype=np.int64),  # 11 mislinks to 12 (true target 13) and to 99
    )
    truth = TrackGraph(
        node_ids=np.arange(3, dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 6], [1, 0, 0, 9], [1, 0, 0, 3]], dtype=np.int64),
        edges=np.array([[0, 1]], dtype=np.int64),
    )
    matching = NodeMatching(gt_rows=np.array([0, 2, 1, -1], dtype=np.int64))  # prediction row -> truth row
    affinity = _Affinity({0: np.array([[0.4, 0.5], [0.9, 0.2]], dtype=np.float64)})  # rows: detections 10, 11

    signal = MislinkSignal.of(prediction, truth, matching, AffinityIndex.of(detections, affinity))

    assert signal.p_true.tolist() == [0.2]
    assert signal.p_chosen.tolist() == [0.9]


def test_pooled():
    """`pooled` concatenates every movie's mislinks, so an eval's inversion is read over the whole proxy."""
    first = MislinkSignal(p_true=np.array([0.1]), p_chosen=np.array([0.6]))
    second = MislinkSignal(p_true=np.array([0.8, 0.3]), p_chosen=np.array([0.2, 0.9]))

    pooled = MislinkSignal.pooled([first, second])

    assert pooled.p_true.tolist() == [0.1, 0.8, 0.3]
    assert pooled.inverted_fraction() == 2 / 3
    assert MislinkSignal.pooled([]).p_true.tolist() == []


def test_inverted_fraction():
    """`inverted_fraction` is the share of mislinks whose chosen neighbour outscored the true successor."""
    signal = MislinkSignal(p_true=np.array([0.1, 0.8, 0.3]), p_chosen=np.array([0.6, 0.2, 0.9]))
    assert signal.inverted_fraction() == 2 / 3  # edges 0 and 2 are inverted; edge 1 is not
    # A graph with no mislinks leaves a retrain nothing to target — a defined zero, not NaN.
    assert MislinkSignal(p_true=np.array([]), p_chosen=np.array([])).inverted_fraction() == 0.0


def test_affinity_index_of():
    """`of` slots every node id to its `(timepoint, per-gap column)` — the matrices' own row order, by id."""
    graph = TrackGraph(
        node_ids=np.array([5, 6, 7], dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 0, 9]], dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )

    index = AffinityIndex.of(graph, _Affinity({}))

    assert index.slots == {5: (0, 0), 6: (1, 0), 7: (1, 1)}


def test_pair():
    """`pair` reads a source's row and its two targets' columns by node ID, and reports an unscored gap absent."""
    graph = TrackGraph(
        node_ids=np.array([5, 6, 7], dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 3], [1, 0, 0, 9]], dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    index = AffinityIndex.of(graph, _Affinity({0: np.array([[0.7, 0.2]], dtype=np.float64)}))

    assert index.pair(5, 7, 6) == (0.2, 0.7)
    assert index.pair(6, 7, 7) is None  # gap 1 was never scored
    assert index.pair(5, 5, 6) is None  # a "target" in the source's own frame is no column of this gap
    assert index.pair(99, 6, 7) is None  # a node the affinity never scored (a bridge's invented midpoint)


def test_means():
    """`means` averages the true- and chosen-target probabilities, and is NaN when there are no mislinks."""
    signal = MislinkSignal(p_true=np.array([0.2, 0.4]), p_chosen=np.array([0.6, 0.8]))
    assert signal.means() == (0.30000000000000004, 0.7)
    empty = MislinkSignal(p_true=np.array([]), p_chosen=np.array([]))
    assert all(math.isnan(value) for value in empty.means())  # both NaN on empty input

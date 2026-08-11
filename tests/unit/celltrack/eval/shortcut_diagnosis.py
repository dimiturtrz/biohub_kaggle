import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack.eval.dense_diagnosis import DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.shortcut_diagnosis import (
    CandidatePairs,
    MatchedAnnotation,
    ShortcutDiagnosis,
    ShortcutReport,
)
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, NodeMatching
from core.paths import DataRoot

_UNIT = Spacing(1.0, 1.0, 1.0)
_GATE_UM = 10.0


class _Affinity:
    """A fixed per-gap probability matrix indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


@dataclass(frozen=True)
class _ScoredRun:
    """A tracker pass's detections and affinity — what `run_scored` hands `by_movie`."""

    detections: TrackGraph
    affinity: _Affinity


class _FakeTracker:
    """A tracker returning canned detections and affinity, so the diagnosis's wiring runs without inference."""

    def __init__(self, detections: TrackGraph, affinity: _Affinity) -> None:
        self._scored = _ScoredRun(detections=detections, affinity=affinity)

    def run_scored(self, video_key: str, path: Path) -> _ScoredRun:
        return self._scored


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph whose node ids are its row indices, so edge ids and rows coincide for the fixture."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def _pairs(probability: list[float], distance_um: list[float], is_true: list[bool]) -> CandidatePairs:
    """A candidate population stated directly, so a statistic is checked against hand arithmetic."""
    return CandidatePairs(
        probability=np.array(probability, dtype=np.float64),
        distance_um=np.array(distance_um, dtype=np.float64),
        is_true=np.array(is_true, dtype=bool),
    )


def _uniform(size: int, positives: int) -> CandidatePairs:
    """A population of a chosen size and positive count — for the base-rate guard, whose only input is those two."""
    return _pairs(
        probability=list(np.linspace(0.0, 1.0, size)),
        distance_um=list(np.linspace(0.0, _GATE_UM, size)),
        is_true=[index < positives for index in range(size)],
    )


def _detections() -> TrackGraph:
    """Two t0 detections and three t1 ones, spread along x so the 10 um gate admits a known subset of pairs."""
    return _graph([[0, 0, 0, 0], [0, 0, 0, 20], [1, 0, 0, 2], [1, 0, 0, 8], [1, 0, 0, 21]], [])


def test_matched_annotation_of():
    """`of` indexes the annotated edges and the rows the annotation speaks about, keeping the detection matching."""
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [1, 0, 0, 9]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 1, UNMATCHED], dtype=np.int64))

    annotation = MatchedAnnotation.of(truth, matching)

    assert annotation.codes.tolist() == [0 * 3 + 1]
    assert annotation.has_successor.tolist() == [True, False, False]
    assert annotation.has_parent.tolist() == [False, True, False]
    assert annotation.gt_rows.tolist() == [0, 1, UNMATCHED]


def test_judge():
    """A pair is judged when the annotation speaks at EITHER endpoint, and true only when it joins them.

    Detection rows 0 (-> gt 0, whose successor the annotation names) and 1 (-> gt 2, which the annotation never
    mentions) face t1 rows 2 (-> gt 1, the annotated successor) and 3 (unmatched). 0->2 is judged and true.
    0->3 is judged FALSE though its target matched nothing: the annotation already named row 0's successor, and
    this is not it — that unmatched neighbour is a real distractor, not unknown truth. 1->2 is judged FALSE from
    the column side. Only 1->3, where the annotation is silent at both ends, is dropped.
    """
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 20]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 2, 1, UNMATCHED], dtype=np.int64))

    judged, is_true = MatchedAnnotation.of(truth, matching).judge(
        np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)
    )

    assert judged.tolist() == [[True, True], [True, False]]
    assert is_true.tolist() == [[True, False], [False, False]]


def test_candidate_pairs_of():
    """Only judged, in-gate pairs enter the population, each carrying its learned P, its distance and its truth.

    Detection rows 0 (t0, x=0) and 1 (t0, x=20) face t1 rows 2 (x=2), 3 (x=8) and 4 (x=21). Ground truth
    annotates one edge, gt 0 -> gt 1, standing for detections 0 -> 2. Within the 10 um gate that leaves
    0->2 (true, 2 um) and 0->3 (false, 8 um); 0->4 is out of gate, and 1->4 has an unmatched endpoint.
    """
    detections = _detections()
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 20], [1, 0, 0, 22]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, 2, 1, 3, UNMATCHED], dtype=np.int64))
    affinity = _Affinity({0: np.array([[0.9, 0.3, 0.1], [0.2, 0.4, 0.8]], dtype=np.float64)})

    candidates = CandidatePairs.of(detections, MatchedAnnotation.of(truth, matching), affinity, _UNIT, _GATE_UM)

    assert candidates.probability.tolist() == [0.9, 0.3]
    assert candidates.distance_um.tolist() == [2.0, 8.0]
    assert candidates.is_true.tolist() == [True, False]


def test_candidate_pairs_of_counts_an_unmatched_neighbour_of_an_annotated_source_as_a_negative():
    """An unmatched TARGET of an annotated source is the population's negative; silence at both ends is dropped.

    Detection 0 (x=0) matches a ground-truth node whose successor the annotation names, so its in-gate
    neighbour 3 (x=8, matching nothing) is a distractor the linker must reject — a real false candidate, and
    the kind that keeping both endpoints matched would have thrown away, leaving a population that is nearly
    all true. Detection 1 (x=20) is annotated no successor and its in-gate neighbour 4 (x=21) matched nothing,
    so that pair carries no truth at either end and is dropped rather than invented as a negative.
    """
    detections = _detections()
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 20]], [[0, 1]])
    unmatched = NodeMatching(gt_rows=np.array([0, 2, 1, UNMATCHED, UNMATCHED], dtype=np.int64))
    affinity = _Affinity({0: np.array([[0.9, 0.3, 0.1], [0.2, 0.4, 0.8]], dtype=np.float64)})

    candidates = CandidatePairs.of(detections, MatchedAnnotation.of(truth, unmatched), affinity, _UNIT, _GATE_UM)

    assert candidates.distance_um.tolist() == [2.0, 8.0]  # 1->4 (1 um) is dropped: no truth at either end
    assert candidates.is_true.tolist() == [True, False]


def test_candidate_pairs_of_rejects_an_affinity_block_that_does_not_align_with_the_gap():
    """A matrix whose shape is not the gap's is a different row space — refused, never indexed into."""
    detections = _detections()
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2]], [[0, 1]])
    matching = NodeMatching(gt_rows=np.array([0, UNMATCHED, 1, UNMATCHED, UNMATCHED], dtype=np.int64))
    affinity = _Affinity({0: np.zeros((2, 2), dtype=np.float64)})

    with pytest.raises(ValueError, match="does not align"):
        CandidatePairs.of(detections, MatchedAnnotation.of(truth, matching), affinity, _UNIT, _GATE_UM)


def test_pooled():
    """`pooled` concatenates populations, so the pooled read-out is one ranking over every movie's candidates."""
    first = _pairs([0.9], [1.0], [True])
    second = _pairs([0.2, 0.4], [3.0, 5.0], [False, True])

    pooled = CandidatePairs.pooled([first, second])

    assert pooled.probability.tolist() == [0.9, 0.2, 0.4]
    assert pooled.distance_um.tolist() == [1.0, 3.0, 5.0]
    assert pooled.is_true.tolist() == [True, False, True]
    assert CandidatePairs.pooled([]).size() == 0


def test_save(tmp_path: Path):
    """`save` writes the population's three columns, creating the directory the caller named."""
    path = tmp_path / "nested" / "movie.npz"

    _pairs([0.9, 0.2, 0.4], [1.0, 3.0, 5.0], [True, False, True]).save(path)

    with np.load(path) as stored:
        assert stored["probability"].tolist() == [0.9, 0.2, 0.4]
        assert stored["distance_um"].tolist() == [1.0, 3.0, 5.0]
        assert stored["is_true"].tolist() == [True, False, True]


def test_load(tmp_path: Path):
    """A saved population reads back identical, so re-banding it costs no detector pass."""
    population = _pairs([0.9, 0.2, 0.4], [1.0, 3.0, 5.0], [True, False, True])
    path = tmp_path / "movie.npz"
    population.save(path)

    restored = CandidatePairs.load(path)

    assert restored.probability.tolist() == population.probability.tolist()
    assert restored.distance_um.tolist() == population.distance_um.tolist()
    assert restored.is_true.tolist() == population.is_true.tolist()


def test_size():
    """`size` is the number of judged in-gate candidates the population holds."""
    assert _pairs([0.1, 0.2], [1.0, 2.0], [True, False]).size() == 2
    assert CandidatePairs.pooled([]).size() == 0


def test_positives():
    """`positives` counts the true pairs — the minority class an AUC's precision rests on."""
    assert _pairs([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [True, False, True]).positives() == 2


def test_base_rate():
    """`base_rate` is the true share, and NaN (not zero) on an empty population — the number printed first."""
    assert _pairs([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], [True, False, False, False]).base_rate() == 0.25
    assert math.isnan(CandidatePairs.pooled([]).base_rate())


def test_is_reportable():
    """The guard: a population near all-true or near all-false earns no AUC, a balanced one does.

    0.975 is the ground-truth-graph artefact this tool exists to refuse (446/457 in-gate pairs true); 0.001 is
    the opposite degeneracy, where the AUC moves with which handful of positives landed in the set.
    """
    assert not _uniform(size=40, positives=39).is_reportable()  # 0.975 — the GT-graph degeneracy
    assert not _uniform(size=1000, positives=1).is_reportable()  # 0.001 — too few positives to rank
    assert _uniform(size=20, positives=4).is_reportable()  # 0.2 — a real-detection gate's regime


def test_auc_probability():
    """Rank AUC of P against the labels, hand-computed: positives at ranks 2 and 4 give (6 - 3) / (2*2) = 0.75."""
    assert _pairs([0.1, 0.2, 0.3, 0.4], [4.0, 3.0, 2.0, 1.0], [False, True, False, True]).auc_probability() == 0.75
    assert math.isnan(_pairs([0.1, 0.2], [1.0, 2.0], [True, True]).auc_probability())  # one class only


def test_auc_proximity():
    """The same statistic for `-distance` — here the true pairs are the FAR ones, so proximity ranks at 0.25."""
    assert _pairs([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], [False, True, False, True]).auc_proximity() == 0.25


def test_correlation():
    """Pearson r between P and distance — exactly -1 when P is an affine function of proximity, NaN when flat."""
    assert _pairs([0.4, 0.3, 0.2, 0.1], [1.0, 2.0, 3.0, 4.0], [True, False, True, False]).correlation() == -1.0
    assert math.isnan(_pairs([0.5, 0.5], [1.0, 2.0], [True, False]).correlation())  # P constant: r undefined


def test_bands():
    """`bands` splits the population into equal-count, ascending distance bands — distance held near-constant."""
    population = _pairs(
        probability=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0],
        distance_um=[0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0],
        is_true=[True, False] * 5,
    )

    bands = population.bands()

    assert [band.size() for band in bands] == [2, 2, 2, 2, 2]
    assert [band.span() for band in bands] == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
    assert bands[0].probability.tolist() == [0.9, 0.8]


def test_span():
    """`span` is the population's distance range — a band's label, NaN on an empty band."""
    assert _pairs([0.1, 0.2], [3.0, 7.0], [True, False]).span() == (3.0, 7.0)
    assert all(math.isnan(edge) for edge in CandidatePairs.pooled([]).span())


def test_lines():
    """The report prints the base rate first, then the AUCs — and only when the guard passes.

    Both degenerate directions are refused with the reason stated, and no AUC appears anywhere in a refused
    report: a separability number on a degenerate population reads like a result, which is the failure mode.
    """
    healthy = ShortcutReport(label="movie", candidates=_uniform(size=20, positives=4)).lines()
    assert healthy[0].startswith("movie: base rate 0.2000 (4/20")
    assert "AUC(P)=" in healthy[1] and "AUC(-distance)=" in healthy[1]
    assert any("band 0 [" in line for line in healthy)

    for degenerate in (_uniform(size=40, positives=39), _uniform(size=1000, positives=1)):
        lines = ShortcutReport(label="movie", candidates=degenerate).lines()
        assert lines[0].startswith("movie: base rate")
        assert "REFUSED" in lines[1]
        assert not any("AUC(" in line for line in lines)

    empty = ShortcutReport(label="movie", candidates=CandidatePairs.pooled([])).lines()
    assert "nothing to rank" in empty[1]


def test_lines_bands_at_the_requested_resolution():
    """`bands` is a report parameter: positives concentrate at short range, so the split has to be re-choosable."""
    lines = ShortcutReport(label="movie", candidates=_uniform(size=20, positives=4), bands=2).lines()

    assert sum("band " in line for line in lines) == 2


def test_candidates():
    """`candidates` reads the DETECTIONS' in-gate pairs, matched against ground truth by the distance matcher."""
    detections = _detections()
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 20], [1, 0, 0, 9], [1, 0, 0, 21]], [[0, 1]])
    affinity = _Affinity({0: np.array([[0.9, 0.3, 0.1], [0.2, 0.4, 0.8]], dtype=np.float64)})

    candidates = ShortcutDiagnosis.candidates(detections, affinity, truth, _UNIT, _GATE_UM)

    assert candidates.probability.tolist() == [0.9, 0.3]
    assert candidates.is_true.tolist() == [True, False]


def test_by_movie(monkeypatch: pytest.MonkeyPatch):
    """Every requested movie's population, keyed by stem, off one mounted tracker."""
    detections = _detections()
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 2], [0, 0, 0, 20], [1, 0, 0, 9], [1, 0, 0, 21]], [[0, 1]])
    affinity = _Affinity({0: np.array([[0.9, 0.3, 0.1], [0.2, 0.4, 0.8]], dtype=np.float64)})
    proxy = TestMovieProxy(
        paths=(Path("m.zarr"),), truths=(AnnotatedTracks(graph=truth, estimated_node_count=5),), spacing=_UNIT
    )
    monkeypatch.setattr(TestMovieProxy, "load", classmethod(lambda cls, root, stems: proxy))
    monkeypatch.setattr(
        DenseDiagnosis, "shipped", staticmethod(lambda root, device, threshold: _FakeTracker(detections, affinity))
    )
    diagnosis = ShortcutDiagnosis(device="cpu", gate_um=_GATE_UM)

    by_movie = diagnosis.by_movie(cast(DataRoot, None), ("m",))

    assert list(by_movie) == ["m"]
    assert by_movie["m"].is_true.tolist() == [True, False]

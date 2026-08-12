"""Unit tests for the ensemble-dispersion diagnosis — the per-seed spread the blend's weighted sum discards.

The fixtures are hand-built matrices rather than mounted seeds: what is under test is the bookkeeping (which
pair each fate names, which row space it is read from, and whether a ranking number is allowed out at all),
not the transformer.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack.analysis.seed_dispersion import (
    DispersionSample,
    PairScores,
    Quartiles,
    SeedDispersion,
    SeedDispersionDiagnosis,
    SeedScoreIndex,
    Separation,
)
from celltrack.edges.blended_edge_scoring import GapSeedScores
from celltrack.eval.dense_diagnosis import DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker, TrackedVideo
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import NodeMatching
from core.paths import DataRoot


class _Affinity:
    """A fixed per-gap probability matrix indexed by timepoint — the `EdgeAffinity` contract, canned."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


# Two sources at t0 and three targets at t1. Source 0's true successor is 3 but the tracker took 2; source 1
# reproduces its annotated edge to 4. The seeds disagree on source 0's row and agree exactly on source 1's.
_SEED_PROBABILITIES = np.array(
    [[[0.70, 0.20, 0.05], [0.05, 0.05, 0.90]], [[0.60, 0.30, 0.05], [0.05, 0.05, 0.90]]], dtype=np.float64
)
_SEED_LOGITS = np.array(
    [[[1.0, 0.0, -2.0], [-1.0, -1.0, 2.0]], [[0.5, 0.4, -2.0], [-1.0, -1.0, 2.0]]], dtype=np.float64
)
_BLENDED = np.array([[0.68, 0.22, 0.05], [0.05, 0.05, 0.90]], dtype=np.float64)


def _scene() -> tuple[TrackGraph, TrackGraph, NodeMatching, SeedScoreIndex]:
    """The prediction, the annotation, their matching, and the per-seed index over the scored node set."""
    coordinates = np.array([[0, 0, 0, 0], [0, 0, 0, 20], [1, 0, 0, 3], [1, 0, 0, 9], [1, 0, 0, 20]], dtype=np.int64)
    scored = TrackGraph(np.arange(5, dtype=np.int64), coordinates, np.empty((0, 2), dtype=np.int64))
    prediction = TrackGraph(np.arange(5, dtype=np.int64), coordinates, np.array([[0, 2], [1, 4]], dtype=np.int64))
    truth = TrackGraph(np.arange(5, dtype=np.int64), coordinates, np.array([[0, 3], [1, 4]], dtype=np.int64))
    index = SeedScoreIndex.of(
        scored,
        [GapSeedScores(timepoint=0, logits=_SEED_LOGITS, probabilities=_SEED_PROBABILITIES)],
        _Affinity({0: _BLENDED}),
    )
    return prediction, truth, NodeMatching(gt_rows=np.arange(5, dtype=np.int64)), index


def test_probability_spread():
    """The probability spread is the seeds' widest gap — `|p1 - p2|` for the shipped two, sign-free for any k."""
    assert PairScores((0.7, 0.6), (1.0, 0.5), 0.68).probability_spread() == pytest.approx(0.1)
    assert PairScores((0.4, 0.9, 0.5), (0.0, 0.0, 0.0), 0.5).probability_spread() == pytest.approx(0.5)


def test_logit_spread():
    """The same disagreement in RAW logit units — the space the blend's weighted sum is taken in."""
    assert PairScores((0.7, 0.6), (1.0, 0.5), 0.68).logit_spread() == pytest.approx(0.5)


def test_seed_score_index_of():
    """`SeedScoreIndex.of` indexes the per-seed gaps by timepoint over the scored graph's own row space."""
    _, _, _, index = _scene()

    assert index.slots.source(1) == (0, 1)
    assert set(index.by_timepoint) == {0}


def test_pair():
    """`pair` reads one source→target cell from every seed and from the blend, and reports absent what it can't."""
    _, _, _, index = _scene()

    scores = index.pair(0, 2)

    assert scores is not None
    assert scores.seed_probabilities == (0.70, 0.60)
    assert scores.seed_logits == (1.0, 0.5)
    assert scores.blended_probability == 0.68
    assert index.pair(0, 1) is None  # a "target" in the source's own frame is no column of this gap
    assert index.pair(2, 3) is None  # gap 1 was never scored
    assert index.pair(99, 2) is None  # a node the seeds never scored (a bridge's invented midpoint)


def test_quartiles_of():
    """`Quartiles.of` reports the median and the quarter points — all NaN on an empty population, never zero."""
    quartiles = Quartiles.of(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert (quartiles.lower, quartiles.median, quartiles.upper) == (2.0, 3.0, 4.0)
    assert np.isnan(Quartiles.of(np.array([])).median)


def test_spread():
    """The inter-quartile range is the width two populations' medians have to be read against."""
    assert Quartiles.of(np.array([1.0, 2.0, 3.0, 4.0, 5.0])).spread() == pytest.approx(2.0)


def test_dispersion_sample_of():
    """`DispersionSample.of` collects the looked-up pairs into aligned columns and keeps the drop count."""
    sample = DispersionSample.of([PairScores((0.7, 0.6), (1.0, 0.5), 0.68)], unscored=2)

    assert sample.probability_spread.tolist() == pytest.approx([0.1])
    assert sample.logit_spread.tolist() == pytest.approx([0.5])
    assert sample.blended_probability.tolist() == [0.68]
    assert sample.unscored == 2


def test_dispersion_sample_size():
    """`size` counts the pairs that carried a dispersion at all — the dropped ones are not among them."""
    assert DispersionSample.of([PairScores((0.7, 0.6), (1.0, 0.5), 0.68)], unscored=5).size() == 1


def _separation(mislinked: list[float], correct: list[float]) -> Separation:
    """A separation whose three statistics all carry the same values, so each AUC is checkable by hand."""

    def column_of(values: list[float]) -> DispersionSample:
        column = np.array(values, dtype=np.float64)
        return DispersionSample(probability_spread=column, logit_spread=column, blended_probability=-column, unscored=0)

    return Separation(label="test", correct=column_of(correct), mislinked=column_of(mislinked))


def test_positives():
    """The minority class is the mislinked pairs — the count the AUC's sampling error rests on."""
    assert _separation([1.0, 2.0], [0.0] * 8).positives() == 2


def test_separation_size():
    """The ranked population is both classes together."""
    assert _separation([1.0, 2.0], [0.0] * 8).size() == 10


def test_base_rate():
    """The base rate is the mislinked share of the ranked population — NaN when there is no population."""
    assert _separation([1.0, 2.0], [0.0] * 8).base_rate() == pytest.approx(0.2)
    assert np.isnan(_separation([], []).base_rate())


def test_is_reportable():
    """The shared guard: a near-all-negative or near-all-positive split gets no AUC at all."""
    assert _separation([1.0] * 3, [0.0] * 97).is_reportable()  # 0.03 — the dense mislink regime
    assert not _separation([1.0], [0.0] * 999).is_reportable()  # 0.001 — too few positives to rank
    assert not _separation([1.0] * 9, [0.0]).is_reportable()  # 0.9 — nothing left to separate from


def test_auc_probability_spread():
    """A dispersion that is strictly larger on every mislink ranks them perfectly; an identical one is chance."""
    assert _separation([1.0, 2.0], [0.0, 0.5]).auc_probability_spread() == pytest.approx(1.0)
    assert _separation([1.0, 1.0], [1.0, 1.0]).auc_probability_spread() == pytest.approx(0.5)  # ties score 0.5


def test_auc_logit_spread():
    """The logit statistic is ranked the same way, independently of the probability one."""
    assert _separation([1.0, 2.0], [0.0, 0.5]).auc_logit_spread() == pytest.approx(1.0)


def test_auc_blended_probability():
    """The reference is stated low-P-first, so it reads on the same 'higher = more likely mislinked' scale."""
    separation = Separation(
        label="test",
        correct=DispersionSample(np.array([0.0]), np.array([0.0]), np.array([0.9]), unscored=0),
        mislinked=DispersionSample(np.array([0.0]), np.array([0.0]), np.array([0.1]), unscored=0),
    )

    assert separation.auc_blended_probability() == pytest.approx(1.0)


def test_auc_softmax_headroom():
    """The control ranks by `4P(1-P)` alone — a mid-range mislink outranks a saturated correct edge, seeds unread.

    Both populations here carry the SAME probability spread, so any separation the control reports is the
    softmax level and nothing else — which is exactly the part of `AUC(|p1-p2|)` it exists to account for.
    """
    separation = Separation(
        label="test",
        correct=DispersionSample(np.array([0.05]), np.array([0.05]), np.array([0.98]), unscored=0),
        mislinked=DispersionSample(np.array([0.05]), np.array([0.05]), np.array([0.50]), unscored=0),
    )

    assert separation.auc_softmax_headroom() == pytest.approx(1.0)
    assert separation.auc_probability_spread() == pytest.approx(0.5)  # the spreads themselves are identical


def test_dispersion_sample_lines():
    """A population's own row: its size and drop count first, then a median and IQR for each statistic."""
    sample = DispersionSample(
        probability_spread=np.array([0.1, 0.3]),
        logit_spread=np.array([1.0, 3.0]),
        blended_probability=np.array([0.5, 0.9]),
        unscored=2,
    )

    reported = sample.lines("CORRECT edges")

    assert reported[0] == "CORRECT edges: n=2 (unscored pairs dropped: 2)"
    assert "|p1-p2|  median 0.2000" in reported[1]
    assert "|l1-l2|  median 2.0000" in reported[2]
    assert "blended P  median 0.7000" in reported[3]


def test_separation_lines():
    """The population line comes first, and a failed guard REPLACES the numbers rather than annotating them."""
    reported = _separation([1.0] * 3, [0.0] * 97).lines()
    assert reported[0].startswith("test: base rate 0.0300 (3/100 ranked pairs mislinked)")
    assert "AUC(|p1-p2|)=1.000" in reported[1]

    refused = _separation([1.0] * 9, [0.0]).lines()
    assert refused[1].startswith("  REFUSED:")
    assert not any("AUC(" in line for line in refused)


def test_seed_dispersion_of():
    """Each annotated edge's fate names its pair: the correct edge, the wrong target we took, the true one we missed.

    Source 0's true successor is 3 but the tracker linked 2, so `chosen` reads column 2 (seeds 0.70/0.60) and
    `missed` reads column 3 (0.20/0.30); source 1 reproduces its edge to 4, where the seeds agree exactly.
    """
    prediction, truth, matching, index = _scene()

    dispersion = SeedDispersion.of(prediction, truth, matching, index)

    assert dispersion.correct.probability_spread.tolist() == pytest.approx([0.0])
    assert dispersion.chosen.probability_spread.tolist() == pytest.approx([0.1])
    assert dispersion.missed.probability_spread.tolist() == pytest.approx([0.1])
    assert dispersion.chosen.logit_spread.tolist() == pytest.approx([0.5])
    assert dispersion.missed.logit_spread.tolist() == pytest.approx([0.4])
    assert dispersion.correct.unscored == 0


def test_seed_dispersion_of_drops_a_pair_no_gap_scored():
    """A pair the seeds never scored is COUNTED as dropped, never imputed — a bridge node has no dispersion."""
    prediction, truth, matching, _ = _scene()
    empty = SeedScoreIndex.of(prediction, [], _Affinity({}))

    dispersion = SeedDispersion.of(prediction, truth, matching, empty)

    assert dispersion.correct.size() == 0
    assert dispersion.correct.unscored == 1
    assert dispersion.chosen.unscored == 1


def test_separations():
    """Both mislink halves are ranked against the SAME correct population — the two decisive readings."""
    prediction, truth, matching, index = _scene()

    labelled = SeedDispersion.of(prediction, truth, matching, index).separations()

    assert [separation.label for separation in labelled] == [
        "MISLINK: the wrong edge we chose",
        "MISLINK: the true edge we missed",
    ]
    assert all(separation.correct.size() == 1 for separation in labelled)


def test_of_movie(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The run scores the seeds over the DETECTIONS the blended affinity was built on, not the finished graph.

    Post-processing moves the finished graph into a different row space, so scoring the seeds over it would
    read a pair's dispersion and its blended probability off two different indexings of the same movie.
    """
    prediction, truth, _, _ = _scene()
    detections = TrackGraph(prediction.node_ids, prediction.coordinates, np.empty((0, 2), dtype=np.int64))
    affinity = _Affinity({0: _BLENDED})
    scored_over: dict[str, TrackGraph] = {}

    class _Scorer:
        def seed_scores(self, path: Path, graph: TrackGraph, device: str) -> list[GapSeedScores]:
            scored_over["graph"] = graph
            return [GapSeedScores(timepoint=0, logits=_SEED_LOGITS, probabilities=_SEED_PROBABILITIES)]

    class _Tracker:
        edge_scorer = _Scorer()

        def run_scored(self, name: str, path: Path) -> TrackedVideo:
            return TrackedVideo(graph=prediction, detections=detections, affinity=affinity)

    monkeypatch.setattr(
        TestMovieProxy,
        "load",
        classmethod(
            lambda cls, root, stems: TestMovieProxy(
                paths=(tmp_path / "m.zarr",),
                truths=(AnnotatedTracks(graph=truth, estimated_node_count=len(truth.node_ids)),),
                spacing=Spacing(1.0, 1.0, 1.0),
            )
        ),
    )
    monkeypatch.setattr(
        DenseDiagnosis, "configured", staticmethod(lambda root, device, config: cast(CellTracker, _Tracker()))
    )

    dispersion = SeedDispersionDiagnosis(device="cpu").of_movie(DataRoot(tmp_path), "m", TrackerConfig())

    assert scored_over["graph"] is detections
    assert dispersion.chosen.probability_spread.tolist() == pytest.approx([0.1])

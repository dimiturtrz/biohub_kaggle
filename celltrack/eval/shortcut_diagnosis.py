"""Did the edge transformer learn APPEARANCE, or re-learn DISTANCE? — separability measured on real detections.

The head sees each candidate pair's relative displacement, and true motion is small (median ~1.7 um), so
"nearest is the successor" is an excellent rule almost everywhere — and exactly wrong in the crowded,
fast-moving neighbourhoods where the score is actually lost. If the head leaned on that shortcut, the learned
term in the linker's `cost = distance - bonus * P` is partly REDUNDANT with the distance term it exists to
correct, which is what a broad flat `affinity_bonus` optimum (20-30 rather than a peak) looks like.

The measurement is a ranking one: over the candidate pairs the linker would actually consider (both endpoints
real detections, within the gate), compare the AUC of `P` against the AUC of `-distance`, and then — the
decisive read — the AUC of `P` WITHIN matched distance bands. Distance is held near-constant inside a band, so
a band AUC at chance says `P` carries nothing beyond proximity, while a band AUC well above chance says it
carries appearance the geometry does not.

*** WHY THIS IS A TOOL AND NOT A SCRATCH SCRIPT. *** Run against the GROUND-TRUTH graph instead of real
detections the same question answers itself: rows and columns are annotated nodes only, the annotation is
sparse, and within a 10 um gate 446 of 457 candidate pairs came back TRUE (base rate 0.976). Every AUC read
~1.0 by construction and every distance band was all-positive. A separability number on a degenerate
population is WORSE than no number, because it reads as a result — so this module prints the base rate FIRST
and refuses to print any AUC outside `_MIN_BASE_RATE .. _MAX_BASE_RATE`, per population and per band.

Reading the output later: on the dense movie the shipped model's mislinks are ~100% affinity-inverted, with
mean P_true 0.134 against P_chosen 0.686 (`dense_diagnosis`) — the head is CONFIDENTLY wrong on real
detections, while being well calibrated on GT pairs (at P >= 0.6, mean predicted 0.987 vs 0.985 actually
true). Calibration is a property of the population it is measured on; this tool exists to characterise the
population that decides the score.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from celltrack.affinity import EdgeAffinity
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.linkers.linkers import LinkerConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The base-rate interval a separability number is allowed to be reported on. Both ends are derived from what
# the candidate population IS, not chosen round:
#
# UPPER 0.5 — a source has at most one true successor, so the base rate is 1/(mean in-gate candidates per
# judged source). A rate of 0.5 already means fewer than two candidates per source: the gate admits the truth
# and almost nothing else, and there is nothing left to separate the truth FROM. This is exactly the
# GT-graph artefact (0.976 = 1.02 candidates per source) that made the first run of this measurement
# worthless; a real-detection gate at 10 um holds order ten neighbours, i.e. a rate well under 0.3.
#
# LOWER 0.01 — the AUC's sampling error is set by the MINORITY class. A real-detection movie yields order
# 1e4 in-gate judged pairs, so 1% keeps the positives in the hundreds, where the rank AUC's standard error
# (~0.5/sqrt(n_minority)) sits at the project's 0.01-0.02 score noise floor. Below that the number moves more
# with which handful of positives landed in the set than with the model. The counts are printed beside the
# rate so the reader can redo that arithmetic for the population actually measured.
_MIN_BASE_RATE = 0.01
_MAX_BASE_RATE = 0.5

# Equal-COUNT bands, so every band carries the same share of the population and none is refused merely for
# being thin. Five puts each band at a fifth of the in-gate distance range — order 1-2 um wide at the 10 um
# gate, i.e. comparable to the ~1.7 um median true step, which is the point of the split: inside one band
# distance no longer identifies the successor, so whatever ranking survives is not proximity.
_DISTANCE_BANDS = 5

# Pearson r is undefined below two points (and on a constant arm), so it is reported as NaN rather than raised.
_MIN_CORRELATION_POINTS = 2

_CHANCE_AUC = 0.5


@dataclass(frozen=True)
class MatchedAnnotation:
    """The ground truth as seen from DETECTION rows: which pairs the annotation joins, and which it judges at all.

    Truth reaches a detection only through a `DistanceMatcher`, and the annotation is sparse, so "not an
    annotated edge" is not the same as "false": a pair with an unmatched endpoint, or one whose source the
    annotation gives no successor AND whose target it gives no parent, is unjudged, and counting it as a
    negative would manufacture the base rate the measurement is about. Only rows the annotation speaks about
    are judged — the same active-row/column convention `core.metrics.edge_auc` applies to the training matrices.
    """

    gt_rows: Int[np.ndarray, "n"]
    codes: Int[np.ndarray, "e"]
    has_successor: Bool[np.ndarray, "g"]
    has_parent: Bool[np.ndarray, "g"]
    stride: int

    @classmethod
    def of(cls, truth: TrackGraph, matching: NodeMatching) -> "MatchedAnnotation":
        """Index the annotated edges as flat `source * stride + target` row codes, plus the judged-row masks."""
        rows = truth.edge_rows()
        stride = max(len(truth.node_ids), 1)
        return cls(
            gt_rows=matching.gt_rows,
            codes=rows[:, 0] * stride + rows[:, 1],
            has_successor=np.bincount(rows[:, 0], minlength=stride) > 0,
            has_parent=np.bincount(rows[:, 1], minlength=stride) > 0,
            stride=stride,
        )

    def judge(
        self, sources: Int[np.ndarray, "s"], targets: Int[np.ndarray, "t"]
    ) -> tuple[Bool[np.ndarray, "s t"], Bool[np.ndarray, "s t"]]:
        """`(judged, is_true)` for every pair of these detection rows — the truth a candidate block carries."""
        source = self.gt_rows[sources][:, None]
        target = self.gt_rows[targets][None, :]
        matched = (source != UNMATCHED) & (target != UNMATCHED)
        safe_source = np.where(matched, source, 0)
        safe_target = np.where(matched, target, 0)
        judged = matched & (self.has_successor[safe_source] | self.has_parent[safe_target])
        return judged, judged & np.isin(safe_source * self.stride + safe_target, self.codes)


@dataclass(frozen=True)
class CandidatePairs:
    """The in-gate candidate pairs a linker would weigh on REAL detections: each one's `P`, distance and truth.

    This is the population the question is about. Scoring the ground-truth graph instead answers a different,
    self-fulfilling question (see the module docstring), so the endpoints here are detector nodes and truth
    arrives through a `DistanceMatcher` — a pair is true only when both endpoints matched ground-truth nodes
    the annotation joins, and unjudged pairs are dropped rather than counted false.
    """

    probability: Float[np.ndarray, "c"]
    distance_um: Float[np.ndarray, "c"]
    is_true: Bool[np.ndarray, "c"]

    @classmethod
    def of(
        cls,
        detections: TrackGraph,
        annotation: MatchedAnnotation,
        affinity: EdgeAffinity,
        spacing: Spacing,
        gate_um: float,
    ) -> "CandidatePairs":
        """Every judged, in-gate source->target pair of the detection graph, with its learned P and its distance."""
        positions_um = spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        gaps = [cls._gap(annotation, positions_um, gate_um, gap) for gap in cls._scored_gaps(timepoints, affinity)]
        return cls.pooled(gaps)

    @classmethod
    def pooled(cls, populations: Sequence["CandidatePairs"]) -> "CandidatePairs":
        """Several movies' (or gaps') candidates as one population — what the pooled read-out is computed over."""
        empty = np.empty(0, dtype=np.float64)
        if not populations:
            return cls(probability=empty, distance_um=empty, is_true=np.empty(0, dtype=bool))
        return cls(
            probability=np.concatenate([population.probability for population in populations]),
            distance_um=np.concatenate([population.distance_um for population in populations]),
            is_true=np.concatenate([population.is_true for population in populations]),
        )

    def size(self) -> int:
        """How many judged candidate pairs the population holds."""
        return int(self.is_true.size)

    def positives(self) -> int:
        """How many of them are true successor pairs — the minority class the AUC's precision rests on."""
        return int(self.is_true.sum())

    def base_rate(self) -> float:
        """The share of judged candidates that are true — NaN on an empty population. Printed before any AUC."""
        return float(self.positives() / self.size()) if self.size() else float("nan")

    def is_reportable(self) -> bool:
        """Whether the base rate leaves a separability number any meaning — the guard every AUC is behind."""
        return bool(_MIN_BASE_RATE <= self.base_rate() <= _MAX_BASE_RATE)

    def auc_probability(self) -> float:
        """How well the learned `P` alone ranks true pairs above false ones (0.5 = chance)."""
        return self._ranking(self.probability)

    def auc_proximity(self) -> float:
        """The same ranking quality for proximity alone — the shortcut's own score, as `P`'s reference point."""
        return self._ranking(-self.distance_um)

    def correlation(self) -> float:
        """Pearson r between `P` and distance — a strongly negative r is the head tracking geometry directly."""
        if self.size() < _MIN_CORRELATION_POINTS or not self.probability.std() or not self.distance_um.std():
            return float("nan")
        return float(np.corrcoef(self.probability, self.distance_um)[0, 1])

    def bands(self, count: int = _DISTANCE_BANDS) -> tuple["CandidatePairs", ...]:
        """The population split into equal-count distance bands, ascending — where `P` is asked to rank alone."""
        order = np.argsort(self.distance_um, kind="stable")
        return tuple(self._select(chunk) for chunk in np.array_split(order, count))

    def span(self) -> tuple[float, float]:
        """The population's distance range in micrometres — a band's label, NaN when it holds nothing."""
        if not self.size():
            return float("nan"), float("nan")
        return float(self.distance_um.min()), float(self.distance_um.max())

    def _select(self, rows: Int[np.ndarray, "k"]) -> "CandidatePairs":
        """The sub-population at these rows."""
        return CandidatePairs(
            probability=self.probability[rows], distance_um=self.distance_um[rows], is_true=self.is_true[rows]
        )

    def _ranking(self, scores: Float[np.ndarray, "c"]) -> float:
        """Rank (ROC) AUC of `scores` against the truth labels — the Mann-Whitney statistic, ties scored 0.5.

        ROC rather than the average precision `core.metrics.edge_auc` reports, because these numbers are
        compared ACROSS populations (whole movie vs distance band) whose base rates differ by design, and
        average precision moves with the base rate while the rank AUC does not.
        """
        positives = self.positives()
        negatives = self.size() - positives
        if positives == 0 or negatives == 0:
            return float("nan")
        ranks = rankdata(scores)
        return float((ranks[self.is_true].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    @staticmethod
    def _scored_gaps(
        timepoints: Int[np.ndarray, "n"], affinity: EdgeAffinity
    ) -> Iterator[tuple[Int[np.ndarray, "s"], Int[np.ndarray, "t"], Float[np.ndarray, "s t"]]]:
        """Each `t -> t+1` gap the affinity actually scored, as its source rows, target rows and probabilities.

        Rows follow `np.flatnonzero(timepoints == t)` over the DETECTION graph, which is the order the affinity
        matrices are built in — the same alignment the linker's cost relies on.
        """
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probabilities = affinity.probabilities(int(timepoint))
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            if probabilities is None or not len(sources) or not len(targets):
                continue
            yield sources, targets, probabilities

    @classmethod
    def _gap(
        cls,
        annotation: MatchedAnnotation,
        positions_um: Float[np.ndarray, "n 3"],
        gate_um: float,
        gap: tuple[Int[np.ndarray, "s"], Int[np.ndarray, "t"], Float[np.ndarray, "s t"]],
    ) -> "CandidatePairs":
        """One frame gap's judged, in-gate candidates."""
        sources, targets, probabilities = gap
        distance = cdist(positions_um[sources], positions_um[targets])
        if probabilities.shape != distance.shape:
            message = f"affinity block {probabilities.shape} does not align with the {distance.shape} frame gap"
            raise ValueError(message)
        judged, is_true = annotation.judge(sources, targets)
        keep = judged & (distance <= gate_um)
        return cls(probability=probabilities[keep], distance_um=distance[keep], is_true=is_true[keep])


@dataclass(frozen=True)
class ShortcutReport:
    """One population's read-out: the base rate FIRST, and every AUC behind the guard that base rate sets.

    The refusal is the deliverable as much as the numbers are. A separability figure computed on a population
    that is 98% positive is not a weak result, it is a measurement of the population's construction, and it
    reads to a later reader exactly like a strong model — so nothing downstream of a failed guard is printed.
    """

    label: str
    candidates: CandidatePairs

    def lines(self) -> list[str]:
        """The report as log lines: population line first, then either the separability block or the refusal."""
        head = self._population(self.label, self.candidates)
        if not self.candidates.is_reportable():
            return [head, f"  REFUSED: {self._refusal(self.candidates)}"]
        return [head, self._separability(), *self._band_lines()]

    def _separability(self) -> str:
        """The pooled ranking read: `P` alone, proximity alone, and how far the two are the same quantity."""
        return (
            f"  AUC(P)={self.candidates.auc_probability():.3f}  "
            f"AUC(-distance)={self.candidates.auc_proximity():.3f}  "
            f"corr(P, distance)={self.candidates.correlation():+.3f}  (chance={_CHANCE_AUC:.1f})"
        )

    def _band_lines(self) -> list[str]:
        """`P`'s ranking quality inside each matched-distance band — the decisive read, or that band's refusal."""
        lines = ["  within matched distance bands (P must rank with distance held near-constant):"]
        for index, band in enumerate(self.candidates.bands()):
            low, high = band.span()
            head = f"    band {index} [{low:5.2f}, {high:5.2f}] um  " + self._population("n", band)
            lines.append(
                f"{head}  AUC(P)={band.auc_probability():.3f}"
                if band.is_reportable()
                else f"{head}  REFUSED: {self._refusal(band)}"
            )
        return lines

    @staticmethod
    def _population(label: str, candidates: CandidatePairs) -> str:
        """The counts every number in the report has to be read against — always emitted before that number."""
        return (
            f"{label}: base rate {candidates.base_rate():.4f} "
            f"({candidates.positives()}/{candidates.size()} judged in-gate pairs true)"
        )

    @staticmethod
    def _refusal(candidates: CandidatePairs) -> str:
        """Why this population gets no AUC — stated as the defect, so the refusal is actionable, not a shrug."""
        if not candidates.size():
            return "no judged in-gate candidate pairs at all — nothing to rank"
        rate = candidates.base_rate()
        side = "so nearly every candidate is true" if rate > _MAX_BASE_RATE else "so almost none is"
        return (
            f"base rate {rate:.4f} is outside [{_MIN_BASE_RATE}, {_MAX_BASE_RATE}], {side}; an AUC here "
            "measures how the population was built, not what the model knows (see the module docstring)"
        )


@dataclass(frozen=True)
class ShortcutDiagnosis:
    """Running the shipped tracker on a movie and reading its real-detection candidate pairs' separability."""

    device: str
    gate_um: float
    # None keeps the shipped detection threshold the mounted tracker already carries.
    threshold: float | None = None

    @staticmethod
    def candidates(
        detections: TrackGraph, affinity: EdgeAffinity, truth: TrackGraph, spacing: Spacing, gate_um: float
    ) -> CandidatePairs:
        """One movie's judged in-gate candidates — the reusable core, so a caller can score a MODIFIED affinity.

        Takes the detections and the affinity rather than a tracker, because those are exactly what the
        population is made of: a caller holding a finetuned edge head scores it against the same detections
        without re-running detection, and a caller that already ran the tracker passes what `run_scored` returned.
        """
        matching = DistanceMatcher(spacing=spacing).match(detections, truth)
        annotation = MatchedAnnotation.of(truth, matching)
        return CandidatePairs.of(detections, annotation, affinity, spacing, gate_um)

    def by_movie(self, root: DataRoot, stems: tuple[str, ...]) -> dict[str, CandidatePairs]:
        """Each requested movie's candidate population, keyed by stem — the tracker mounted once for all of them.

        One `run_scored` pass per movie hands back the detections AND the affinity the linker costs with, so the
        population is read off the pass the tracker already made rather than costing a second forward.
        """
        proxy = TestMovieProxy.load(root, stems)
        tracker = DenseDiagnosis.shipped(root, self.device, self.threshold)
        populations: dict[str, CandidatePairs] = {}
        for path, truth in zip(proxy.paths, proxy.truths, strict=True):
            scored = tracker.run_scored(path.name, path)
            population = self.candidates(scored.detections, scored.affinity, truth.graph, proxy.spacing, self.gate_um)
            populations[path.stem] = population
        return populations


def main() -> None:
    """Report, per movie and pooled, whether the edge head separates true pairs beyond what proximity already does."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Is the edge head's P appearance, or a re-learned distance?")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", action="append", dest="movies", help="movie stem (repeatable)")
    parser.add_argument("--threshold", type=float, help="detector threshold; default is the shipped one")
    parser.add_argument("--gate-um", type=float, default=LinkerConfig().gate_um, help="linker gate radius, um")
    args = parser.parse_args()

    diagnosis = ShortcutDiagnosis(device=args.device, gate_um=args.gate_um, threshold=args.threshold)
    by_movie = diagnosis.by_movie(DataRoot.from_config(args.config), tuple(args.movies or (DENSE_MOVIE,)))
    threshold = "shipped" if args.threshold is None else f"{args.threshold:g}"
    logger.info("threshold=%s  gate=%g um  movies=%s", threshold, args.gate_um, ", ".join(by_movie))
    reports = [ShortcutReport(label=stem, candidates=candidates) for stem, candidates in by_movie.items()]
    reports.append(ShortcutReport(label="POOLED", candidates=CandidatePairs.pooled(list(by_movie.values()))))
    for line in [line for report in reports for line in report.lines()]:
        logger.info("%s", line)


if __name__ == "__main__":
    main()

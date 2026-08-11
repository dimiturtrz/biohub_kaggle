"""Do the two INDEPENDENTLY-TRAINED edge seeds disagree where the linker mislinks? — the blend's discarded spread.

`BlendedEdgeTransformerScorer` mounts two separately-trained heads, sums their RAW logits 0.8/0.2 and
soft-maxes. The weighted sum keeps the MEAN and destroys the DISPERSION — the one quantity an ensemble has
that a single model does not — before any linker, cost or diagnosis can look at it. This module asks whether
that discarded quantity knows anything: is the seeds' disagreement systematically LARGER on the pairs the
tracker gets wrong than on the ones it gets right?

*** THIS IS NOT THE REFUTED AGREEMENT GATE. *** Mutual consistency between the FORWARD and the REVERSE
affinity — one model scored both ways, `BlendedEdgeTransformerScorer.fuse` and the linker's agreement knobs —
was tested across all four linker pairings and is dead. That is ONE model asked the same question twice. This
is TWO models trained from different initialisations asked the question once, which is a different quantity
supporting a different claim, and it must not be retired by association with that result.

Both dispersion statistics are reported, because the blend happens in LOGIT space and a probability spread
would hide a pure scale difference there: `|p1 - p2|` (what the linker's input would move by) and `|l1 - l2|`
(the raw pre-softmax gap the weighted sum actually acts on). Reported as distributions — median and IQR — not
means, since a handful of near-tied pairs would carry a mean anywhere. The probability spread additionally
carries a CONTROL (`Separation.auc_softmax_headroom`), because it is not independent of the probability level
it sits at and the three populations here sit at very different levels by construction.

The decisive number is the rank AUC of dispersion separating mislinked pairs from correct ones, and it sits
behind `celltrack.eval.shortcut_diagnosis.BaseRateGuard`, the shared refusal that keeps a separability figure
off a degenerate population — this project has been burned by that twice already, and the mislink class here
is small (order 3% of the annotated edges) precisely because the tracker is mostly right.

Reading either outcome, fixed before the run so it cannot be narrated afterwards. Dispersion clearly higher
on mislinks means the ensemble carries an uncertainty signal the mean discards, and something can be built on
it. Dispersion flat across fates means the two heads are confidently wrong TOGETHER — which closes the
ensemble-uncertainty direction and is itself the strongest evidence available for a feature/resolution wall,
since two independently-trained models failing identically on the same pairs is not an initialisation
accident.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from celltrack.affinity import EdgeAffinity
from celltrack.edges.blended_edge_scoring import GapSeedScores
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis, DenseFateDiagnosis, Fate, GapSlots
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.proxy_eval import ConfigOverride
from celltrack.eval.shortcut_diagnosis import BaseRateGuard
from celltrack.operating_point import TrackerConfig
from core.data.tracks import Adjacency, TrackGraph
from core.metrics.edge_auc import EdgeAUC
from core.metrics.matching import DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CHANCE_AUC = 0.5


@dataclass(frozen=True)
class PairScores:
    """One candidate pair as every seed saw it, beside the blended number the linker actually consumed.

    Spread is `max - min` rather than a signed difference so it stays the same quantity for any seed count and
    carries no sign convention to get backwards; with the shipped two seeds it is exactly `|p1 - p2|`.
    """

    seed_probabilities: tuple[float, ...]
    seed_logits: tuple[float, ...]
    blended_probability: float

    def probability_spread(self) -> float:
        """How far apart the seeds' probabilities for this pair are — the disagreement in the linker's units."""
        return max(self.seed_probabilities) - min(self.seed_probabilities)

    def logit_spread(self) -> float:
        """The same disagreement in the RAW logit units the weighted sum is taken in, where a scale gap shows."""
        return max(self.seed_logits) - min(self.seed_logits)


@dataclass(frozen=True)
class SeedScoreIndex:
    """Per-seed gap scores reached by node ID — `GapSlots` over the graph the seeds were scored on.

    The blended affinity travels with them so a pair's dispersion and the number the linker consumed for that
    same pair come out of ONE lookup: the reference an AUC of dispersion has to be read against is what the
    mean already separates, and reading the two from separately-indexed structures is how they end up
    describing different pairs.
    """

    slots: GapSlots
    by_timepoint: dict[int, GapSeedScores]
    blended: EdgeAffinity

    @classmethod
    def of(cls, scored: TrackGraph, seed_scores: Sequence[GapSeedScores], blended: EdgeAffinity) -> "SeedScoreIndex":
        """Index one video's per-seed scores by node ID, over the graph they and the blend were computed on."""
        return cls(
            slots=GapSlots.of(scored),
            by_timepoint={gap.timepoint: gap for gap in seed_scores},
            blended=blended,
        )

    def pair(self, source_id: int, target_id: int) -> PairScores | None:
        """Every seed's probability and raw logit for one source→target pair, or `None` if no gap scored it."""
        source = self.slots.source(source_id)
        if source is None:
            return None
        timepoint, row = source
        column = self.slots.column(target_id, timepoint)
        gap = self.by_timepoint.get(timepoint)
        matrix = self.blended.probabilities(timepoint)
        if column is None or gap is None or matrix is None:
            return None
        return PairScores(
            seed_probabilities=tuple(float(value) for value in gap.probabilities[:, row, column]),
            seed_logits=tuple(float(value) for value in gap.logits[:, row, column]),
            blended_probability=float(matrix[row, column]),
        )


@dataclass(frozen=True)
class Quartiles:
    """A distribution's median and inter-quartile range — how a spread is reported, since a mean would not do.

    The populations here are small (order 40 mislinks) and heavy-tailed: one pair where the seeds happen to
    straddle a softmax saturation carries a mean somewhere the other thirty-nine never go. The median says
    where the typical pair sits and the IQR says how much of the population moves with it.
    """

    median: float
    lower: float
    upper: float

    @classmethod
    def of(cls, values: Float[np.ndarray, "n"]) -> "Quartiles":
        """The values' 25th, 50th and 75th percentiles — all NaN on an empty population, never zero."""
        if not values.size:
            return cls(median=float("nan"), lower=float("nan"), upper=float("nan"))
        lower, median, upper = (float(value) for value in np.percentile(values, (25, 50, 75)))
        return cls(median=median, lower=lower, upper=upper)

    def spread(self) -> float:
        """The inter-quartile range itself — the width the medians of two populations have to be read against."""
        return self.upper - self.lower


@dataclass(frozen=True)
class DispersionSample:
    """One fate's pairs as three aligned columns: both dispersion statistics and the blended P beside them.

    `unscored` is carried, not dropped silently: a pair whose endpoints the seeds never scored — a bridge's
    invented midpoint, or a chosen target more than one gap ahead — has no dispersion to report, and how many
    of a population went missing that way decides whether the rest of the population still stands for it.
    """

    probability_spread: Float[np.ndarray, "n"]
    logit_spread: Float[np.ndarray, "n"]
    blended_probability: Float[np.ndarray, "n"]
    unscored: int

    @classmethod
    def of(cls, scores: Sequence[PairScores], unscored: int) -> "DispersionSample":
        """Collect the looked-up pairs into columns, plus the count of pairs no gap could be found for."""
        return cls(
            probability_spread=np.array([score.probability_spread() for score in scores], dtype=np.float64),
            logit_spread=np.array([score.logit_spread() for score in scores], dtype=np.float64),
            blended_probability=np.array([score.blended_probability for score in scores], dtype=np.float64),
            unscored=unscored,
        )

    def lines(self, label: str) -> list[str]:
        """One population's distributions — median and IQR for both statistics, with its drop count beside them."""
        probability, logit = Quartiles.of(self.probability_spread), Quartiles.of(self.logit_spread)
        blended = Quartiles.of(self.blended_probability)
        return [
            f"{label}: n={self.size()} (unscored pairs dropped: {self.unscored})",
            f"  |p1-p2|  median {probability.median:.4f}  IQR [{probability.lower:.4f}, {probability.upper:.4f}]"
            f"  width {probability.spread():.4f}",
            f"  |l1-l2|  median {logit.median:.4f}  IQR [{logit.lower:.4f}, {logit.upper:.4f}]"
            f"  width {logit.spread():.4f}",
            f"  blended P  median {blended.median:.4f}  IQR [{blended.lower:.4f}, {blended.upper:.4f}]",
        ]

    def size(self) -> int:
        """How many pairs of this fate carried a dispersion at all."""
        return int(self.probability_spread.size)


@dataclass(frozen=True)
class Separation:
    """How well dispersion ranks ONE mislink population above the correct edges — the decisive read, guarded.

    The blended probability's own AUC rides along as the reference every dispersion AUC has to beat to have
    found anything NEW: the mean is already known to separate these populations (the dense mislinks are
    ~100% affinity-inverted, `dense_diagnosis`), so a dispersion AUC near it is not a second signal, and a
    dispersion AUC at chance says the discarded quantity carried nothing at all.
    """

    label: str
    correct: DispersionSample
    mislinked: DispersionSample

    def positives(self) -> int:
        """The minority class — the mislinked pairs, whose count sets the AUC's sampling error."""
        return self.mislinked.size()

    def size(self) -> int:
        """Every pair the ranking is taken over: the mislinked ones plus the correct ones they rank against."""
        return self.mislinked.size() + self.correct.size()

    def base_rate(self) -> float:
        """The share of the ranked population that is mislinked — printed BEFORE any AUC, never after."""
        return float(self.positives() / self.size()) if self.size() else float("nan")

    def is_reportable(self) -> bool:
        """Whether the base rate leaves a separability number any meaning — the shared repo-wide guard."""
        return BaseRateGuard.allows(self.base_rate())

    def auc_probability_spread(self) -> float:
        """Rank AUC of `|p1 - p2|` at putting mislinked pairs above correct ones (0.5 = the seeds know nothing)."""
        return self._auc(self.mislinked.probability_spread, self.correct.probability_spread)

    def auc_logit_spread(self) -> float:
        """The same for the RAW logit gap — the space the blend's weighted sum is actually taken in."""
        return self._auc(self.mislinked.logit_spread, self.correct.logit_spread)

    def auc_blended_probability(self) -> float:
        """The reference: how well the MEAN the blend kept already separates the two populations, low-P first.

        Negated because a mislinked pair is expected to carry the LOWER blended probability, so the reference
        is stated on the same "higher score = more likely mislinked" scale as the dispersion AUCs beside it.
        """
        return self._auc(-self.mislinked.blended_probability, -self.correct.blended_probability)

    def auc_softmax_headroom(self) -> float:
        """The control `AUC(|p1-p2|)` has to beat: how well the blend's DISTANCE FROM SATURATION ranks alone.

        A probability spread is not free of the level it sits at. Two seeds a fixed logit gap apart map to
        nearly the same probability when the softmax has saturated (P near 0 or 1) and to their widest
        separation at P = 0.5, so `|p1 - p2|` is bounded above by `4P(1-P)` up to the gap — and the three
        populations here sit at very different levels by construction (a correct edge is confident, the wrong
        edge we chose is mid-range, the true edge we missed is near zero). Ranking by the headroom ALONE, with
        no reference to the seeds at all, therefore reproduces whatever part of the probability-spread AUC is
        the level rather than the disagreement. If this control matches `AUC(|p1-p2|)`, the spread found
        nothing the blended mean did not already say; the raw logit gap, which no softmax has compressed, is
        the statistic that answers the question either way.
        """
        return self._auc(self._headroom(self.mislinked), self._headroom(self.correct))

    @staticmethod
    def _headroom(sample: DispersionSample) -> Float[np.ndarray, "n"]:
        """`4P(1-P)`: how much room the softmax leaves two seeds to differ in probability at this pair's level."""
        return 4.0 * sample.blended_probability * (1.0 - sample.blended_probability)

    def _auc(self, mislinked: Float[np.ndarray, "m"], correct: Float[np.ndarray, "c"]) -> float:
        """One statistic's rank AUC over the pooled population, mislinked labelled positive."""
        labels = np.concatenate([np.ones(mislinked.size, dtype=bool), np.zeros(correct.size, dtype=bool)])
        return EdgeAUC.ranked(np.concatenate([mislinked, correct]), labels)

    def lines(self) -> list[str]:
        """The population line first, then either the three AUCs or the refusal that replaces them."""
        head = (
            f"{self.label}: base rate {self.base_rate():.4f} ({self.positives()}/{self.size()} ranked pairs mislinked)"
        )
        if not self.is_reportable():
            return [head, f"  REFUSED: {BaseRateGuard.refusal(self.base_rate())}"]
        return [
            head,
            f"  AUC(|p1-p2|)={self.auc_probability_spread():.3f}  "
            f"AUC(|l1-l2|)={self.auc_logit_spread():.3f}  "
            f"AUC(4P(1-P), the saturation control)={self.auc_softmax_headroom():.3f}  "
            f"AUC(-blended P, the reference)={self.auc_blended_probability():.3f}  (chance={_CHANCE_AUC:.1f})",
        ]


@dataclass(frozen=True)
class SeedDispersion:
    """The three populations the fate split defines, each carrying both seeds' disagreement on its own pairs.

    `chosen` and `missed` are the two halves of the SAME mislink: the wrong edge the tracker took, and the
    true edge it did not. They are kept apart because they answer different questions — whether the seeds
    disagreed about the thing we did, and whether they disagreed about the thing we should have done — and a
    gate built on dispersion would need the first while a re-routing lever would need the second.
    """

    correct: DispersionSample
    chosen: DispersionSample
    missed: DispersionSample

    @classmethod
    def of(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, index: SeedScoreIndex
    ) -> "SeedDispersion":
        """Split every annotated edge by its `Fate` and read the seeds' disagreement on the pairs it names.

        `Fate` (per annotated edge) rather than `Charge` (per emitted link) because the three populations the
        question needs are all defined against the ANNOTATED edge: the correct ones, the wrong target we chose
        instead of it, and the true target we missed. `Charge.MISLINK` names the same wrong links (the two
        decompositions reconcile by construction) but has no handle at all on the true edge we did not emit.
        """
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        predicted_of_truth = DenseFateDiagnosis.predicted_of_truth(matching, len(truth.node_ids))
        successors = Adjacency.of(prediction).successors
        edges = truth.edge_rows()
        correct = cls._sample(
            index, prediction.node_ids, predicted_of_truth, edges[fates == Fate.CORRECT], successors=None
        )
        mislinked = edges[(fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE)]
        return cls(
            correct=correct,
            chosen=cls._sample(index, prediction.node_ids, predicted_of_truth, mislinked, successors=successors),
            missed=cls._sample(index, prediction.node_ids, predicted_of_truth, mislinked, successors=None),
        )

    @staticmethod
    def _sample(
        index: SeedScoreIndex,
        node_ids: Int[np.ndarray, "n"],
        predicted_of_truth: Int[np.ndarray, "g"],
        edges: Int[np.ndarray, "e 2"],
        successors: Mapping[int, Sequence[int]] | None,
    ) -> DispersionSample:
        """These annotated edges' pairs, targeting the truth's own successor or (with `successors`) ours instead.

        One walk serves both mislink halves: the source row is the same either way, and what distinguishes
        "the edge we chose" from "the edge we missed" is only which node the target is read from.
        """
        scores: list[PairScores] = []
        unscored = 0
        for source, target in edges.tolist():
            source_row = int(predicted_of_truth[source])
            target_row = int(predicted_of_truth[target]) if successors is None else int(successors[source_row][0])
            found = index.pair(int(node_ids[source_row]), int(node_ids[target_row]))
            if found is None:
                unscored += 1
            else:
                scores.append(found)
        return DispersionSample.of(scores, unscored)

    def separations(self) -> tuple[Separation, ...]:
        """Both mislink halves against the correct edges — the two rankings the question is decided on."""
        return (
            Separation(label="MISLINK: the wrong edge we chose", correct=self.correct, mislinked=self.chosen),
            Separation(label="MISLINK: the true edge we missed", correct=self.correct, mislinked=self.missed),
        )


@dataclass(frozen=True)
class SeedDispersionDiagnosis:
    """Running one movie under a given operating point and reading its per-fate seed disagreement."""

    device: str

    def of_movie(self, root: DataRoot, movie: str, config: TrackerConfig) -> SeedDispersion:
        """Track the movie, score the seeds separately over the SAME detections, and split the pairs by fate.

        The seeds are scored over `tracked.detections` rather than the finished graph because that is the node
        set the blended affinity the linker consumed was built on, so a pair's dispersion and its blended
        probability are read from one row space (`GapSlots`) instead of two that post-processing has moved
        apart. One extra affinity pass, no extra detection — the response cache already holds the detections.
        """
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, self.device, config)
        path, truth = proxy.paths[0], proxy.truths[0].graph
        tracked = tracker.run_scored(path.name, path)
        seed_scores = tracker.edge_scorer.seed_scores(path, tracked.detections, self.device)
        index = SeedScoreIndex.of(tracked.detections, seed_scores, tracked.affinity)
        matching = DistanceMatcher(spacing=proxy.spacing).match(tracked.graph, truth)
        return SeedDispersion.of(tracked.graph, truth, matching, index)


def main() -> None:
    """Report, per fate, how far the two independently-trained edge seeds disagree — and whether that separates."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Does the discarded ensemble dispersion know where we mislink?")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to measure")
    parser.add_argument("--threshold", type=float, help="detector threshold; default is the shipped one")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any tracker knob, exactly as `proxy_eval --set` does, so an arm's config can be measured",
    )
    args = parser.parse_args()

    shipped = TrackerConfig.shipped()
    config = shipped if args.threshold is None else replace(shipped, threshold=args.threshold)
    for assignment in args.overrides:
        config = ConfigOverride.apply(config, assignment)
    logger.info("movie=%s  threshold=%g  set=%s", args.movie, config.threshold, list(args.overrides))
    dispersion = SeedDispersionDiagnosis(device=args.device).of_movie(
        DataRoot.from_config(args.config), args.movie, config
    )
    lines = [
        *dispersion.correct.lines("CORRECT edges"),
        *dispersion.chosen.lines("MISLINK: the wrong edge we chose"),
        *dispersion.missed.lines("MISLINK: the true edge we missed"),
        *[line for separation in dispersion.separations() for line in separation.lines()],
    ]
    for line in lines:
        logger.info("%s", line)


if __name__ == "__main__":
    main()

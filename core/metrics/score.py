"""The competition score: one number per video, then one number per split.

Two things about the shape of this score drive what a good prediction looks like.

The node-count adjustment is the *only* penalty for predicting too much — the edge tally ignores links in
unannotated tissue entirely. Overshooting the estimated cell count by a fraction `f` costs `0.1 * f` of
the Jaccard, while the extra detections it buys are worth their full recall, so the useful operating
point sits at or a little above the estimate. Undershooting pays the same `0.1 * f` as a bonus but loses
roughly `f` of the Jaccard outright, because every unfound node takes its links with it as false
negatives. The bonus is a losing trade.

That losing trade is invisible LOCALLY, though: our proxy's annotation is sparse, so the trimmed cells cost
little Jaccard here while the bonus pays in full. Hence the second, clamped form of the same quantity
(`clamped_edge_jaccard` / `SplitScore.selection_score`) that checkpoint selection reads instead — the faithful
metric stays untouched, because it is what the leaderboard is compared against.

Across videos, the adjusted Jaccard is averaged by sample weight rather than pooled, so a video with
dense annotation counts for far more than a sparsely annotated one — and the two acquisitions in this
dataset differ by nearly an order of magnitude in annotation density.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.data.tracks import AnnotatedTracks, TrackGraph
from core.metrics.divisions import DivisionCounts, DivisionScoring
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher

ADJUSTMENT_ALPHA = 0.1
DIVISION_WEIGHT = 0.1


@dataclass(frozen=True)
class VideoMetrics:
    """One video's contribution to the score."""

    edges: EdgeCounts
    divisions: DivisionCounts
    predicted_nodes: int
    estimated_nodes: int

    @classmethod
    def of(cls, prediction: TrackGraph, truth: AnnotatedTracks, matcher: DistanceMatcher) -> "VideoMetrics":
        """Score one predicted graph against one video's annotation."""
        return cls(
            edges=EdgeCounts.of(prediction, truth.graph, matcher.match(prediction, truth.graph)),
            divisions=DivisionScoring.of(prediction, truth.graph, matcher).counts(),
            predicted_nodes=len(prediction.node_ids),
            estimated_nodes=truth.estimated_node_count,
        )

    def total_node_ratio(self) -> float:
        """How far the prediction's node count sits above the estimated true count, as a signed fraction."""
        if self.estimated_nodes <= 0:
            return float("nan")
        return (self.predicted_nodes - self.estimated_nodes) / self.estimated_nodes

    def adjusted_edge_jaccard(self) -> float:
        """The edge Jaccard after the node-count adjustment, floored at zero — the competition's own number."""
        return self._scaled_jaccard(self._node_count_factor())

    def clamped_edge_jaccard(self) -> float:
        """The same Jaccard with the node-count factor capped at its neutral 1.0 — the CHECKPOINT-SELECTION form.

        `adjusted_edge_jaccard` stays exactly the competition metric and is what gets compared to the
        leaderboard. It is not a safe thing to *select* on. Its factor `1 - 0.1 * ratio` exceeds 1 whenever the
        prediction undershoots the estimated node count, so a checkpoint that merely detects fewer cells is
        PAID for detecting fewer cells. That bonus is the one component of this pipeline measured to
        anti-transfer: a matched A/B differing only in the detection threshold (0.995 vs 0.99) moved +0.006 on
        the local proxy and -0.007 on the leaderboard. The mechanism is the annotation density. Our proxy is
        annotated SPARSELY, so the true cells a stricter threshold trims are mostly unannotated here — they
        cost no Jaccard while still collecting the bonus — whereas the hidden evaluation annotates the same
        movies DENSELY, where those very cells are annotated and their loss is paid in real Jaccard.

        Capping the factor at 1.0 keeps the overshoot penalty exactly as the competition applies it and makes
        undershooting earn nothing rather than earn a reward. 1.0 is the factor's own neutral point ("no
        adjustment"), so this introduces no constant to tune and nothing to sweep.
        """
        factor = self._node_count_factor()
        if math.isnan(factor):
            return float("nan")
        return self._scaled_jaccard(min(1.0, factor))

    def _node_count_factor(self) -> float:
        """The competition's multiplier on the Jaccard — above 1 exactly when the prediction undershoots."""
        return 1 - ADJUSTMENT_ALPHA * self.total_node_ratio()

    def _scaled_jaccard(self, factor: float) -> float:
        """The edge Jaccard under one node-count factor, floored at zero and undefined where either half is."""
        jaccard = self.edges.jaccard()
        if math.isnan(jaccard) or math.isnan(factor):
            return float("nan")
        return max(0.0, jaccard * factor)


@dataclass(frozen=True)
class SplitScore:
    """What the leaderboard reports for a set of videos — `score` — beside the selection form that clamps the bonus.

    `score` is the faithful competition metric and the only number to compare against the leaderboard.
    `selection_score` is the same aggregate over `VideoMetrics.clamped_edge_jaccard` (see there for why): it is
    what a checkpoint chooser should maximise, since the faithful metric pays for under-detection that the
    hidden, densely-annotated evaluation charges for.
    """

    adjusted_edge_jaccard: float
    clamped_edge_jaccard: float
    division_jaccard: float
    score: float
    selection_score: float

    @classmethod
    def of(cls, samples: Sequence[VideoMetrics]) -> "SplitScore":
        """Aggregate per-video metrics the way the competition does, and again with the node-count bonus clamped."""
        adjusted = cls._weighted_mean([(sample.edges.weight(), sample.adjusted_edge_jaccard()) for sample in samples])
        clamped = cls._weighted_mean([(sample.edges.weight(), sample.clamped_edge_jaccard()) for sample in samples])
        divisions = DivisionCounts(
            tp=sum(sample.divisions.tp for sample in samples),
            fp=sum(sample.divisions.fp for sample in samples),
            fn=sum(sample.divisions.fn for sample in samples),
        )
        division_jaccard = divisions.jaccard()
        return cls(
            adjusted_edge_jaccard=adjusted,
            clamped_edge_jaccard=clamped,
            division_jaccard=division_jaccard,
            score=cls._combined(adjusted, division_jaccard),
            selection_score=cls._combined(clamped, division_jaccard),
        )

    @staticmethod
    def _combined(edge_jaccard: float, division_jaccard: float) -> float:
        """The edge half plus the division term — or the edge half alone where no division exists to score."""
        if math.isnan(division_jaccard):
            return edge_jaccard
        return edge_jaccard + DIVISION_WEIGHT * division_jaccard

    @staticmethod
    def _weighted_mean(weighted: Sequence[tuple[int, float]]) -> float:
        """Average the values that exist, weighted by sample size."""
        usable = [(weight, value) for weight, value in weighted if not math.isnan(value)]
        total = sum(weight for weight, _ in usable)
        if total == 0:
            return float("nan")
        return sum(weight * value for weight, value in usable) / total

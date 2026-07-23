"""The competition score: one number per video, then one number per split.

Two things about the shape of this score drive what a good prediction looks like.

The node-count adjustment is the *only* penalty for predicting too much — the edge tally ignores links in
unannotated tissue entirely. Overshooting the estimated cell count by a fraction `f` costs `0.1 * f` of
the Jaccard, while the extra detections it buys are worth their full recall, so the useful operating
point sits at or a little above the estimate. Undershooting pays the same `0.1 * f` as a bonus but loses
roughly `f` of the Jaccard outright, because every unfound node takes its links with it as false
negatives. The bonus is a losing trade.

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
        """The edge Jaccard after the node-count penalty, floored at zero."""
        jaccard, ratio = self.edges.jaccard(), self.total_node_ratio()
        if math.isnan(jaccard) or math.isnan(ratio):
            return float("nan")
        return max(0.0, jaccard * (1 - ADJUSTMENT_ALPHA * ratio))


@dataclass(frozen=True)
class SplitScore:
    """What the leaderboard reports for a set of videos."""

    adjusted_edge_jaccard: float
    division_jaccard: float
    score: float

    @classmethod
    def of(cls, samples: Sequence[VideoMetrics]) -> "SplitScore":
        """Aggregate per-video metrics the way the competition does."""
        adjusted = cls._weighted_mean([(sample.edges.weight(), sample.adjusted_edge_jaccard()) for sample in samples])
        divisions = DivisionCounts(
            tp=sum(sample.divisions.tp for sample in samples),
            fp=sum(sample.divisions.fp for sample in samples),
            fn=sum(sample.divisions.fn for sample in samples),
        )
        division_jaccard = divisions.jaccard()
        scored = adjusted + DIVISION_WEIGHT * division_jaccard
        combined = adjusted if math.isnan(division_jaccard) else scored
        return cls(adjusted_edge_jaccard=adjusted, division_jaccard=division_jaccard, score=combined)

    @staticmethod
    def _weighted_mean(weighted: Sequence[tuple[int, float]]) -> float:
        """Average the values that exist, weighted by sample size."""
        usable = [(weight, value) for weight, value in weighted if not math.isnan(value)]
        total = sum(weight for weight, _ in usable)
        if total == 0:
            return float("nan")
        return sum(weight * value for weight, value in usable) / total

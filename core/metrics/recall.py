"""Detection recall — the share of annotated cells a detector actually finds.

This is the number the detection task is judged on: of the ground-truth nodes, how many have a predicted
centre within the competition's 7 um matching radius. It reuses the same per-timepoint distance matching
the edge metric uses, so a recall figure here is measured on exactly the terms the leaderboard cares about.
Recall is compared between detectors at a *matched node count* — both asked for the same number of centres
— so the comparison isolates where the detector looks, not how much it predicts.
"""

from dataclasses import dataclass

from core.data.tracks import TrackGraph
from core.metrics.matching import UNMATCHED, DistanceMatcher


@dataclass(frozen=True)
class DetectionRecall:
    """How many annotated cells were found, and out of how many."""

    found: int
    total: int

    @classmethod
    def of(cls, detections: TrackGraph, truth: TrackGraph, matcher: DistanceMatcher) -> "DetectionRecall":
        """Match detections to the annotation and count the ground-truth cells that got one."""
        matched_gt = matcher.match(detections, truth).gt_rows
        found = len({int(row) for row in matched_gt if row != UNMATCHED})
        return cls(found=found, total=len(truth.node_ids))

    def fraction(self) -> float:
        """The recall — found over total, or NaN when there is nothing to find."""
        return self.found / self.total if self.total > 0 else float("nan")

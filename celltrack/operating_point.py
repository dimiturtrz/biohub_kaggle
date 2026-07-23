"""Sweeping the detector's node budget against the realised competition score.

The metric's node-count term is signed and only clamped from below, so predicting fewer nodes than the
estimated true count multiplies the edge Jaccard above 1 while predicting more costs 0.1 per unit of
excess — but under-detecting also drops edges as false negatives. Where those two forces balance is an
empirical operating point, not something the formula alone settles. This sweep runs the trained detector
at a range of budgets (as a fraction of each video's estimated cell count), links each with the same
nearest-neighbour linker, and scores the result with the local evaluator — per acquisition, since the two
prefixes differ by an order of magnitude in density and the node-count term bites hardest where annotation
is densest.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from celltrack.bracket import ValidationFold
from celltrack.detection import CELL_SCALE_UM
from celltrack.inference import LearnedDetector
from celltrack.linking import Linker, NearestNeighbourLinker
from celltrack.peaks import PeakExtractor
from celltrack.training.model import Checkpoint, DetectionUNet
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[1] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
_WEIGHTS = "detector.pt"
_LINK_GATE_UM = 15.0
_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepPoint:
    """One video scored at one node budget."""

    ratio: float
    prefix: str
    metrics: VideoMetrics


@dataclass(frozen=True)
class OperatingPointSweep:
    """Runs the detector at several node budgets and scores each end to end."""

    detector: LearnedDetector
    linker: Linker
    matcher: DistanceMatcher

    @classmethod
    def from_checkpoint(
        cls, spacing: Spacing, checkpoint: Checkpoint, gate_um: float, device: str
    ) -> "OperatingPointSweep":
        """Build the learned detector from a checkpoint and pair it with the nearest-neighbour linker."""
        model = DetectionUNet.from_checkpoint(checkpoint)
        return cls(
            detector=LearnedDetector(
                model=model.to(device).eval(),
                peaks=PeakExtractor(spacing=spacing, scale_um=CELL_SCALE_UM),
                window=int(checkpoint["window"]),
                device=device,
            ),
            linker=NearestNeighbourLinker(spacing=spacing, max_distance_um=gate_um),
            matcher=DistanceMatcher(spacing=spacing),
        )

    def at(self, ratio: float, prefix: str, video: CellVideo, truth: AnnotatedTracks) -> SweepPoint:
        """Detect at `ratio` of the estimated count, link, and score this video."""
        keep = max(1, round(ratio * truth.estimated_node_count))
        linked = self.linker.link(self.detector.detect(video, keep))
        return SweepPoint(ratio=ratio, prefix=prefix, metrics=VideoMetrics.of(linked, truth, self.matcher))

    @staticmethod
    def report(points: list[SweepPoint]) -> None:
        """Log the score at each budget per acquisition, and the budget that maximises it."""
        for prefix in sorted({point.prefix for point in points}):
            logger.info("acquisition %s:", prefix)
            best = (0.0, float("-inf"))
            for ratio in sorted({point.ratio for point in points}):
                samples = [p.metrics for p in points if p.prefix == prefix and p.ratio == ratio]
                score = SplitScore.of(samples).score
                logger.info("  N_pred/N_total=%.2f -> score %.4f", ratio, score)
                best = max(best, (score, ratio), key=lambda pair: pair[0])
            logger.info("  best: score %.4f at N_pred/N_total=%.2f", best[0], best[1])


def main() -> None:
    """Sweep the detector's node budget on the held-out fold and log the score curve per acquisition."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Sweep detector node budget against realised score.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--per-prefix", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    arguments = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    fold = ValidationFold.load(root, arguments.fold).stratified_subset(arguments.per_prefix)
    checkpoint = cast(Checkpoint, torch.load(root.processed(_DATASET) / _WEIGHTS, map_location=arguments.device))
    sweep = OperatingPointSweep.from_checkpoint(fold.spacing, checkpoint, _LINK_GATE_UM, arguments.device)

    points = [
        sweep.at(ratio, AcquisitionFolds.prefix_of(path), video, truth)
        for ratio in _RATIOS
        for path, (video, truth) in zip(fold.videos, fold.images(), strict=True)
    ]
    sweep.report(points)


if __name__ == "__main__":
    main()

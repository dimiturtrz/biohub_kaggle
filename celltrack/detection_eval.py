"""Comparing the learned detector's recall against the classical floor, at a matched node count.

This is bd 3ws's verdict: on the held-out prefix split, does the trained network find more of the
annotated cells than the classical Laplacian-of-Gaussian when both are asked for the same number of
centres? Recall is measured with the competition's own 7 um matching, per acquisition, because the two
prefixes differ by an order of magnitude in density and a pooled number would hide which one improved.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from celltrack.bracket import ValidationFold
from celltrack.detection import CELL_SCALE_UM, HIGH, LOW, BlobDetector
from celltrack.inference import LearnedDetector
from celltrack.peaks import PeakExtractor
from celltrack.training.model import Checkpoint, DetectionUNet
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher
from core.metrics.recall import DetectionRecall
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[1] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
_WEIGHTS = "detector.pt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecallComparison:
    """One video's recall under each detector at the same node budget."""

    prefix: str
    classical: DetectionRecall
    learned: DetectionRecall


@dataclass(frozen=True)
class DetectionComparison:
    """Runs both detectors at a matched node count and reports recall per acquisition."""

    classical: BlobDetector
    learned: LearnedDetector
    matcher: DistanceMatcher

    @classmethod
    def from_checkpoint(cls, spacing: Spacing, checkpoint: Checkpoint, device: str) -> "DetectionComparison":
        """Build the classical and checkpoint-loaded learned detectors, sharing the anisotropic peak-picker."""
        model = DetectionUNet.from_checkpoint(checkpoint)
        peaks = PeakExtractor(spacing=spacing, scale_um=CELL_SCALE_UM)
        return cls(
            classical=BlobDetector(spacing=spacing),
            learned=LearnedDetector(
                model=model.to(device).eval(), peaks=peaks, window=int(checkpoint["window"]), device=device
            ),
            matcher=DistanceMatcher(spacing=spacing),
        )

    def compare(self, prefix: str, video: CellVideo, truth: AnnotatedTracks) -> RecallComparison:
        """Both detectors on one video at its estimated node count, each scored for recall."""
        keep = truth.estimated_node_count
        classical = self.classical.detect(video.normalised_frames(LOW, HIGH), video.timepoint_count, keep)
        learned = self.learned.detect(video, keep)
        return RecallComparison(
            prefix=prefix,
            classical=DetectionRecall.of(classical, truth.graph, self.matcher),
            learned=DetectionRecall.of(learned, truth.graph, self.matcher),
        )

    @staticmethod
    def report(comparisons: list[RecallComparison]) -> None:
        """Log per-prefix recall and whether the learned detector beat the classical floor per acquisition."""
        for prefix in sorted({comparison.prefix for comparison in comparisons}):
            rows = [comparison for comparison in comparisons if comparison.prefix == prefix]
            classical = sum(row.classical.found for row in rows) / sum(row.classical.total for row in rows)
            learned = sum(row.learned.found for row in rows) / sum(row.learned.total for row in rows)
            verdict = "learned wins" if learned > classical else "classical holds"
            logger.info("%s: classical recall %.3f, learned recall %.3f — %s", prefix, classical, learned, verdict)


def main() -> None:
    """Score classical and learned detection recall on the held-out fold and log the per-prefix comparison."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Compare learned vs classical detection recall.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--per-prefix", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    arguments = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    fold = ValidationFold.load(root, arguments.fold).stratified_subset(arguments.per_prefix)
    checkpoint = cast(Checkpoint, torch.load(root.processed(_DATASET) / _WEIGHTS, map_location=arguments.device))
    comparison = DetectionComparison.from_checkpoint(fold.spacing, checkpoint, arguments.device)

    results = [
        comparison.compare(AcquisitionFolds.prefix_of(path), video, truth)
        for path, (video, truth) in zip(fold.videos, fold.images(), strict=True)
    ]
    comparison.report(results)


if __name__ == "__main__":
    main()

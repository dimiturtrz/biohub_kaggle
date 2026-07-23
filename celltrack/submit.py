"""Predicting the test videos and writing a submission CSV.

The classical baseline taken end to end on the held-out test split: detect, link, and flatten to the
one-table format the competition scores. Its purpose is not a good score — it is to prove that the whole
path is sound (the CSV convention, the per-video keying, and our reading of the hidden split) by having
the leaderboard return a real number rather than an error.

Test videos ship no annotation, so the detector's node budget cannot come from a per-video estimate the
way the local floor's does. It comes instead from the training split: the mean estimated cell count of
each acquisition, applied to the test videos of the same acquisition. That is the assumption-free stand-in
for the missing per-video count — a real operating point still has to be chosen without it, which is the
sweep task.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from celltrack.detection import CELL_SCALE_UM, HIGH, LOW, BlobDetector
from celltrack.linking import Linker, NearestNeighbourLinker
from core.data.split import AcquisitionFolds
from core.data.submission import Submission
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[1] / "paths.yaml"
_LINK_GATE_UM = 15.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeBudget:
    """How many nodes to detect per test video, keyed by acquisition — derived from the training split."""

    per_prefix: dict[str, int]

    @classmethod
    def from_training(cls, root: DataRoot) -> "NodeBudget":
        """The mean estimated cell count of each acquisition across the training videos."""
        totals: dict[str, list[int]] = {}
        for video in root.videos("train"):
            estimate = AnnotatedTracks.from_geff(root.track_store(video)).estimated_node_count
            totals.setdefault(AcquisitionFolds.prefix_of(video), []).append(estimate)
        return cls(per_prefix={prefix: round(sum(counts) / len(counts)) for prefix, counts in totals.items()})

    def for_video(self, video: Path) -> int:
        """The node budget for one test video, from its acquisition prefix."""
        return self.per_prefix[AcquisitionFolds.prefix_of(video)]


@dataclass(frozen=True)
class TestPredictor:
    """Detects and links a test video into a predicted track graph."""

    detector: BlobDetector
    linker: Linker
    budget: NodeBudget

    @classmethod
    def classical(cls, root: DataRoot, scale_um: float, gate_um: float) -> "TestPredictor":
        """The classical baseline: LoG detection, NN linking, budget from the training split."""
        spacing = CellVideo.from_ome_zarr(root.videos("test")[0]).spacing
        return cls(
            detector=BlobDetector(spacing=spacing, scale_um=scale_um),
            linker=NearestNeighbourLinker(spacing=spacing, max_distance_um=gate_um),
            budget=NodeBudget.from_training(root),
        )

    def predict(self, video: Path) -> TrackGraph:
        """Detect to this video's budget and link, returning its predicted track graph."""
        cell = CellVideo.from_ome_zarr(video)
        frames = cell.normalised_frames(LOW, HIGH)
        detections = self.detector.detect(frames, cell.timepoint_count, self.budget.for_video(video))
        return self.linker.link(detections)

    def submission(self, root: DataRoot) -> Submission:
        """Predict every test video, keyed by its dataset name for the submission table."""
        return Submission(graphs={video.stem: self.predict(video) for video in root.videos("test")})


def main() -> None:
    """Predict the test split and write a submission CSV."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Predict the test videos and write a submission CSV.")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--gate-um", type=float, default=_LINK_GATE_UM)
    parser.add_argument("--scale-um", type=float, default=CELL_SCALE_UM)
    arguments = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    predictor = TestPredictor.classical(root, arguments.scale_um, arguments.gate_um)
    logger.info("node budget per prefix: %s", predictor.budget.per_prefix)
    submission = predictor.submission(root)
    written = submission.write_csv(arguments.out)
    frame = submission.to_frame()
    logger.info("wrote %s: %d rows across %d videos", written, len(frame), len(submission.graphs))


if __name__ == "__main__":
    main()

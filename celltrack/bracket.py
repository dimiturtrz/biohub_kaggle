"""Bracketing the score before any model is trained.

Two runs bound the problem. The CEILING links the ground-truth nodes themselves, so there is no
detection error at all and whatever the linker scores is the most a nearest-neighbour linker can reach.
The FLOOR runs a classical blob detector and then the same linker end to end. The gap between them is the
budget a learned detector has to win, and which end of it is loose decides whether effort goes into
detection or into linking.

One honesty note the metric forces: the ceiling links only the ~6 % of cells that are annotated, so its
predicted-node count sits far below the estimated true count and the node-count adjustment hands it a
large, unrepresentative bonus. The linking quality is therefore read off the unadjusted **edge Jaccard**.
The floor detects to the estimated count, so there its adjusted score is meaningful and reported too.
"""

import argparse
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from celltrack.detection import CELL_SCALE_UM, HIGH, LOW, BlobDetector
from celltrack.division_recovery import DivisionRecovery
from celltrack.linking import Linker, NearestNeighbourLinker
from celltrack.motion_linking import MotionHungarianLinker
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[1] / "paths.yaml"
# Swept on validation fold 0: the edge Jaccard peaks near 15 um (0.998), above the 7 um matching cutoff
# because a true successor may move further than the matcher tolerates, and below the point where a wrong
# cell in a dense frame becomes the cheaper partner. See interpretations/tracking/2026-07-23_bracket.md.
_LINK_GATE_UM = 15.0
_DEFAULT_FOLD = 0
# The shipping kernel's motion gates, so a ceiling run measures the linker we actually submit.
_TIGHT_GATE_UM, _LOOSE_GATE_UM = 6.0, 10.0
# Post-link division gates. The parent gate is the frontier's 10.5 um, which a fold-0 sweep independently
# picks out as the optimum (0.4468 division Jaccard, falling to 0.2558 by 14 um as false forks cost the edge
# term too). Their 8.0 um sister gate is far too tight for this data — 0.2593 — and the measured plateau
# starts at 18 um, where the gate is close to inert anyway: two daughters each within 10.5 um of one mother
# can be at most 21 um apart. Swept on ground-truth nodes, so it wants re-tuning against real detections,
# where unparented false positives are far more common than they are here.
_PARENT_GATE_UM, _SISTER_GATE_UM = 10.5, 18.0
_MAX_DIVISION_FRACTION = 0.02

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmResult:
    """What one bracket arm scored, kept as per-video metrics so the pooled numbers can be derived."""

    label: str
    metrics: tuple[VideoMetrics, ...]

    def edge_jaccard(self) -> float:
        """Linking quality: the micro-averaged edge Jaccard, unadjusted by the node count."""
        return EdgeCounts.pooled(metric.edges for metric in self.metrics).jaccard()

    def score(self) -> SplitScore:
        """The full competition score for this arm — meaningful only when the node count is realistic."""
        return SplitScore.of(self.metrics)

    def predicted_nodes(self) -> int:
        """How many nodes the arm predicted in total, against which the node-count penalty is measured."""
        return sum(metric.predicted_nodes for metric in self.metrics)

    def estimated_nodes(self) -> int:
        """The estimated true cell count the arm is measured against."""
        return sum(metric.estimated_nodes for metric in self.metrics)

    def report(self) -> str:
        """A one-line summary of what this arm reached.

        The division Jaccard is reported alongside the edge Jaccard because a one-to-one linker scores
        exactly zero on it by construction, and a summary that omits it hides that whole term.
        """
        scored = self.score()
        return (
            f"{self.label}: edge_jaccard={self.edge_jaccard():.4f} division_jaccard={scored.division_jaccard:.4f} "
            f"score={scored.score:.4f} nodes predicted={self.predicted_nodes()} vs estimated={self.estimated_nodes()}"
        )


@dataclass(frozen=True)
class Ceiling:
    """Links the ground-truth nodes directly — the best a linker could do given perfect detection."""

    linker: Linker
    matcher: DistanceMatcher
    label: str = "ceiling"
    recovery: DivisionRecovery | None = None

    @classmethod
    def with_gate(cls, spacing: Spacing, gate_um: float) -> "Ceiling":
        """A ceiling whose linker reaches out to `gate_um` and whose matcher uses the competition cutoff."""
        return cls(
            linker=NearestNeighbourLinker(spacing=spacing, max_distance_um=gate_um),
            matcher=DistanceMatcher(spacing=spacing),
        )

    @classmethod
    def with_motion(cls, spacing: Spacing, recovery: DivisionRecovery | None) -> "Ceiling":
        """A ceiling running the shipping motion linker, optionally followed by the division-recovery pass."""
        return cls(
            linker=MotionHungarianLinker(spacing=spacing, tight_gate_um=_TIGHT_GATE_UM, loose_gate_um=_LOOSE_GATE_UM),
            matcher=DistanceMatcher(spacing=spacing),
            label="ceiling-motiondiv" if recovery else "ceiling-motion",
            recovery=recovery,
        )

    def over(self, annotations: Iterable[AnnotatedTracks]) -> ArmResult:
        """Score the linker on the annotations of every video."""
        metrics = [VideoMetrics.of(self._predict(truth), truth, self.matcher) for truth in annotations]
        return ArmResult(label=self.label, metrics=tuple(metrics))

    def _predict(self, truth: AnnotatedTracks) -> TrackGraph:
        """Link this video's ground-truth nodes, recovering divisions when the arm asks for them."""
        linked = self.linker.link(truth.graph.without_edges())
        return self.recovery.transform(linked) if self.recovery else linked


@dataclass(frozen=True)
class Floor:
    """Runs the classical detector and then the same linker, end to end."""

    detector: BlobDetector
    linker: Linker
    matcher: DistanceMatcher

    @classmethod
    def with_scale_and_gate(cls, spacing: Spacing, scale_um: float, gate_um: float) -> "Floor":
        """A floor detecting at a physical cell scale and linking within a physical gate."""
        return cls(
            detector=BlobDetector(spacing=spacing, scale_um=scale_um),
            linker=NearestNeighbourLinker(spacing=spacing, max_distance_um=gate_um),
            matcher=DistanceMatcher(spacing=spacing),
        )

    def over(self, videos: Iterable[tuple[CellVideo, AnnotatedTracks]]) -> ArmResult:
        """Detect to each video's estimated cell count, link, and score end to end."""
        metrics = [self._score_one(cell, truth) for cell, truth in videos]
        return ArmResult(label="floor", metrics=tuple(metrics))

    def _score_one(self, cell: CellVideo, truth: AnnotatedTracks) -> VideoMetrics:
        """Detect, link and score a single video."""
        frames = cell.normalised_frames(LOW, HIGH)
        detections = self.detector.detect(frames, cell.timepoint_count, truth.estimated_node_count)
        return VideoMetrics.of(self.linker.link(detections), truth, self.matcher)


@dataclass(frozen=True)
class ValidationFold:
    """The held-out videos of one fold, so a bracket run reads its data from a single place."""

    spacing: Spacing
    videos: tuple[Path, ...]
    annotations: tuple[AnnotatedTracks, ...]

    @classmethod
    def load(cls, root: DataRoot, fold: int) -> "ValidationFold":
        """Read the fold's held-out video paths and annotations, spacing from the first store."""
        held_out = AcquisitionFolds().split(root.videos("train"), fold).validation
        return cls(
            spacing=CellVideo.from_ome_zarr(held_out[0]).spacing,
            videos=held_out,
            annotations=tuple(AnnotatedTracks.from_geff(root.track_store(video)) for video in held_out),
        )

    def stratified_subset(self, per_prefix: int) -> "ValidationFold":
        """A subset with the first few videos of each acquisition kept — for the expensive floor run."""
        keep = {video for group in AcquisitionFolds.by_prefix(self.videos).values() for video in group[:per_prefix]}
        chosen = [index for index, video in enumerate(self.videos) if video in keep]
        return ValidationFold(
            spacing=self.spacing,
            videos=tuple(self.videos[index] for index in chosen),
            annotations=tuple(self.annotations[index] for index in chosen),
        )

    def images(self) -> list[tuple[CellVideo, AnnotatedTracks]]:
        """Open each video's image alongside its annotation — what the floor needs."""
        return [
            (CellVideo.from_ome_zarr(video), truth) for video, truth in zip(self.videos, self.annotations, strict=True)
        ]


def main() -> None:
    """Run a bracket arm on a validation fold and log what it reached."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Bracket the competition score with simple baselines.")
    parser.add_argument("--arm", choices=("ceiling", "ceiling-motion", "ceiling-motiondiv", "floor"), default="ceiling")
    parser.add_argument("--fold", type=int, default=_DEFAULT_FOLD)
    parser.add_argument("--gate-um", type=float, default=_LINK_GATE_UM)
    parser.add_argument("--scale-um", type=float, default=CELL_SCALE_UM)
    parser.add_argument("--floor-per-prefix", type=int, default=3)
    parser.add_argument("--parent-gate-um", type=float, default=_PARENT_GATE_UM)
    parser.add_argument("--sister-gate-um", type=float, default=_SISTER_GATE_UM)
    parser.add_argument("--division-fraction", type=float, default=_MAX_DIVISION_FRACTION)
    arguments = parser.parse_args()

    fold = ValidationFold.load(DataRoot.from_config(_CONFIG), arguments.fold)
    logger.info("fold %d: %d validation videos", arguments.fold, len(fold.videos))

    if arguments.arm == "floor":
        sample = fold.stratified_subset(arguments.floor_per_prefix)
        logger.info("floor sample: %d videos", len(sample.videos))
        floor = Floor.with_scale_and_gate(fold.spacing, arguments.scale_um, arguments.gate_um)
        logger.info("%s", floor.over(sample.images()).report())
    elif arguments.arm == "ceiling":
        logger.info("%s", Ceiling.with_gate(fold.spacing, arguments.gate_um).over(fold.annotations).report())
    else:
        recovery = (
            DivisionRecovery(
                spacing=fold.spacing,
                parent_gate_um=arguments.parent_gate_um,
                sister_gate_um=arguments.sister_gate_um,
                max_added_fraction=arguments.division_fraction,
            )
            if arguments.arm == "ceiling-motiondiv"
            else None
        )
        logger.info("%s", Ceiling.with_motion(fold.spacing, recovery).over(fold.annotations).report())


if __name__ == "__main__":
    main()

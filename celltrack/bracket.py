"""Bracketing the score before any model is trained.

Two runs bound the problem. The CEILING links the ground-truth nodes themselves, so there is no
detection error at all and whatever the linker scores is the most a nearest-neighbour linker can reach.
The FLOOR runs a classical blob detector and then the same linker end to end. The gap between them is the
budget a learned detector has to win, and which end of it is loose decides whether effort goes into
detection or into linking.

One honesty note the metric forces: the ceiling links only the ~6 % of cells that are annotated, so its
predicted-node count sits far below the estimated true count and the node-count adjustment hands it a
large, unrepresentative bonus. The linking quality is therefore read off the unadjusted **edge Jaccard**,
not the adjusted score — the adjustment only means something once detection produces a realistic node
count, which is the floor's job.
"""

import argparse
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from celltrack.linking import Linker, NearestNeighbourLinker
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[1] / "paths.yaml"
# Swept on validation fold 0: the edge Jaccard peaks near 15 um (0.998), above the 7 um matching cutoff
# because a true successor may move further than the matcher tolerates, and below the point where a wrong
# cell in a dense frame becomes the cheaper partner. See interpretations/tracking/2026-07-23_bracket.md.
_CEILING_GATE_UM = 15.0
_DEFAULT_FOLD = 0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BracketResult:
    """One bracket run over a set of videos, kept as counts so several can be pooled."""

    label: str
    edges: EdgeCounts
    predicted_nodes: int
    annotated_nodes: int

    def edge_jaccard(self) -> float:
        """The linking quality — TP/(TP+FP+FN) over edges, unadjusted by the node count."""
        return self.edges.jaccard()

    def report(self) -> str:
        """A one-line summary of what this arm reached."""
        return (
            f"{self.label}: edge_jaccard={self.edge_jaccard():.4f} "
            f"(tp={self.edges.tp} fp={self.edges.fp} fn={self.edges.fn}), "
            f"nodes predicted={self.predicted_nodes} vs annotated={self.annotated_nodes}"
        )


@dataclass(frozen=True)
class Ceiling:
    """Links the ground-truth nodes directly, measuring the best a linker could do given perfect detection."""

    linker: Linker
    matcher: DistanceMatcher

    @classmethod
    def with_gate(cls, spacing: Spacing, gate_um: float) -> "Ceiling":
        """A ceiling whose linker reaches out to `gate_um` and whose matcher uses the competition cutoff."""
        return cls(
            linker=NearestNeighbourLinker(spacing=spacing, max_distance_um=gate_um),
            matcher=DistanceMatcher(spacing=spacing),
        )

    def over(self, videos: Iterable[AnnotatedTracks]) -> BracketResult:
        """Score the linker on the annotations of every video, pooling the edge counts."""
        counts: list[EdgeCounts] = []
        predicted = annotated = 0
        for truth in videos:
            detections = TrackGraph(
                node_ids=truth.graph.node_ids,
                coordinates=truth.graph.coordinates,
                edges=truth.graph.edges[:0],
            )
            linked = self.linker.link(detections)
            counts.append(EdgeCounts.of(linked, truth.graph, self.matcher.match(linked, truth.graph)))
            predicted += len(linked.node_ids)
            annotated += len(truth.graph.node_ids)
        return BracketResult(
            label="ceiling", edges=EdgeCounts.pooled(counts), predicted_nodes=predicted, annotated_nodes=annotated
        )


@dataclass(frozen=True)
class ValidationFold:
    """The held-out videos of one fold, loaded once so a bracket run reads its data from a single place."""

    spacing: Spacing
    annotations: tuple[AnnotatedTracks, ...]

    @classmethod
    def load(cls, root: DataRoot, fold: int) -> "ValidationFold":
        """Read the fold's held-out videos and their annotations off disk, spacing from the first store."""
        held_out = AcquisitionFolds().split(root.videos("train"), fold).validation
        return cls(
            spacing=CellVideo.from_ome_zarr(held_out[0]).spacing,
            annotations=tuple(AnnotatedTracks.from_geff(root.track_store(video)) for video in held_out),
        )


def main() -> None:
    """Run the ceiling on a validation fold and log its linking quality."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Bracket the competition score with simple baselines.")
    parser.add_argument("--fold", type=int, default=_DEFAULT_FOLD)
    parser.add_argument("--gate-um", type=float, default=_CEILING_GATE_UM)
    arguments = parser.parse_args()

    fold = ValidationFold.load(DataRoot.from_config(_CONFIG), arguments.fold)
    logger.info("fold %d: %d validation videos", arguments.fold, len(fold.annotations))

    ceiling = Ceiling.with_gate(fold.spacing, arguments.gate_um).over(fold.annotations)
    logger.info("%s", ceiling.report())


if __name__ == "__main__":
    main()

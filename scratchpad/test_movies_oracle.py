"""Per-movie linker ceiling on the 4 test movies — is the dense-movie loss detector or linker?

Links the GT nodes directly (perfect detection) on each test movie and reports the edge Jaccard ceiling,
next to the real-detection score. If 6bba_05db0fb1's ceiling is high but its real score is ~0.82, the gap
is real-detection crowding the linker (a linker-robustness problem), not missed/false detection.
"""

import logging
from pathlib import Path

from celltrack.linkers import LinkerConfig
from celltrack.motion_linking import MotionHungarianLinker
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.paths import DataRoot

_CONFIG = Path("D:/personal_projects/biohub_kaggle/paths.yaml")
_TEST = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
_REAL = {"44b6_0113de3b": 0.912, "44b6_0b24845f": 0.920, "6bba_05b6850b": 0.998, "6bba_05db0fb1": 0.821}
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = DataRoot.from_config(_CONFIG)
    by_stem = {p.stem: p for p in root.videos("train")}
    paths = [by_stem[s] for s in _TEST]
    truths = [AnnotatedTracks.from_geff(root.track_store(p)) for p in paths]
    spacing = CellVideo.from_ome_zarr(paths[0]).spacing
    matcher = DistanceMatcher(spacing=spacing)
    linker = LinkerConfig(name="motion", tight_um=6.0, gate_um=8.5).build(spacing)

    for stem, truth in zip(_TEST, truths, strict=True):
        linked = linker.link(truth.graph.without_edges())
        counts = EdgeCounts.of(linked, truth.graph, matcher.match(linked, truth.graph))
        logger.info(
            "%-14s GT-node linker ceiling edge_jac=%.4f (%d annotated nodes) | real detections=%.3f",
            stem, counts.jaccard(), len(truth.graph.node_ids), _REAL[stem],
        )


if __name__ == "__main__":
    main()

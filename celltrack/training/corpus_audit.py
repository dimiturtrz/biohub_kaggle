"""CLI: how crowded each training corpus's candidate sets are, in the linker's own micrometres.

The case for training on the fully-labelled synthetic corpus is that its candidate sets are COMPLETE — the
crowded near-neighbours are true negatives in the edge matrix rather than unannotated detections outside it.
That case dies if synthetic frames are simply emptier than the dense movie: no crowding, nothing to learn.
This is the gating measurement, and it runs before any arm does (`celltrack.data.candidate_density` owns the
arithmetic; this owns the mount).

Three populations, one quantity — mean candidates inside `LinkerConfig.gate_um` of a source:
the synthetic ground truth (complete), the competition ground truth (what training sees today), and the
shipped detector's own detections on a movie (what the tracker actually chooses among, and therefore where
the mislinks live).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from celltrack.data.candidate_density import CandidateDensity
from celltrack.data.detection_pairs import DetectionPairs
from celltrack.data.frame_source import ZarrFrames
from celltrack.data.synthetic_pairs import SyntheticPairs
from celltrack.eval.dense_diagnosis import DENSE_MOVIE
from celltrack.eval.proxy import TestMovieProxy
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from celltrack.training.joint_config import WARM_PACKS
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_DATASET = "biohub_cell_tracking"
_SYNTHETIC_DATASET = "biohub_synthetic"
_SYNTHETIC_SEQUENCES = "sequences"
# The intensity window a competition video is read through — the same quantiles the trainer's own reader uses.
_Q_LOW, _Q_HIGH = 0.001, 0.999


def main() -> None:
    """Measure and log the three populations' in-gate candidate counts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Compare candidate-set crowding across the training corpora.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE, help="the real movie the detected population is read off")
    parser.add_argument("--sequences", type=int, default=64, help="synthetic sequences to measure")
    args = parser.parse_args()

    config = TrackerConfig.shipped()
    gate = config.linker.gate_um
    root = DataRoot.from_config(args.config)
    logger.info("gate = %.1f um (the shipped linker's own)", gate)

    sequences = SyntheticPairs.sequences(root.synthetic(_SYNTHETIC_DATASET) / _SYNTHETIC_SEQUENCES, args.sequences)
    graphs = [SyntheticPairs.graph(path) for path in sequences]
    CandidateDensity.of(graphs, CandidateDensity.synthetic_spacing(sequences[0]), gate).report("synthetic GT")

    proxy = TestMovieProxy.load(root, (args.movie,))
    video_path = proxy.paths[0]
    spacing = CellVideo.from_ome_zarr(video_path).spacing
    truth: TrackGraph = proxy.truths[0].graph
    CandidateDensity.of([truth], spacing, gate).report(f"real GT ({args.movie})")

    proc = root.processed(_DATASET)
    tracker = CellTracker.ephemeral(proc / WARM_PACKS["seed1"], proc / WARM_PACKS["seed2"], args.device, config)
    detected = tracker.detector.nodes(args.movie, video_path, config.threshold, None)
    CandidateDensity.of([detected], spacing, gate).report(f"real detected ({args.movie})")

    # The assembled CORPUS, not a population: DetectionPairs supervises only the rows the annotation can
    # adjudicate, and a matched detection whose successor was also detected could plausibly be an EASIER cell.
    # If that selection re-introduced the uncontested population the corpus exists to escape, this row would
    # read like the annotated one and no training result from it would mean anything.
    pairs = DetectionPairs.of(detected, truth, ZarrFrames(video_path, _Q_LOW, _Q_HIGH), spacing, gate)
    CandidateDensity.of_pairs(pairs, spacing, gate).report(f"detection pairs ({args.movie})")
    rows = sum(len(pair.source_centres) for pair in pairs)
    logger.info("detection pairs: %d gaps, %d supervised source rows", len(pairs), rows)


if __name__ == "__main__":
    main()

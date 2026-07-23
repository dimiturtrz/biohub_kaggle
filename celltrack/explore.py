"""Dataset survey — the statistics that decide method choices, as a committed command.

Answers the open questions in `docs/PLAN.md`: how much of the detection signal is supervised, whether
the 7 um matching cutoff is tight against the real cell spacing, how many divisions there are to win
the 0.1 division term, and whether the two acquisition prefixes are different enough that the CV split
must be grouped by them.

    python -m celltrack.explore --split train --limit 20
"""

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_MATCH_CUTOFF_UM = 7.0
_PREFIX_LENGTH = 4
_MOTION_TAIL_QUANTILE = 0.99


@dataclass(frozen=True)
class VideoSummary:
    """What one annotated video looks like, in the units the metric cares about."""

    dataset: str
    prefix: str
    annotated_nodes: int
    estimated_nodes: int
    annotated_fraction: float
    divisions: int
    median_motion_um: float
    tail_motion_um: float
    beyond_cutoff_share: float


class DatasetSurvey:
    """Walks a split and reduces every video to a `VideoSummary`."""

    def __init__(self, root: DataRoot) -> None:
        self._root = root

    def summarise(self, video: Path) -> VideoSummary:
        """Summarise one video from its image metadata and its GEFF annotation."""
        opened = CellVideo.from_ome_zarr(video)
        tracks = AnnotatedTracks.from_geff(self._root.track_store(video))
        graph = tracks.graph
        motion = graph.link_displacements(opened.spacing)
        return VideoSummary(
            dataset=video.stem,
            prefix=video.stem[:_PREFIX_LENGTH],
            annotated_nodes=len(graph.node_ids),
            estimated_nodes=tracks.estimated_node_count,
            annotated_fraction=tracks.annotated_fraction,
            divisions=len(graph.division_parents()),
            median_motion_um=float(np.median(motion)),
            tail_motion_um=float(np.quantile(motion, _MOTION_TAIL_QUANTILE)),
            beyond_cutoff_share=float(np.mean(motion > _MATCH_CUTOFF_UM)),
        )

    def run(self, split: str, limit: int) -> pl.DataFrame:
        """Summarise a whole split (or its first `limit` videos) into one table."""
        videos = self._root.videos(split)[:limit]
        return pl.DataFrame([asdict(self.summarise(video)) for video in videos])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="survey the annotated videos of a competition split")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()

    survey = DatasetSurvey(DataRoot.from_config(arguments.paths))
    table = survey.run(arguments.split, arguments.limit)
    logger.info(table)
    logger.info(table.group_by("prefix").agg(pl.all().exclude("dataset").mean()).sort("prefix"))


if __name__ == "__main__":
    main()

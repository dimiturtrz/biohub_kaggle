"""Dataset statistics that decide method choices, as one committed command with two subcommands.

Both surveys reduce the same raw material — an annotated video plus its ground-truth track store — so
they share the loader and differ only in the reduction:

`stats` walks a split and reports, per video, the node counts, the supervised fraction, the per-link
motion and the number of divisions — the numbers behind the 7 um matching cutoff, the 0.1 division term,
and whether the two acquisition prefixes force a grouped CV split (`docs/PLAN.md`).

`scale` measures cell blob scale from the Laplacian-of-Gaussian response at ground-truth centres, in
micrometres so it is spacing-independent; the detector converts back to per-axis voxels.

    python -m tools.dataset_stats stats --split train --limit 20
    python -m tools.dataset_stats scale --split train
"""

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from jaxtyping import Float, Int
from scipy.ndimage import gaussian_laplace

from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_MATCH_CUTOFF_UM = 7.0
_PREFIX_LENGTH = 4
_MOTION_TAIL_QUANTILE = 0.99
_SCALE_TAIL_PERCENTILES = (10, 90)
_PERCENT = 100.0


def load_annotated_video(root: DataRoot, video: Path) -> tuple[CellVideo, AnnotatedTracks]:
    """Open a video and its ground-truth annotation together — the pair both surveys reduce."""
    return CellVideo.from_ome_zarr(video), AnnotatedTracks.from_geff(root.track_store(video))


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
        opened, tracks = load_annotated_video(self._root, video)
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


@dataclass(frozen=True)
class BlobScaleConfig:
    """The LoG scale sweep and the cheap sampling it runs over.

    A handful of frames per video, the response cropped to a window around each centre — enough centres
    to estimate the scale distribution without touching every voxel of every timepoint.
    """

    sigmas_um: Float[np.ndarray, "s"] = field(
        default_factory=lambda: np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]),
    )
    crop_half: Int[np.ndarray, "3"] = field(default_factory=lambda: np.array([6, 24, 24]))
    frames_per_video: int = 6
    low: float = 0.001
    high: float = 0.999


class BlobScaleSurvey:
    """Reduces annotated centres to the LoG scale that responds most strongly at each."""

    def __init__(self, root: DataRoot, config: BlobScaleConfig) -> None:
        self._root = root
        self._config = config

    def peak_sigma_at(
        self,
        frame: Float[np.ndarray, "z y x"],
        spacing: Spacing,
        point: Int[np.ndarray, "3"],
    ) -> float:
        """The sigma whose scale-normalised LoG response peaks at `point` — that cell's blob scale."""
        half = self._config.crop_half
        lo = np.maximum(point - half, 0)
        hi = np.minimum(point + half + 1, frame.shape)
        crop = frame[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        centre = point - lo
        strengths = [
            -gaussian_laplace(crop, sigma=sigma_um / spacing.as_array())[centre[0], centre[1], centre[2]]
            * (sigma_um**2)
            for sigma_um in self._config.sigmas_um
        ]
        return float(self._config.sigmas_um[int(np.argmax(strengths))])

    def measure(self, videos: list[Path], rng: np.random.Generator) -> Float[np.ndarray, "p"]:
        """Peak sigma at every sampled centre across the given videos, in micrometres."""
        peaks: list[float] = []
        for video in videos:
            cell, tracks = load_annotated_video(self._root, video)
            coords = tracks.graph.coordinates
            timepoints = np.unique(coords[:, 0])
            sample_size = min(self._config.frames_per_video, len(timepoints))
            for timepoint in rng.choice(timepoints, sample_size, replace=False):
                frame = cell.quantiles.normalise(cell.frame(int(timepoint)), self._config.low, self._config.high)
                peaks.extend(
                    self.peak_sigma_at(frame, cell.spacing, point) for point in coords[coords[:, 0] == timepoint][:, 1:]
                )
        return np.array(peaks)


def _report_stats(root: DataRoot, arguments: argparse.Namespace) -> None:
    table = DatasetSurvey(root).run(arguments.split, arguments.limit)
    logger.info(table)
    logger.info(table.group_by("prefix").agg(pl.all().exclude("dataset").mean()).sort("prefix"))


def _report_scale(root: DataRoot, arguments: argparse.Namespace) -> None:
    config = BlobScaleConfig()
    videos = root.videos(arguments.split)[:2] + root.videos(arguments.split)[-2:]
    peaks = BlobScaleSurvey(root, config).measure(videos, np.random.default_rng(0))
    low, high = _SCALE_TAIL_PERCENTILES
    logger.info(
        "n=%d peak sigma um: p%d=%s median=%s p%d=%s mean=%.2f",
        len(peaks),
        low,
        np.percentile(peaks, low),
        np.median(peaks),
        high,
        np.percentile(peaks, high),
        peaks.mean(),
    )
    for sigma in config.sigmas_um:
        logger.info("  %.1f um: %.0f%%", sigma, float(np.mean(peaks == sigma)) * _PERCENT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="dataset statistics that decide method choices")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    subcommands = parser.add_subparsers(required=True)

    stats = subcommands.add_parser("stats", help="node/motion/division survey over a split")
    stats.add_argument("--split", default="train")
    stats.add_argument("--limit", type=int, default=20)
    stats.set_defaults(report=_report_stats)

    scale = subcommands.add_parser("scale", help="LoG blob-scale percentiles over a few videos")
    scale.add_argument("--split", default="train")
    scale.set_defaults(report=_report_scale)

    arguments = parser.parse_args()
    arguments.report(DataRoot.from_config(arguments.paths), arguments)


if __name__ == "__main__":
    main()

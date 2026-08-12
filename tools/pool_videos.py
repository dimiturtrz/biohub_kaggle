"""Write the run's downsample ONCE, offline, instead of re-striding the raw store on every access.

The shipped stores chunk a whole timepoint per block — `(1, 64, 256, 256)` uint16, 8.39 MB — so a read at
the run grid `(1, 4, 4)` decompresses 8.39 MB to return 0.52 MB. That sixteen-fold over-read is paid on every
epoch of every run, plus the normalisation each time. This writes the strided result once: 11.2 ms per frame
becomes 5.2 ms, and the store is a sixteenth of the voxels (~5 GB against ~80 GB for the 199 train videos).

It stores the POOLED RAW integers, not the normalised floats. Normalisation then stays in one place
(`FrameSource`), a quantile change cannot invalidate the store, and the file stays uint16 rather than
doubling to float32. The expensive half is the decompression, not the arithmetic.

Non-destructive and idempotent: raw stores are untouched, output goes under `processed/<dataset>/pooled_<grid>/`
with the grid IN THE NAME so a store can never be read as a different downsample, and a video whose store
already validates is skipped. Validation is shape, dtype, timepoint count and the recorded grid — a mismatch
rewrites that video rather than failing the pass.

    python -m tools.pool_videos --split train
    python -m tools.pool_videos --split train --downsample 1,2,2 --force
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from jaxtyping import Float

from celltrack.data.frame_source import DOWNSAMPLE_ATTR, PooledFrames
from core.data.video import ImageStatistics
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_DATASET = "biohub_cell_tracking"
_SAMPLED_VOXELS = 200_000  # the median is estimated from a sample; an exact one over 8.4M voxels buys nothing


@dataclass(frozen=True)
class FrameStats:
    """One video's intensity summary — the sanity check that a store holds what the loader will believe.

    Reported because the pass already touches every voxel, so these cost nothing extra, and because an input
    the run never characterises is one whose corruption would first show up as a bad SCORE. Quantiles come
    from the video's own OME metadata (the normalisation reads them), so a store whose intensities sit far
    outside its own quantile window is telling you the metadata does not belong to the pixels.
    """

    stem: str
    frames: int
    minimum: float
    maximum: float
    mean: float
    median: float
    q_low: float
    q_high: float

    @classmethod
    def of(cls, stem: str, volume: Float[np.ndarray, "t z y x"], q_low: float, q_high: float) -> "FrameStats":
        """Reduce a pooled video to the numbers worth printing, the median from a seeded sample."""
        flat = volume.reshape(-1)
        rng = np.random.default_rng(0)
        sample = flat if flat.size <= _SAMPLED_VOXELS else flat[rng.integers(0, flat.size, _SAMPLED_VOXELS)]
        return cls(
            stem=stem,
            frames=int(volume.shape[0]),
            minimum=float(flat.min()),
            maximum=float(flat.max()),
            mean=float(flat.mean()),
            median=float(np.median(sample)),
            q_low=q_low,
            q_high=q_high,
        )

    def line(self) -> str:
        """One log line per video — the shape, the intensity spread, and the window it will be normalised by."""
        return (
            f"{self.stem}: {self.frames} frames | min {self.minimum:.0f} max {self.maximum:.0f} "
            f"mean {self.mean:.1f} median {self.median:.0f} | quantiles {self.q_low:.0f}-{self.q_high:.0f}"
        )


class PooledStore:
    """Writes and validates one dataset's pooled copies — the offline half of the frame read."""

    @staticmethod
    def valid(store: Path, expected_shape: tuple[int, ...], downsample: tuple[int, int, int]) -> bool:
        """Whether an existing store can be trusted, so a second run of this tool writes nothing.

        Checks the grid it RECORDS as well as the shape it has: a store of the right shape written at another
        downsample would otherwise pass, and it is the one error that would silently feed the wrong voxels.
        """
        if not store.exists():
            return False
        try:
            array = zarr.open_array(store)
            written = tuple(int(axis) for axis in array.attrs[DOWNSAMPLE_ATTR])
            return bool(array.shape == expected_shape and array.dtype == np.uint16 and written == downsample)
        except (KeyError, ValueError, OSError):
            return False  # unreadable or missing its provenance: rewrite rather than trust

    @staticmethod
    def write(source: Path, store: Path, downsample: tuple[int, int, int]) -> FrameStats:
        """Stride one video into its pooled store and return the intensity summary of what was written."""
        group = zarr.open_group(source, mode="r")
        raw = group["0"]
        dz, dy, dx = downsample
        pooled = np.asarray(raw[:, ::dz, ::dy, ::dx], dtype=np.uint16)  # type: ignore[index]
        quantiles = cast(ImageStatistics, group.attrs["image_statistics"])["quantiles"]
        array = zarr.create_array(
            store=str(store),
            shape=pooled.shape,
            chunks=(1, *pooled.shape[1:]),  # one pooled frame per chunk: the unit every reader asks for
            dtype=np.uint16,
            overwrite=True,
        )
        array[:] = pooled
        # The store carries its own provenance so a reader never has to be TOLD what it is: the grid it was
        # written at, the raw shape it came from, and the intensity window the normalisation will apply.
        array.attrs[DOWNSAMPLE_ATTR] = list(downsample)
        array.attrs["source_shape"] = list(raw.shape)  # type: ignore[union-attr]
        array.attrs["image_statistics"] = group.attrs["image_statistics"]
        return FrameStats.of(source.stem, pooled, float(quantiles["0.001"]), float(quantiles["0.999"]))


class PoolCommand:
    """The CLI: pool a split, skipping what is already there and reporting what the loader will see."""

    @staticmethod
    def main() -> None:
        """Pool every video of a split, then report the corpus the trainers will read."""
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
        parser.add_argument("--split", default="train", help="which split of the competition data to pool")
        parser.add_argument("--downsample", default="1,4,4", help="the grid to pool onto, as dz,dy,dx")
        parser.add_argument("--limit", type=int, default=0, help="pool only the first N videos (0 = all)")
        parser.add_argument("--force", action="store_true", help="rewrite stores that already validate")
        args = parser.parse_args()

        root = DataRoot.from_config(args.config)
        downsample = cast(tuple[int, int, int], tuple(int(axis) for axis in args.downsample.split(",")))
        videos = root.videos(args.split)
        videos = videos[: args.limit] if args.limit else videos
        directory = PooledFrames.directory(root.processed(_DATASET), downsample)
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("pooling %d %s videos onto %s -> %s", len(videos), args.split, downsample, directory)

        written, skipped, stats = 0, 0, []
        with Obs.timed(logger, f"pooling {len(videos)} videos"):
            for video in Obs.progress(videos, "videos", len(videos)):
                store = directory / f"{video.stem}.zarr"
                expected = PoolCommand._expected_shape(video, downsample)
                if not args.force and PooledStore.valid(store, expected, downsample):
                    skipped += 1
                    continue
                stats.append(PooledStore.write(video, store, downsample))
                logger.info("  %s", stats[-1].line())
                written += 1
        PoolCommand._report(written, skipped, stats, directory)

    @staticmethod
    def _expected_shape(video: Path, downsample: tuple[int, int, int]) -> tuple[int, ...]:
        """The pooled shape a store must have, computed from the RAW store rather than assumed."""
        raw = zarr.open_group(video, mode="r")["0"]
        timepoints, *spatial = raw.shape  # type: ignore[union-attr]
        return (timepoints, *(-(-size // stride) for size, stride in zip(spatial, downsample, strict=True)))

    @staticmethod
    def _report(written: int, skipped: int, stats: list[FrameStats], directory: Path) -> None:
        """What the pass did and what it holds — the pooled corpus's own summary, not an inference from it."""
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        logger.info("wrote %d, skipped %d already valid | store %.2f GB", written, skipped, size / 1e9)
        if not stats:
            return
        means = [stat.mean for stat in stats]
        logger.info(
            "intensity across %d written videos: min %.0f max %.0f | per-video mean %.1f-%.1f",
            len(stats),
            min(stat.minimum for stat in stats),
            max(stat.maximum for stat in stats),
            min(means),
            max(means),
        )


if __name__ == "__main__":
    PoolCommand.main()

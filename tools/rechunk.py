"""Re-chunk the training videos to crop-sized zarr blocks so a training crop is a small read.

The shipped stores chunk one whole timepoint per block — `(1, 64, 256, 256)`, ~8 MB. Training samples a
`(16, 64, 64)` crop, so keeping a crop meant decompressing a whole 8 MB plane for ~0.25 MB of signal: a
32x over-read that left the GPU waiting on the CPU. Re-chunking to small 3D blocks means `CellVideo.window`
touches only the handful of blocks a crop overlaps.

Non-destructive: the raw stores are untouched; re-chunked copies go under `processed/…/rechunked/`, carrying
the same OME metadata so `CellVideo.from_ome_zarr` opens them unchanged. Videos are converted in parallel
across a process pool because the rewrite is ~80 GB.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

from core.paths import DataRoot

CONFIG = Path(__file__).parents[1] / "paths.yaml"
_COMPETITION = "biohub_cell_tracking"
_FULL_RESOLUTION = "0"
# One timepoint deep, a quarter of z, an eighth of y/x: ~0.5 MB, so a (16, 64, 64) crop overlaps a few
# blocks instead of a whole plane. Aligned to the crop so the read amplification stays near one.
_CHUNKS = (1, 16, 128, 128)

logger = logging.getLogger(__name__)


def rechunked_store(root: DataRoot, video: Path) -> Path:
    """Where a raw video's re-chunked copy lives — same stem under `processed/…/rechunked/`."""
    return root.processed(_COMPETITION) / "rechunked" / video.name


def _convert(source: Path, destination: Path) -> str:
    """Copy one video into a small-chunk store, timepoint by timepoint, carrying its OME metadata."""
    if (destination / _FULL_RESOLUTION / "zarr.json").exists():
        return f"skip {source.name} (exists)"
    source_group = zarr.open_group(source, mode="r")
    frames = source_group[_FULL_RESOLUTION]
    destination_group = zarr.open_group(destination, mode="w")
    destination_group.attrs.update(dict(source_group.attrs))
    out = destination_group.create_array(_FULL_RESOLUTION, shape=frames.shape, chunks=_CHUNKS, dtype=frames.dtype)
    for timepoint in range(frames.shape[0]):
        out[timepoint] = np.asarray(frames[timepoint])
    return f"done {source.name} {frames.shape}"


def main() -> None:
    """Re-chunk every training video into a crop-sized store under the processed root, in parallel."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Re-chunk training videos to crop-sized zarr blocks.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="cap videos converted; 0 = all")
    arguments = parser.parse_args()

    root = DataRoot.from_config(CONFIG)
    videos = root.videos(arguments.split)
    videos = videos[: arguments.limit] if arguments.limit else videos
    logger.info("re-chunking %d %s videos -> %s", len(videos), arguments.split, _CHUNKS)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(_convert, video, rechunked_store(root, video)): video for video in videos}
        for done, future in enumerate(as_completed(futures), start=1):
            logger.info("[%d/%d] %s", done, len(videos), future.result())


if __name__ == "__main__":
    main()

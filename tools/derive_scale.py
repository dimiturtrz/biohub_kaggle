"""Measure cell blob scale from the LoG response at ground-truth centres, over a few videos.

The scale at which an annotated cell responds most strongly to a Laplacian-of-Gaussian *is* its scale.
Reported in micrometres so it is spacing-independent; the detector converts back to per-axis voxels.
Kept cheap: a handful of frames per video, response cropped to a window around each centre.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_laplace

from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.paths import DataRoot

CONFIG = Path(__file__).parents[1] / "paths.yaml"
SIGMAS_UM = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
FRAMES_PER_VIDEO = 6
HALF = np.array([6, 24, 24])  # crop half-extent in voxels (z, y, x): ~10um each way given spacing
LOW, HIGH = 0.001, 0.999


def peak_sigma_at(frame, spacing, point):
    lo = np.maximum(point - HALF, 0)
    hi = np.minimum(point + HALF + 1, frame.shape)
    crop = frame[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    centre = point - lo
    strengths = []
    for sigma_um in SIGMAS_UM:
        log = gaussian_laplace(crop, sigma=sigma_um / spacing.as_array()) * (sigma_um**2)
        strengths.append(-log[centre[0], centre[1], centre[2]])
    return SIGMAS_UM[int(np.argmax(strengths))]


def main():
    root = DataRoot.from_config(CONFIG)
    rng = np.random.default_rng(0)
    videos = root.videos("train")[:2] + root.videos("train")[-2:]
    peaks = []
    for video in videos:
        cell = CellVideo.from_ome_zarr(video)
        coords = cell and AnnotatedTracks.from_geff(root.track_store(video)).graph.coordinates
        timepoints = np.unique(coords[:, 0])
        for t in rng.choice(timepoints, min(FRAMES_PER_VIDEO, len(timepoints)), replace=False):
            frame = cell.quantiles.normalise(cell.frame(int(t)), LOW, HIGH)
            for point in coords[coords[:, 0] == t][:, 1:]:
                peaks.append(peak_sigma_at(frame, cell.spacing, point))
    peaks = np.array(peaks)
    print(
        f"n={len(peaks)} peak sigma um: p10={np.percentile(peaks, 10)} "
        f"median={np.median(peaks)} p90={np.percentile(peaks, 90)} mean={peaks.mean():.2f}"
    )
    for s in SIGMAS_UM:
        print(f"  {s:.1f} um: {(peaks == s).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()

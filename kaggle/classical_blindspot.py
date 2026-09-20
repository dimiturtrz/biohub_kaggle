"""Does a detector with NO learned weights see the cells the whole learned family misses?

E36 read node recall at cell separation (2.87 um) as 0.824 and found it FLAT across all nine cached
detectors. Flat across nine variants of one training recipe is not a tuning knob that was left unturned;
it is the signature of a blind spot the whole family SHARES. Nothing in that measurement says whether the
missing 17.6% is unreachable in the pixels or merely unreachable to a UNet trained the way we train.

A zero-weight detector answers it. The multi-scale difference-of-Gaussians below is harvested from
`fabriciodasilva/biohub-dodecatiad-cell-tracking` -- a CPU-only, model-free tracker, and so the one
detector family that cannot share the learned pool's inductive bias. If it recovers a real slice of the
cells the champion misses, detector-side decorrelation is alive and it acts on the NODE term, which E56
showed is the term the board actually follows. If it recovers nothing beyond chance, 0.824 is a property
of the data or the annotation and the learned family is exonerated.

"Recovers" has to beat coincidence, because a blob detector emits peaks freely: with hundreds of
detections in a ~100 um cube something is always a few microns away. So every rescue is scored twice --
once against the DoG detections of the same frame, once against its detections from a frame far away
(NULL_SHIFT_FRAMES). The null preserves the detector's count, density and spatial distribution exactly
and destroys only the cell-to-detection identity, so the excess of real over null is the whole claim.

    .venv/Scripts/python kaggle/classical_blindspot.py --submission <champion.csv> --train <dir of *.geff>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, SPACING
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from short_component_prune import read_submission

from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo

log = logging.getLogger(__name__)

SEPARATION_UM = 2.87
NULL_SHIFT_FRAMES = 50
XY_DOWNSAMPLE = 4
DOG_SCALES_UM = ((2.0, 6.0), (3.0, 9.0), (4.5, 13.5))
MIN_DISTANCE_UM = 3.0
RELATIVE_THRESHOLD = 0.04
INTENSITY_PERCENTILE = 50.0
NORMALISATION_QUANTILES = (1.0, 99.7)
MAX_PEAKS = 30000


def detect_blobs(volume, spacing):
    """Multi-scale DoG local maxima, in original voxel coordinates -- no weights, no training.

    Harvested from the dodecatiad kernel. The scale-space max over several DoG bands is what lets one
    threshold cover cells of different sizes, which is the part a single-scale learned decoder cannot
    express without having seen that size during training.
    """
    downsampled = volume.astype(np.float32)[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE]
    effective = spacing * np.array([1.0, XY_DOWNSAMPLE, XY_DOWNSAMPLE])

    low, high = np.percentile(downsampled, NORMALISATION_QUANTILES)
    normalised = np.clip((downsampled - low) / max(high - low, 1.0), 0.0, None)

    response = None
    for small_um, large_um in DOG_SCALES_UM:
        band = gaussian_filter(normalised, sigma=small_um / effective) - gaussian_filter(
            normalised, sigma=large_um / effective
        )
        response = band if response is None else np.maximum(response, band)

    peaks = (
        (response == maximum_filter(response, footprint=_ball(MIN_DISTANCE_UM, effective), mode="nearest"))
        & (response >= RELATIVE_THRESHOLD)
        & (normalised >= np.percentile(normalised, INTENSITY_PERCENTILE))
    )
    coordinates = np.argwhere(peaks).astype(np.float64)
    strongest = np.argsort(response[peaks])[::-1][:MAX_PEAKS]
    coordinates = coordinates[strongest]
    coordinates[:, 1:] *= XY_DOWNSAMPLE
    return coordinates


def _ball(radius_um, effective_spacing):
    """An ellipsoid covering radius_um of physical space on an anisotropic grid."""
    radius = np.maximum(1, np.round(radius_um / effective_spacing).astype(int))
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    return sum((axis / r) ** 2 for axis, r in zip(grids, radius, strict=True)) <= 1.0


def _within(query_um, reference_um, ruler_um):
    """Which query points have a reference point inside the ruler."""
    if len(reference_um) == 0 or len(query_um) == 0:
        return np.zeros(len(query_um), dtype=bool)
    return cKDTree(reference_um).query(query_um, k=1)[0] <= ruler_um


def measure(submission: Path, train: Path, stride: int, ruler_um: float):
    """Per frame: what the champion missed, and whether the model-free detector saw it."""
    nodes, _ = read_submission(submission)
    scale = SPACING.as_array()
    tally = dict.fromkeys(("gt", "champion_hit", "missed", "rescued", "rescued_null"), 0)
    densities = []

    for dataset in sorted(nodes):
        geff = train / f"{dataset}.geff"
        store = train / f"{dataset}.zarr"
        if not (geff.exists() and store.exists()):
            continue

        graph = AnnotatedTracks.from_geff(geff).graph
        times, positions = graph.timepoints(), graph.positions().astype(float)
        predicted = np.array(list(nodes[dataset].values()), dtype=float)
        video = CellVideo.from_ome_zarr(store)
        horizon = video.timepoint_count

        for timepoint in range(0, horizon, stride):
            truth = positions[times == timepoint]
            if len(truth) == 0:
                continue
            champion = predicted[predicted[:, 0] == timepoint][:, 1:]
            classical = detect_blobs(video.frame(timepoint), scale)
            elsewhere = detect_blobs(video.frame((timepoint + NULL_SHIFT_FRAMES) % horizon), scale)
            densities.append((len(champion), len(classical)))

            found = _within(truth * scale, champion * scale, ruler_um)
            missed = truth[~found]
            tally["gt"] += len(truth)
            tally["champion_hit"] += int(found.sum())
            tally["missed"] += len(missed)
            tally["rescued"] += int(_within(missed * scale, classical * scale, ruler_um).sum())
            tally["rescued_null"] += int(_within(missed * scale, elsewhere * scale, ruler_um).sum())
        log.info("  %s done (%d frames)", dataset, len(range(0, horizon, stride)))

    return tally, np.array(densities, dtype=float)


def report(tally, densities, ruler_um):
    missed = max(tally["missed"], 1)
    champion_per_frame, classical_per_frame = densities.mean(axis=0)
    log.info("")
    log.info("ruler %.2f um   GT cells %d over %d frames", ruler_um, tally["gt"], len(densities))
    log.info("  champion node recall                  %.4f", tally["champion_hit"] / max(tally["gt"], 1))
    log.info("  GT cells the champion MISSES          %d", tally["missed"])
    log.info("  ... a model-free detection within ruler   %.4f", tally["rescued"] / missed)
    log.info("  ... the SAME against a shifted frame      %.4f   (this is chance)", tally["rescued_null"] / missed)
    log.info("  EXCESS over the matched null              %+.4f", (tally["rescued"] - tally["rescued_null"]) / missed)
    log.info("")
    log.info(
        "  detections per frame: champion %.0f | model-free %.0f (%.1fx)",
        champion_per_frame,
        classical_per_frame,
        classical_per_frame / max(champion_per_frame, 1.0),
    )
    log.info("A rescue rate at the null is a coincidence; the excess is the only part that is a finding.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="the learned family's nodes")
    parser.add_argument("--train", type=Path, required=True, help="directory holding *.geff and *.zarr")
    parser.add_argument("--stride", type=int, default=10, help="frame subsampling; the detector is the cost")
    parser.add_argument("--ruler", type=float, default=SEPARATION_UM, help="match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    log.info("official node match is %.1f um; reading at %.2f um", OFFICIAL_UM, args.ruler)
    report(*measure(args.submission, args.train, args.stride, args.ruler), args.ruler)


if __name__ == "__main__":
    main()

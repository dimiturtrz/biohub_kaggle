"""Is each competition train crop a window cut out of a PUBLIC, densely annotated embryo?

The Biohub server publishes `ZSNS00N_tracks.csv` -- `track_id,t,z,y,x,parent_track_id`, our exact GT
schema, for the whole embryo. The competition's own GT is a sparse sample of the same imaging: one crop
here labels 52 of an estimated 25755 cells, 0.2%. If a crop's annotated cells are literally rows of that
public table, then every cell the crop leaves unlabelled -- E54c's median-61-frame "thieves", E58's
mid-track decoy nodes -- already has a published track, and the supervision the centre-offset head has
never had exists on disk.

A crop is anonymised: integer voxel coordinates, time restarted at zero, no translation transform in its
zarr metadata. But its GT chains run 40-100 frames, and a chain's sequence of step LENGTHS is invariant to
both the unknown translation and any rotation, while being far too specific to collide by chance. So this
takes the longest chain in a crop, converts it to micrometres, and slides it against the step-length
sequence of every public track. A match recovers the volume, the track id and the time offset at once.

Only TRAIN crops are ever read. Running this against the test crops would read the competition's answer out
of a public file; that is leakage, not tracking, and the CLI refuses any directory that is not train/.

    .venv/Scripts/python kaggle/public_gt_registration.py --train <geff dir> --public <ZSNS001_tracks.csv>
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fragment_selection_gate import SPACING

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

STEP_TOLERANCE_UM = 1.5
"""A crop coordinate is rounded to its voxel, so a step length can be off by about one anisotropic voxel."""

MIN_CHAIN = 12
"""Shorter chains stop being unique against thirty million public steps."""


def longest_chain(tracks: AnnotatedTracks) -> np.ndarray:
    """The coordinates of the longest single-parent chain in a crop's GT, in time order."""
    successors: dict[int, list[int]] = defaultdict(list)
    for source, target in tracks.graph.edges.tolist():
        successors[source].append(target)
    row_of = {int(node): row for row, node in enumerate(tracks.graph.node_ids.tolist())}
    length: dict[int, int] = {}
    order = sorted(row_of, key=lambda node: -tracks.graph.coordinates[row_of[node], 0])
    for node in order:
        length[node] = 1 + max((length.get(child, 0) for child in successors[node]), default=0)
    walk = [max(length, key=lambda node: length[node])]
    while successors[walk[-1]]:
        walk.append(max(successors[walk[-1]], key=lambda child: length.get(child, 0)))
    return tracks.graph.coordinates[[row_of[node] for node in walk]]


def step_lengths(positions_um: np.ndarray) -> np.ndarray:
    """The distance travelled between consecutive positions of one chain."""
    return np.linalg.norm(np.diff(positions_um, axis=0), axis=1)


def public_steps(csv: Path, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every public track's step lengths, concatenated, with the track id and frame each step starts at.

    `stride` is how many published timepoints one competition frame advances -- unknown, because a crop
    restarts its clock at zero and says nothing about the cadence it was sampled at.
    """
    table = pd.read_csv(csv, usecols=["track_id", "t", "z", "y", "x"]).to_numpy(dtype=np.float64)
    table = table[np.lexsort((table[:, 1], table[:, 0]))]
    same_track = table[stride:, 0] == table[:-stride, 0]
    spaced = np.isclose(table[stride:, 1] - table[:-stride, 1], float(stride))
    keep = same_track & spaced
    lengths = np.linalg.norm(table[stride:, 2:5] - table[:-stride, 2:5], axis=1)
    return lengths[keep], table[:-stride, 0][keep], table[:-stride, 1][keep]


def windows_matching(haystack: np.ndarray, track_of: np.ndarray, needle: np.ndarray, tolerance: float) -> np.ndarray:
    """Start offsets where every step of the needle sits within tolerance, inside a single public track."""
    span = len(needle)
    starts = np.arange(len(haystack) - span + 1)
    starts = starts[track_of[starts] == track_of[starts + span - 1]]
    for offset, step in enumerate(needle.tolist()):
        starts = starts[np.abs(haystack[starts + offset] - step) <= tolerance]
        if not len(starts):
            break
    return starts


def discriminative(needle: np.ndarray, tolerance: float) -> bool:
    """Does the chain actually move enough for a step-length match to mean anything?

    A cell whose every step sits inside one tolerance band matches any slow track in the volume, so a hit
    on such a chain is not evidence and a miss is not evidence either.
    """
    return bool(needle.max() - needle.min() > 2.0 * tolerance)


def probe(train_dir: Path, public_csv: Path, limit: int, stride: int) -> None:
    if train_dir.name != "train":
        message = f"refusing a directory that is not train/: {train_dir}"
        raise SystemExit(message)
    log.info("reading %s", public_csv)
    haystack, track_of, frame_of = public_steps(public_csv, stride)
    log.info("public steps: %d over %d tracks", len(haystack), len(np.unique(track_of)))
    scale = SPACING.as_array()
    hits = 0
    tested = 0
    for crop in sorted(train_dir.glob("*.geff"))[:limit]:
        chain = longest_chain(AnnotatedTracks.from_geff(crop))
        needle = step_lengths(chain[:, 1:] * scale)
        if len(chain) < MIN_CHAIN or not discriminative(needle, STEP_TOLERANCE_UM):
            log.info("%s chain=%d SKIPPED (no power)", crop.name, len(chain))
            continue
        tested += 1
        starts = windows_matching(haystack, track_of, needle, STEP_TOLERANCE_UM)
        hits += bool(len(starts))
        log.info(
            "%s chain=%d spread=%.2fum matches=%d %s",
            crop.name,
            len(chain),
            float(needle.max() - needle.min()),
            len(starts),
            [(int(track_of[s]), int(frame_of[s])) for s in starts[:3].tolist()],
        )
    log.info("crops with at least one public match: %d of %d tested", hits, tested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    probe(arguments.train, arguments.public, arguments.limit, arguments.stride)


if __name__ == "__main__":
    main()

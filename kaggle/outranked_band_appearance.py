"""Does raw appearance separate the TRUE target from the RIVAL that outranked it?

E45 closed the geometric cues on the outranked band and concluded it was appearance-bound. The prior
appearance kill was measured at n=37; this holds the band at n>1200 with the E43 denominator.

The measurement only means something with a control, so every number below is computed twice:

    BAND     true edges a rival outranks -- where a tie-breaker would have to work.
    MUTUAL   true edges that are mutual-best, scored against their runner-up rival the same way.

and both are stratified by the true edge's displacement, because the band is a fast-cell population
(E45) and appearance decorrelates with motion on its own. Only the MATCHED stratum licenses a verdict.
Patch variance is reported so a null can be told apart from a dead read.

    .venv/Scripts/python kaggle/outranked_band_appearance.py --cache <dir of *.npz> --train <dir of *.geff + *.zarr>
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from fragment_selection_gate import OFFICIAL_UM, matched_detections

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

AFFINITY_COLUMN = 2
SUBSTRATE = np.array([1.0, 4.0, 4.0])
RADIUS = np.array([2, 6, 6])
STRATA = ((0.0, 2.0), (2.0, 3.46), (3.46, np.inf))
MIN_STRATUM = 20


def patch(frame, point):
    """Cell-sized cube around a detection, or None where the cube leaves the volume."""
    low = np.maximum(point - RADIUS, 0)
    high = np.minimum(point + RADIUS + 1, frame.shape)
    block = frame[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
    if block.size != np.prod(2 * RADIUS + 1):
        return None
    return block.astype(np.float64).ravel()


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return None if scale == 0 else float(a @ b / scale)


def contests(cache_path: Path, geff: Path, ruler_um: float):
    """Yield (source, true target, rival target, group) for every GT edge with a competitor."""
    cached = np.load(cache_path)
    coords = cached["coords"].astype(np.float64)
    proposed = cached["edges"]
    source = proposed[:, 0].astype(int)
    target = proposed[:, 1].astype(int)
    affinity = proposed[:, AFFINITY_COLUMN]

    out_of, into = defaultdict(list), defaultdict(list)
    for a, b, score in zip(source, target, affinity, strict=True):
        out_of[a].append((score, b))
        into[b].append(score)
    weight = {(int(a), int(b)): s for a, b, s in zip(source, target, affinity, strict=True)}

    graph = AnnotatedTracks.from_geff(geff).graph
    matched = matched_detections(graph, coords, ruler_um)
    row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}
    for head, tail in np.asarray(graph.edges):
        a, b = matched[row[int(head)]], matched[row[int(tail)]]
        if a < 0 or b < 0:
            continue
        score = weight.get((int(a), int(b)))
        if score is None:
            continue
        best_score, best_target = max(out_of[a])
        mutual = score >= best_score and score >= max(into[b])
        others = [(s, t) for s, t in out_of[a] if t != b]
        if not others:
            continue
        rival = best_target if not mutual and best_target != b else max(others)[1]
        yield int(a), int(b), int(rival), "mutual" if mutual else "band"


def collect(cache_dir: Path, train_dir: Path, ruler_um: float):
    scored = {"band": [], "mutual": []}
    variance = []
    for cache_path in sorted(cache_dir.glob("*.npz")):
        geff, volume = train_dir / f"{cache_path.stem}.geff", train_dir / f"{cache_path.stem}.zarr"
        if not geff.exists() or not volume.exists():
            continue
        coords = np.load(cache_path)["coords"].astype(int)
        array = zarr.open(str(volume))["0"]
        by_frame = defaultdict(list)
        for contest in contests(cache_path, geff, ruler_um):
            by_frame[int(coords[contest[0], 0])].append(contest)

        for time in sorted(by_frame):
            if time + 1 >= array.shape[0]:
                continue
            here, ahead = np.asarray(array[time]), np.asarray(array[time + 1])
            for a, b, rival, group in by_frame[time]:
                cubes = (patch(here, coords[a, 1:]), patch(ahead, coords[b, 1:]), patch(ahead, coords[rival, 1:]))
                if any(cube is None for cube in cubes):
                    continue
                true_score, rival_score = correlation(cubes[0], cubes[1]), correlation(cubes[0], cubes[2])
                if true_score is None or rival_score is None:
                    continue
                displacement = np.linalg.norm((coords[b, 1:] - coords[a, 1:]) / SUBSTRATE)
                scored[group].append((true_score, rival_score, displacement))
                variance.append((cubes[0].std(), cubes[1].std()))
    return scored, np.array(variance)


def report(scored, variance):
    log.info(
        "patch std: source median %.1f  target median %.1f   zero-variance %d / %d",
        np.median(variance[:, 0]),
        np.median(variance[:, 1]),
        int((variance == 0).sum()),
        variance.size,
    )
    for group in ("band", "mutual"):
        values = np.array(scored[group])
        decisive = values[:, 0] != values[:, 1]
        log.info(
            "%-7s n=%5d  NCC true %+.4f  rival %+.4f  picks true %.4f (chance 0.5, n=%d)",
            group,
            len(values),
            values[:, 0].mean(),
            values[:, 1].mean(),
            (values[decisive, 0] > values[decisive, 1]).mean(),
            int(decisive.sum()),
        )
        for low, high in STRATA:
            inside = (values[:, 2] >= low) & (values[:, 2] < high) & decisive
            if inside.sum() < MIN_STRATUM:
                continue
            log.info(
                "    displacement [%.2f, %.2f)  n=%5d  picks true %.4f",
                low,
                high,
                int(inside.sum()),
                (values[inside, 0] > values[inside, 1]).mean(),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching *.geff and *.zarr")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(*collect(args.cache, args.train, args.ruler))


if __name__ == "__main__":
    main()

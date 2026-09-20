"""How many GT edges does the detector actually cost us -- and how much of that answer is the ruler?

An edge is lost to DETECTION when an endpoint has nothing near it, so the count depends entirely on what
"near" means, and the two candidate rulers disagree by more than an order of magnitude:

    2.87 um   one cell separation -- the tightest distance at which two cells are still distinct
    7.00 um   what the official metric matches at, and therefore what scores

Reading the loss at cell separation counts a cell whose own detection sits 3-4 um off as undetected, which
is a localization error the official matcher never sees. The tell is that the tight ruler returns an edge
loss far too large to be compatible with the leaderboard score the same detector earns -- an arithmetic
contradiction, not a judgement call. Running every ruler side by side makes that visible instead of
arguable, and sizes the population a recall-recovery pass should actually target.

    .venv/Scripts/python kaggle/edge_loss_by_ruler.py --cache <dir of *.npz> --train <dir of *.geff>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from core.data.tracks import AnnotatedTracks
from core.geometry import Spacing

log = logging.getLogger(__name__)

SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
SEPARATION_UM = 2.87
OFFICIAL_UM = 7.0
RULERS_UM = (SEPARATION_UM, 5.0, OFFICIAL_UM)


def nearest_detection(graph, coords):
    """Per GT node: distance in um to the nearest detection in that node's own frame."""
    scale = SPACING.as_array()
    times, positions = graph.timepoints(), graph.positions()
    distance = np.full(len(positions), np.inf)
    for t in np.unique(times):
        frame = times == t
        detected = coords[coords[:, 0] == t][:, 1:]
        if len(detected) == 0:
            continue
        distance[frame] = cKDTree(detected * scale).query(positions[frame].astype(np.float64) * scale, k=1)[0]
    return distance


def measure(cache_dir: Path, train_dir: Path):
    nodes, edges = 0, 0
    blind_nodes = dict.fromkeys(RULERS_UM, 0)
    dead_edges = dict.fromkeys(RULERS_UM, 0)

    for path in sorted(cache_dir.glob("*.npz")):
        geff = train_dir / f"{path.stem}.geff"
        if not geff.exists():
            continue
        graph = AnnotatedTracks.from_geff(geff).graph
        distance = nearest_detection(graph, np.load(path)["coords"].astype(np.float64))

        row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}
        pairs = np.asarray(graph.edges)
        source = np.array([row[int(a)] for a in pairs[:, 0]])
        target = np.array([row[int(b)] for b in pairs[:, 1]])

        nodes += len(distance)
        edges += len(pairs)
        for ruler in RULERS_UM:
            blind = distance > ruler
            blind_nodes[ruler] += int(blind.sum())
            dead_edges[ruler] += int((blind[source] | blind[target]).sum())

    return nodes, edges, blind_nodes, dead_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached detector *.npz (coords)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    nodes, edges, blind_nodes, dead_edges = measure(args.cache, args.train)
    log.info("GT nodes %d   GT edges %d", nodes, edges)
    for ruler in RULERS_UM:
        log.info(
            "  ruler %4.2f um   blind nodes %6d = %.4f   edges killed %6d = %.4f",
            ruler,
            blind_nodes[ruler],
            blind_nodes[ruler] / nodes,
            dead_edges[ruler],
            dead_edges[ruler] / edges,
        )


if __name__ == "__main__":
    main()

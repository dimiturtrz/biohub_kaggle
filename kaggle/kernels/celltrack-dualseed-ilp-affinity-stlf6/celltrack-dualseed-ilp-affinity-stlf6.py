"""Dual-seed detector + GLOBAL prob-driven ILP linker (edge-transformer affinities) + stlf6 + linefit.

The frontier's 0.913 core, ours: instead of the per-frame greedy motion linker, a global min-cost integer
program over the whole video picks the edge set the learned associations most support — cost per candidate
edge = distance - bonus * P(s->t), no flat link reward, so an edge is taken only where its learned score
beats its distance (frontier edge_weight = -prob). MaxParents(1) + MaxChildren(1) = 1-to-1 (forks lost links
on the proxy). Solved with motile/ilpy/SCIP, all mounted as offline wheels.

Dual-seed 4-test-movie proxy: greedy motion+bonus 0.8989 -> global ILP 0.9125 (+0.0136), crossing 0.91;
dense 6bba_05db0fb1 0.828->0.846, the other three at/above ceiling. Detector blend, edge transformer, stlf6
and linefit are the shipped code path; only the linker is swapped greedy -> global.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

_INPUTS = os.listdir("/kaggle/input")
print("INPUT DIRS:", _INPUTS, flush=True)


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CONFIGS = glob.glob("/kaggle/input/**/weights/unet_transformer/split_0/config.json", recursive=True)
_PACK1 = next(Path(c).parent for c in _CONFIGS if "support-pack" in c or "50ep" in c)
_PACK2 = next(Path(c).parent for c in _CONFIGS if "seed314159" in c)
_PACK_SRC = Path(_find("repo/src/biohub_tracking/models/__init__.py")).parents[2]
_MARKER = _find("celltrack/motion_linking.py")

_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
_WHEELS = sorted(glob.glob(f"{Path(_KIT_ROOT).parent}/**/*.whl", recursive=True))
print("PACKS:", _PACK1, _PACK2, flush=True)
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

sys.path.insert(0, _KIT_ROOT)
sys.path.insert(0, str(_PACK_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from celltrack.edge_scoring import EdgeTransformerScorer  # noqa: E402
from celltrack.ilp_linking import ILPLinker  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_THRESHOLD = 0.99
_GATE_UM = 10.0
_EDGE_BONUS = _GATE_UM  # a certain learned link overrides any within-gate distance
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _spacing(path: str) -> Spacing:
    """The video's physical voxel spacing, read from its OME coordinate transform."""
    scale = zarr.open_group(path, mode="r").attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    return Spacing(z=float(scale[1]), y=float(scale[2]), x=float(scale[3]))


def _blend_nodes(det1: TemporalUNetDetector, det2: TemporalUNetDetector, recipe: object, path: Path) -> TrackGraph:
    """Average the two seeds' per-frame logits on the device, sigmoid, and read the peaks into an edge-free graph."""
    logits1 = det1.logit_volumes(path, recipe, _DEVICE)
    logits2 = det2.logit_volumes(path, recipe, _DEVICE)
    blended = [
        torch.sigmoid(torch.as_tensor(np.stack([a, b]), device=_DEVICE).mean(dim=0)).cpu().numpy()
        for a, b in zip(logits1, logits2)
    ]
    return TemporalUNetDetector.graph_from_volumes(blended, det1.voxel_scale(path), _THRESHOLD, recipe, _DEVICE)


def main() -> None:
    tests = sorted(glob.glob(_TEST_GLOB, recursive=True))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    det1, recipe = TemporalUNetDetector.from_pack(_PACK1, map_location=_DEVICE)
    det2, _ = TemporalUNetDetector.from_pack(_PACK2, map_location=_DEVICE)
    det1, det2 = det1.to(_DEVICE).eval(), det2.to(_DEVICE).eval()
    edge_scorer = EdgeTransformerScorer.from_pack(_PACK1, _DEVICE)
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        nodes = _blend_nodes(det1, det2, recipe, Path(path))
        affinity = edge_scorer.affinities(Path(path), nodes, _DEVICE)
        linker = ILPLinker(
            spacing=_spacing(path),
            max_distance_um=_GATE_UM,
            division=False,
            affinity=affinity,
            affinity_bonus=_EDGE_BONUS,
        )
        graph = smooth.transform(short.transform(linker.link(nodes)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()

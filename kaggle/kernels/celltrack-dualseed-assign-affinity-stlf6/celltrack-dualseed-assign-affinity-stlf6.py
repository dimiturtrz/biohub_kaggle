"""Dual-seed detector + global 1-to-1 assignment linker (edge-transformer affinities) + stlf6 + linefit.

The frontier's 0.913 linker core, ours, with no SCIP. The prob-driven ILP's shipped configuration —
MaxParents(1) + MaxChildren(1) — decomposes into an independent min-cost bipartite matching per frame gap, so
`AssignmentLinker` (pure scipy `linear_sum_assignment` + a zero-cost skip per node) selects the identical edge
set the motile/ilpy/SCIP program did, verified edge-for-edge on the dense test movie (jaccard 1.0). This drops
the offline SCIP wheel stack — whose solver threw `SCIP: unspecified error!` inside the kernel — for numpy and
scipy already in the base image. Edge cost per candidate = distance - bonus * P(s->t); an edge is taken only
where the learned association beats its distance.

Dual-seed 4-test-movie proxy: 0.9125 (assignment) and 0.9134 (+ density gap-bridge), matching the ILP exactly.
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

from celltrack.assignment_linking import AssignmentLinker  # noqa: E402
from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer  # noqa: E402
from celltrack.gap_closer import DensityGapBridge  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_THRESHOLD = 0.99
_GATE_UM = 10.0
# A certain learned association (P=1) must overpower up to two gate-widths of distance: in dense tissue the
# true successor's displacement routinely exceeds one gate, so the nearest-distance prior is actively
# misleading (the crowding-mislink), and the edge transformer — the discriminative signal — has to win against
# it. bonus = 2·gate; the proxy peaks broadly over 2–3·gate (0.9154 at 1·gate → 0.9227 at 2·gate), declining
# past it as pure-association over-trusts confident-but-wrong links. A median-scale autobalance is bead 88x.
_EDGE_BONUS = 2.0 * _GATE_UM
# Two edge-transformer seeds blended in logit space; 0.8·seed1 + 0.2·seed2 is the proxy peak (0.9134→0.9154
# at bonus=1·gate; the sweep extends both together, bead 88x).
_EDGE_BLEND = (0.8, 0.2)
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
    edge_scorer = BlendedEdgeTransformerScorer.from_packs((_PACK1, _PACK2), _EDGE_BLEND, _DEVICE)
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        spacing = _spacing(path)
        nodes = _blend_nodes(det1, det2, recipe, Path(path))
        affinity = edge_scorer.affinities(Path(path), nodes, _DEVICE)
        linker = AssignmentLinker(
            spacing=spacing,
            max_distance_um=_GATE_UM,
            affinity=affinity,
            affinity_bonus=_EDGE_BONUS,
        )
        bridge = DensityGapBridge(spacing=spacing, reach_um=10.0, max_added_fraction=0.05)
        linked = bridge.transform(short.transform(linker.link(nodes)))
        graph = smooth.transform(linked)
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()

"""Dual-seed submission kernel — logit-blend of two pilkwang detectors @0.99 + motion + short-track(6) + linefit.

The only change from `celltrack-pilktunet-t99-motion-stlf6` (LB 0.873): the single detector is replaced by a
logit-level blend of TWO public seeds (`support-pack-50ep` + `temporal-unet3d-seed314159`). Each seed's
pre-sigmoid logits are averaged (the unsaturated blend space — near-1.0 probabilities lose the peak
distinctions the NMS reads), then the mean is sigmoided and the shared peak read-out turns it into nodes.
No retraining — two mounted checkpoints, averaged. Local fold-0 0.9090 -> 0.9184 (+0.0094).

The blend is inlined per video (not the library `BlendDetectorScorer`, which caches to disk) so the whole
hidden test set streams through in memory. Three mounts: both packs + celltrack-kit. Everything downstream —
`graph_from_volumes` read-out, motion linker, short-track(6), linefit — is the shipped fold-0 code path.
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

from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_THRESHOLD = 0.99
_TIGHT_GATE_UM, _LOOSE_GATE_UM = 6.0, 10.0
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
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        nodes = _blend_nodes(det1, det2, recipe, Path(path))
        linker = MotionHungarianLinker(
            spacing=_spacing(path), tight_gate_um=_TIGHT_GATE_UM, loose_gate_um=_LOOSE_GATE_UM
        )
        graph = smooth.transform(short.transform(linker.link(nodes)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()

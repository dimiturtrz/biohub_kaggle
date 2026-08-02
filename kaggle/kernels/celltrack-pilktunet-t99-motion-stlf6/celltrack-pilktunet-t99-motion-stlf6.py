"""Submission kernel — pilkwang detector @0.99 + motion linker + short-track(6) + linefit.

Three changes from `celltrack-pilktunet-motion-stlf` (LB 0.859), all operating-point, no new model:

1. **Detection threshold 0.5 -> 0.99.** The frontier reads the detector at 0.99; our full-41 sweep is
   monotone upward as the threshold rises (fold-0 0.896 -> 0.907) — a sharp detector's extra low-confidence
   peaks are false, and shedding them trims the node count toward the true cell budget.
2. **The shared `TemporalUNetDetector.detections`** instead of a hand-inlined read-out. Its plateau-collapsing
   `PeakExtractor` (one centre per connected component of maxima) replaces the naive `logits == maxpool` mask
   that over-counts a saturated blob; fold-0 rose 0.877 -> 0.896 at the same threshold from that alone. The
   kernel and the local number are now one code path.
3. **Short-track minimum length 3 -> 6**, the frontier's `OUTPUT_MIN_TRACK_LEN`. Dropping tracks shorter than
   six frames removes the short false chains a per-frame detector spawns; fold-0 0.9074 -> 0.9090.

Two frontier post-proc stages were built, measured and **left off** because they do not pay on our
detections without the model half we have not mounted:

- **Division recovery** (`celltrack.division_recovery`, frontier gates 4.7/7.2/7.8): +0.045 on ground-truth
  nodes but detection-limited on real detections (div_jaccard only 0.0114 — most second daughters are never
  detected), and net-negative once short-track(6) already prunes the false chains it adds (0.9090 -> 0.9085).
- **Gap-close reuse** (`celltrack.gap_closer`): inert here (0.9090 -> 0.9090) — at 0.99 there are few isolated
  detections near a gap midpoint to reuse; the frontier's gain comes from *synthetic* insertion, gated by a
  DeepCenter centre-prior veto not mounted here.

Both wait on the DeepCenter recall/veto model. Two mounts: the pilkwang support pack and celltrack-kit.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

_INPUTS = os.listdir("/kaggle/input")
print("INPUT DIRS:", _INPUTS, flush=True)


# Kaggle mounts nest under /kaggle/input/{datasets,competitions}/... — anchor every lookup on a known file.
def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_PACK = Path(_find("weights/unet_transformer/split_0/config.json")).parent
_PACK_SRC = Path(_find("repo/src/biohub_tracking/models/__init__.py")).parents[2]
_MARKER = _find("celltrack/motion_linking.py")

# The detector is pure torch, but celltrack (zarr/jaxtyping/beartype/numcodecs) needs the kit's offline wheels.
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
_WHEELS = sorted(glob.glob(f"{Path(_KIT_ROOT).parent}/**/*.whl", recursive=True))
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

sys.path.insert(0, _KIT_ROOT)
sys.path.insert(0, str(_PACK_SRC))

import torch  # noqa: E402
import zarr  # noqa: E402

from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402
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


def main() -> None:
    tests = sorted(glob.glob(_TEST_GLOB, recursive=True))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    detector, recipe = TemporalUNetDetector.from_pack(_PACK, map_location=_DEVICE)
    detector = detector.to(_DEVICE).eval()
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        detections = detector.detections(Path(path), _THRESHOLD, recipe, _DEVICE)
        linker = MotionHungarianLinker(
            spacing=_spacing(path), tight_gate_um=_TIGHT_GATE_UM, loose_gate_um=_LOOSE_GATE_UM
        )
        graph = smooth.transform(short.transform(linker.link(detections)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()

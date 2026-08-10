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

- **Division recovery** (`celltrack.postproc.division_recovery`, frontier gates 4.7/7.2/7.8): +0.045 on ground-truth
  nodes but detection-limited on real detections (div_jaccard only 0.0114 — most second daughters are never
  detected), and net-negative once short-track(6) already prunes the false chains it adds (0.9090 -> 0.9085).
- **Gap-close reuse** (`celltrack.postproc.gap_closer`): inert here (0.9090 -> 0.9090) — at 0.99 there are few isolated
  detections near a gap midpoint to reuse; the frontier's gain comes from *synthetic* insertion, gated by a
  DeepCenter centre-prior veto not mounted here.

Both wait on the DeepCenter recall/veto model. Two mounts: the pilkwang support pack and celltrack-kit. The
kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`; this file is the
bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/tracker.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import KernelRuntime  # noqa: E402

# kernel_runtime's steps are @staticmethods on KernelRuntime (the arch gate forbids top-level functions);
# alias them to the free names the kernel body uses so the bootstrap stays a flat script.
install_wheels = KernelRuntime.install_wheels
pack_source = KernelRuntime.pack_source
run_submission = KernelRuntime.run_submission
test_videos = KernelRuntime.test_videos

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

import torch  # noqa: E402
import zarr  # noqa: E402

from celltrack.postproc.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linkers.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.detectors.tunet import TemporalUNetDetector  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402


# Kaggle mounts nest under /kaggle/input/{datasets,competitions}/... — anchor the weight lookup on a known file.
def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_PACK = Path(_find("weights/unet_transformer/split_0/config.json")).parent

_THRESHOLD = 0.99
_TIGHT_GATE_UM, _LOOSE_GATE_UM = 6.0, 8.5
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _spacing(path: Path) -> Spacing:
    """The video's physical voxel spacing, read from its OME coordinate transform."""
    scale = zarr.open_group(path, mode="r").attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    return Spacing(z=float(scale[1]), y=float(scale[2]), x=float(scale[3]))


def main() -> None:
    detector, recipe = TemporalUNetDetector.from_pack(_PACK, map_location=_DEVICE)
    detector = detector.to(_DEVICE).eval()
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)

    def predict(_name: str, path: Path) -> TrackGraph:
        detections = detector.detections(path, _THRESHOLD, recipe, _DEVICE)
        linker = MotionHungarianLinker(
            spacing=_spacing(path), tight_gate_um=_TIGHT_GATE_UM, loose_gate_um=_LOOSE_GATE_UM
        )
        return smooth.transform(short.transform(linker.link(detections)))

    run_submission(predict, test_videos())


main()

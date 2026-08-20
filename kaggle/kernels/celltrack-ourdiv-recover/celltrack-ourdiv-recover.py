"""Submission kernel — our-trained temporal-U-Net detector + motion linking + division recovery.

Same OUR-trained detector and motion Hungarian linker as `celltrack-ourstunet-motion-stlf`. The change is a
post-link DIVISION-RECOVERY pass: a one-to-one matching structurally cannot represent a fork, so at every division
one daughter is left unparented and the competition's `+0.1 * division_jaccard` term is forfeited outright. This
pass proposes the missing second daughter under the public frontier's three geometry gates (parent 4.7um, sister
7.2um, existing-child 7.8um) — published constants, not our tuning, and no dual-seed. It runs BEFORE the short-track
filter so the filter's division-preserving carve-out can protect the recovered forks.

The kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`; this file is the
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
from celltrack.postproc.division_recovery import DivisionRecovery  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.detectors.tunet import TemporalUNetDetector  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CHECKPOINT = Path(_find("detector_tunet_ours.pt"))

_THRESHOLD = 0.97
_PARENT_GATE_UM = 4.7
_SISTER_GATE_UM = 7.2
_EXISTING_CHILD_GATE_UM = 7.8
_MAX_ADDED_FRACTION = 0.05
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _spacing(path: Path) -> Spacing:
    group = zarr.open_group(path, mode="r")
    scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1:]
    return Spacing(z=scale[0], y=scale[1], x=scale[2])


def main() -> None:
    detector, recipe = TemporalUNetDetector.from_checkpoint(_CHECKPOINT, map_location=_DEVICE)
    detector = detector.to(_DEVICE).eval()
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)

    def predict(_name: str, path: Path) -> TrackGraph:
        spacing = _spacing(path)
        detections = detector.detections(path, _THRESHOLD, recipe, _DEVICE)
        linker = MotionHungarianLinker(spacing=spacing, tight_gate_um=6.0, loose_gate_um=10.0)
        division = DivisionRecovery(
            spacing=spacing,
            parent_gate_um=_PARENT_GATE_UM,
            sister_gate_um=_SISTER_GATE_UM,
            max_added_fraction=_MAX_ADDED_FRACTION,
            existing_child_gate_um=_EXISTING_CHILD_GATE_UM,
        )
        divided = division.transform(linker.link(detections))
        return smooth.transform(short.transform(divided))

    run_submission(predict, test_videos())


main()

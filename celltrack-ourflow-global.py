"""Submission kernel — our-trained temporal-U-Net detector + a global min-cost-flow linker.

Same OUR-trained detector as `celltrack-ourstunet-motion-stlf` (`detector_tunet_ours.pt`, our fold split, not the
published weights), fed through the shared `celltrack.detectors.tunet` inference path. The change is the LINKER:
instead of the greedy two-pass motion Hungarian, this ships `FlowLinker` — one min-cost circulation over the whole
video that prices a track boundary, so a globally cheaper set of longer trajectories can beat the per-frame-optimal
one exactly where the greedy solver severs a true track across a crowded frame. No dual-seed, no swept precision
hack: `boundary_cost = max_distance_um` (a boundary costs about one max-travel's worth of distance).

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
from celltrack.linkers.flow_linking import FlowLinker  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.detectors.tunet import TemporalUNetDetector  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CHECKPOINT = Path(_find("detector_tunet_ours.pt"))

_THRESHOLD = 0.97
_MAX_DISTANCE_UM = 10.0
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
        linker = FlowLinker(spacing=spacing, max_distance_um=_MAX_DISTANCE_UM, boundary_cost=_MAX_DISTANCE_UM)
        return smooth.transform(short.transform(linker.link(detections)))

    run_submission(predict, test_videos())


main()

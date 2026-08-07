"""Submission kernel — our-trained temporal-U-Net detector + our motion linker.

The detector is the pilkwang architecture (`TemporalUNet3D` + detection head) with weights WE trained on our
fold split (`detector_tunet_ours.pt`), not the published ones. Inference is the shared `celltrack.detectors.tunet`
path — ×4 Y/X downsample, per-video quantile-norm, fake-pair forward, flip-TTA, max-pool peaks — feeding OUR
MotionHungarianLinker + short-track + linefit. The support pack is mounted only for the architecture class;
the weights and the whole pipeline are ours.

The kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`; this file is
the bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/motion_linking.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import install_wheels, pack_source, run_submission, test_videos  # noqa: E402

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


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CHECKPOINT = Path(_find("detector_tunet_ours.pt"))

_THRESHOLD = 0.5
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
        return smooth.transform(short.transform(linker.link(detections)))

    run_submission(predict, test_videos())


main()

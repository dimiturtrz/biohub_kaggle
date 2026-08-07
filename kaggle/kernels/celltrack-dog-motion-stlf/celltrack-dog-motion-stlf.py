"""Submission kernel — classical: multi-scale DoG detector + motion linker + short-track + linefit.

No learned weights. Multi-scale Difference-of-Gaussians detection (GPU) to a per-acquisition node budget
(derived from the training split), linked by our MotionHungarianLinker, cleaned by short-track + linefit.
The honest classical ceiling (dog-motion-stlf), mounting only celltrack-kit — no model pack.

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

from celltrack.kernel_runtime import install_wheels, run_submission, test_videos  # noqa: E402

install_wheels(Path(_KIT_ROOT))

import torch  # noqa: E402

from celltrack.detectors.detection import HIGH, LOW  # noqa: E402
from celltrack.detectors.dog_detection import DoGDetector  # noqa: E402
from celltrack.postproc.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linkers.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.data.video import CellVideo  # noqa: E402

_BUDGET = {"44b6": 36887, "6bba": 16454}
_GATE_UM = 10.0
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)

    def predict(name: str, path: Path) -> TrackGraph:
        cell = CellVideo.from_ome_zarr(path)
        detector = DoGDetector(spacing=cell.spacing, device=_DEVICE)
        linker = MotionHungarianLinker(spacing=cell.spacing, tight_gate_um=6.0, loose_gate_um=_GATE_UM)
        detections = detector.detect(cell.normalised_frames(LOW, HIGH), cell.timepoint_count, _BUDGET[name[:4]])
        return smooth.transform(short.transform(linker.link(detections)))

    run_submission(predict, test_videos())


main()

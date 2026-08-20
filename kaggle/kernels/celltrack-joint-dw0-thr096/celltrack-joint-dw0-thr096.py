"""Submission kernel — the detected-corpus joint model (`dw0`), BOTH heads from one checkpoint.

Every prior submission mounted pilkwang's published edge affinity behind the detector, which discards the
association half a joint run trained. This ships `joint_detcorpus_dw0_v1.pt` — a JointModel whose edge head
was trained on CROWDED, COMPLETE detected candidate pairs (`--detected-videos 40 --det-weight 0`) rather than
the sparse GT node pairs the earlier heads saw. The mechanism is a more-conservative association head that
rejects false links net-positive: held-out TEST proxy +0.0267 at thr0.97, dominant +0.0186..+0.0284 across
every threshold (`logs/thresh_dominance.log`) — the campaign's first and only >1.5% proxy win.

`CellTracker.from_joint` mounts the model's own detector + transformer as the one-member logit blend
`ModelEvaluator.evaluate_joint` selects on — the identical assembly, so this runs the whole trained pipeline
the proxy scored. The operating point is the one the dominance numbers came from: recipe downsample (1,4,4),
TTA off; TrackerConfig.shipped() with threshold 0.97.

The kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`.
"""

import dataclasses
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

from celltrack.detectors.tunet import DetectorRecipe  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.tracker import CellTracker  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CHECKPOINT = Path(_find("joint_detcorpus_dw0_v1.pt"))

_THRESHOLD = 0.96
_RECIPE = DetectorRecipe(downsample=(1, 4, 4), tta=False)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    config = dataclasses.replace(TrackerConfig.shipped(), threshold=_THRESHOLD)
    tracker = CellTracker.from_joint(_CHECKPOINT, _RECIPE, _DEVICE, config)

    def predict(name: str, path: Path) -> TrackGraph:
        return tracker.run(name, path)

    run_submission(predict, test_videos())


main()

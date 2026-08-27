"""Submission kernel — finer (1,2,2) joint with slack-links HOCT head (`rzvw` cheap-first-cut).

Anchor bet of the "fill the slots" window: a JointModel whose detector decodes on the FINER (1,2,2) grid
(half the y/x voxel = ~2.4um faithful NMS suppression, vs (1,4,4)'s 1.6um under-suppression) and whose HOCT
edge head was trained head-only over that detector with SLACK links — a per-source outside-option that lets
the association head DECLINE a spurious candidate rather than force-assign it. The mechanism under test: does
slack absorb finer's intra-frame over-splits (finer-solo = 0.779 faithful, fragments-on-sparse) so the
recall axis the (1,4,4) champion structurally lacks becomes net-positive?

GATED: only shipped if the off_eval6 faithful ruler (reproduces LB 0.921) clears ~0.910 with margin. Below
that = HEAD-BOUND, escalate to g89y segment-selection ILP, NOT a submission.

`CellTracker.from_joint` mounts the checkpoint's OWN detector + transformer (head width/depth read off the
checkpoint), validating the recipe downsample against the checkpoint's — a hard guard against decoding at a
grid the transformer never saw. Operating point: recipe downsample (1,2,2), TTA off, TrackerConfig.shipped()
at threshold picked by the faithful gate.
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


_CHECKPOINT = Path(_find("joint_hoct_slack_cut.pt"))

_THRESHOLD = 0.97
_RECIPE = DetectorRecipe(downsample=(1, 2, 2), tta=False)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    config = dataclasses.replace(TrackerConfig.shipped(), threshold=_THRESHOLD)
    tracker = CellTracker.from_joint(_CHECKPOINT, _RECIPE, _DEVICE, config)

    def predict(name: str, path: Path) -> TrackGraph:
        return tracker.run(name, path)

    run_submission(predict, test_videos())


main()

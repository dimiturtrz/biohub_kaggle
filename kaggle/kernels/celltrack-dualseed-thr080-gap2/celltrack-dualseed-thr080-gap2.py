"""Champion (dual-seed flow @thr0.80, proven 0.900 LB) + TWO-FRAME motion-synthetic gap bridge.

Independent card-free recall arm. The champion runs the density-gated motion-synthetic bridge at max_gap=1
(recover a cell missed for ONE frame). The frontier's 0.915 config runs a two-frame gap close
(`recover_strict_gap2`) — recover a cell the detector missed for TWO consecutive frames, where NO detection
sits at either intermediate frame. This flips `BridgeConfig.max_gap` 1 -> 2: the SAME density-adaptive radius
gates each inserted node, and the wider gap is only ever taken when the motion prediction still lands on a
single unambiguous track start (see DensityGapBridge docstring).

Why this is a distinct axis from every recall arm already on the LB (all 0.899-0.900):
  - threshold recall (thr0.80/0.85) lowers the DETECTION bar — more raw peaks. Orthogonal: gap-2 recovers
    cells with NO peak at any threshold across two frames, via motion prediction.
  - reuse gap-close (0.899, GapCloser reuse) reuses an existing isolated t+1 detection — a DIFFERENT
    mechanism (no invented node) and it only bridges ONE frame.
  - This inserts a motion-predicted synthetic node across a two-frame dropout. Never LB-tested.

Gate on MECHANISM (bridge fires at sane counts, bounded by max_added_fraction=0.05), not proxy — the proxy
blinds/inverts above 0.90 and this adds unlabeled-tissue recall the sparse 4-movie proxy cannot score.
Everything else is byte-identical to the 0.900 champion so the delta isolates the two-frame bridge.
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
pilkwang_packs = KernelRuntime.pilkwang_packs
test_videos = KernelRuntime.test_videos

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.postproc.gap_closer import BridgeConfig  # noqa: E402

_shipped = TrackerConfig.shipped()
_CONFIG = dataclasses.replace(_shipped, threshold=0.80, bridge=BridgeConfig(max_gap=2))
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    MultiGpuSubmission(packs=pilkwang_packs(), config=_CONFIG).run(test_videos(), _SUBMISSION)


if __name__ != "__mp_main__":
    main()

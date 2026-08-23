"""HIGH-THRESHOLD (thr0.99) dual-seed flow + TWO-FRAME density-gated recovery bridge.

Instrument arm for the 0.027-gap COUPLING hypothesis (see memory
celltrack-0027-carrier-is-deepcenter-recovery-stack). Primary source: the public 0.915 pipeline runs
DET_THRESHOLD=0.99 (drops many cells for precision) COUPLED with a DeepCenter-confirmed recovery stack
that adds the dropped true cells back. The hmlo replicate ran the high threshold (0.96875) but NO
recovery -> 0.888 (below champion): high threshold alone LOSES recall.

This arm runs the high threshold WITH our density-gated two-frame bridge (max_gap=2), but WITHOUT the
DeepCenter confirmer -- to isolate whether density-gate confirmation SUFFICES at high threshold, or
whether the DeepCenter second-opinion is essential:
  - thr0.99+gap2 >= 0.900  -> density-gate recovery works at high threshold; cheap win, no DeepCenter build.
  - thr0.99+gap2  < 0.900  -> high-threshold base drops cells the density gate cannot confirm-recover;
                              DeepCenter confirmer is essential -> build the full recovery arm (Option 3).
  - thr0.99+gap2 ~= 0.915  -> full frontier recall recovered card-free.

The local 4-movie proxy is BLIND to this axis (recovers unlabeled-tissue recall it cannot score, and
inverts above 0.90), so this is gated on the LB read, not the proxy. Only threshold + bridge.max_gap
differ from the 0.900 champion, so the delta isolates the high-threshold + two-frame-recovery coupling.
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
_CONFIG = dataclasses.replace(_shipped, threshold=0.99, bridge=BridgeConfig(max_gap=2))
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    MultiGpuSubmission(packs=pilkwang_packs(), config=_CONFIG).run(test_videos(), _SUBMISSION)


if __name__ != "__mp_main__":
    main()

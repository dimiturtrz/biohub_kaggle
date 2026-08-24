"""HIGH-THRESHOLD (thr0.99) dual-seed flow + DeepCenter-CONFIRMED recovery stack.

Instrument arm for the 0.027-gap COUPLING hypothesis (memory celltrack-0027-carrier-is-deepcenter-
recovery-stack). Primary source: the public 0.915 pipeline runs DET_THRESHOLD=0.99 (drops many cells for
precision) COUPLED with a DeepCenter-confirmed recovery stack that adds dropped true cells back. hmlo ran
the high threshold with NO recovery -> 0.888 (recall lost). Option 2 (this arm's sibling) runs thr0.99 with
DENSITY-gated recovery but NO DeepCenter confirmer, to isolate whether the second-opinion is essential.

This arm runs thr0.99 + the two-frame density bridge (max_gap=2) + the ONE-frame GapCloser with the
DeepCenter CenterConfirmer gating synthetic insertion (reuse=ReuseConfig(veto=CenterVetoConfig(),
synthetic=SyntheticGap())) -- the card-free replica of the frontier's confirmed recovery stack:
  - vs Option 2: the delta is the DeepCenter confirmer on synthetic-insertion (density-confirm vs center-confirm).
  - >= 0.900 and > Option 2 -> DeepCenter second-opinion recovers cells the density gate cannot -> essential.
  - ~= 0.915 -> full frontier recall recovered card-free.

The local 4-movie proxy is BLIND to this axis (recovers unlabeled-tissue recall it cannot score, inverts
above 0.90), so this is gated on the LB read. Requires the DeepCenter dataset attached; the runtime locates
it by the pack's own file, falling back to the shipped path (no confirmed recovery) if absent.
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
center_prior_pack = KernelRuntime.center_prior_pack
test_videos = KernelRuntime.test_videos

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

from celltrack.detectors.center_prior import CenterVetoConfig  # noqa: E402
from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.postproc.gap_closer import BridgeConfig, ReuseConfig, SyntheticGap  # noqa: E402

_shipped = TrackerConfig.shipped()
_CONFIG = dataclasses.replace(
    _shipped,
    threshold=0.99,
    bridge=BridgeConfig(max_gap=3),
    reuse=ReuseConfig(veto=CenterVetoConfig(), synthetic=SyntheticGap()),
)
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    MultiGpuSubmission(packs=pilkwang_packs(), config=_CONFIG, center_pack=center_prior_pack()).run(
        test_videos(), _SUBMISSION
    )


if __name__ != "__mp_main__":
    main()

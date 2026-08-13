"""Faithful replication of the reproducible public 0.915 bundle — the ONE measurement that collapses the
six piecemeal ports (bead hmlo). Dual-seed detector + two-pass MOTION linker + the CC0 local-association
re-ranker + their output repairs + safe divisions, all at THEIR constants rather than ours.

Every piece already existed here and was tested one-at-a-time in OUR cost, each LB-flat-or-negative — the
recurring ambiguity being "the part is weak" vs "our harness lacks something else the part depends on". This
assembles them TOGETHER at their constants so a single leaderboard number decides it. The proxy cannot
arbitrate this (it reads the motion linker, the ranker, the threshold and the appearance-side knobs blind or
anti-transfer, as it read our own recall gains low), so this is an LB-only test by construction.

The constants: motion two-pass tight 6um / loose 10um (VELOCITY_DAMPING already 0.5 = their velocity weight);
their learned-P weight 0.75 split 85/15 ranker/edge -> ranker_bonus 0.6375, affinity_bonus 0.1125; threshold
0.96875; TopologyConfig output repairs (enforce-next-frame, single-parent, prune-isolated, edge_max 14um);
safe divisions carried by shipped(). The ranker artifact is the published CC0 MLP, mounted per shard.
"""

import dataclasses
import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Bootstrap: find the mounted kit by a known module, path it, hand off to kernel_runtime.
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

from celltrack.linkers.linkers import LinkerConfig  # noqa: E402
from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.postproc.topology_repair import TopologyConfig  # noqa: E402

# The frontier bundle at THEIR constants. shipped() supplies the dual-seed detector, safe divisions, smoothing
# and min-track; this replaces the linker with the two-pass motion linker carrying the re-ranker and edge
# probability at their 85/15 split, adds their output repairs, and drops the threshold to their 0.96875.
_shipped = TrackerConfig.shipped()
_CONFIG = dataclasses.replace(
    _shipped,
    linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0, ranker_bonus=0.6375, affinity_bonus=0.1125),
    topology=TopologyConfig(),
    threshold=0.96875,
    min_track_length=7,
)
# The CC0 re-ranker artifact mounts here; MultiGpuSubmission mounts it per shard via CellTracker.with_ranker.
_RANKER = Path(next(iter(glob.glob("/kaggle/input/**/ASSOCIATION_RANKER_MANIFEST.json", recursive=True)))).parent
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    MultiGpuSubmission(packs=pilkwang_packs(), config=_CONFIG, ranker_pack=_RANKER).run(test_videos(), _SUBMISSION)


# Spawned per-GPU workers re-enter this module as `__mp_main__`; exclude exactly that so a child launches no
# submission of its own without assuming the host runs this file as `__main__`.
if __name__ != "__mp_main__":
    main()

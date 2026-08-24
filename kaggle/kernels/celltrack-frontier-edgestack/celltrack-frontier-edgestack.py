"""hmlo + the frontier EDGE stack — the one piece the 0.888 faithful-motion-ranker replication left off.

Bead hmlo assembled the frontier bundle (dual-seed detector + two-pass motion linker + CC0 re-ranker at
85/15 + output repairs + safe divisions, all at their constants) and scored 0.888 on the board — below the
0.900 champion. hmlo ran the DEFAULT edge path though: equal-mean dual-seed blend, no view-TTA, no
moment-alignment. The published 0.915 bundle's remaining, un-replicated levers all live on the edge side:

  - EDGE_TTA (view_tta): each gap re-scored over 4 flip views, fused by JS-reliability log-opinion pool
    re-centred on identity. Down-weights the views a gap disagrees with -> SHARPER affinity exactly on the
    dense affinity-confusor (the measured 85%-of-gap bottleneck), where a wrong near neighbour out-scores
    the true successor. This is the direction the confusor needs; naive averaging (which a proxy arm once
    read as -0.0049) is the opposite and is not what the reliability pool does.
  - align_seed_moments: the secondary seed's edge logits are mean/std-aligned to the primary before the
    convex blend, so a per-seed scale gap does not leak into the fused probability.
  - the blend weights themselves: SECONDARY_EDGE_WEIGHT 0.15 -> edge_blend (0.85, 0.15);
    SECONDARY_DETECTION_WEIGHT 0.475 -> detector_blend 0.525 (seed1's share = 1 - 0.475).

Everything else is hmlo's config verbatim, so a single leaderboard number isolates the edge stack as the
delta from the known 0.888 base. GAP acknowledged: the frontier's secondary detector is an 8-view TTA
mean/std-aligned to primary before the 0.475 blend; our pipeline has no 8-view secondary detector TTA, so
detector_blend approximates only the seed weighting, not that TTA. The load-bearing, confusor-facing levers
(edge view_tta + align_seed_moments) ARE fully replicated. The proxy cannot arbitrate this (it reads the
appearance-side knobs blind or anti-transfer, as it read our own recall gains low) -> LB-only by construction.
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

from celltrack.edges.blended_edge_scoring import EdgeBlendOptions  # noqa: E402
from celltrack.linkers.linkers import LinkerConfig  # noqa: E402
from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.postproc.topology_repair import TopologyConfig  # noqa: E402

# hmlo's config verbatim (motion two-pass, ranker 85/15 at 0.6375/0.1125, thr 0.96875, output repairs, safe
# divisions from shipped()), PLUS the frontier edge stack: view-TTA + seed-moment alignment, and the blend
# weights those levers ship with. edge_blend (seed1, seed2) with seed2 = SECONDARY_EDGE_WEIGHT 0.15;
# detector_blend = seed1's share = 1 - SECONDARY_DETECTION_WEIGHT 0.475 = 0.525.
_shipped = TrackerConfig.shipped()
_CONFIG = dataclasses.replace(
    _shipped,
    linker=LinkerConfig(
        name="motion",
        tight_um=6.0,
        gate_um=10.0,
        ranker_bonus=0.6375,
        affinity_bonus=0.1125,
    ),
    topology=TopologyConfig(),
    threshold=0.96875,
    edge_blend=(0.85, 0.15),
    detector_blend=0.525,
    edge_options=EdgeBlendOptions(view_tta=True, align_seed_moments=True),
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

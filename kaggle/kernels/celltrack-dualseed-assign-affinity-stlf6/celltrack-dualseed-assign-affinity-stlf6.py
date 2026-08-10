"""Dual-seed detector + global 1-to-1 assignment linker (edge-transformer affinities) + stlf6 + linefit.

The frontier's 0.913 linker core, ours, with no SCIP. The prob-driven ILP's shipped configuration —
MaxParents(1) + MaxChildren(1) — decomposes into an independent min-cost bipartite matching per frame gap, so
`AssignmentLinker` (pure scipy `linear_sum_assignment` + a zero-cost skip per node) selects the identical edge
set the motile/ilpy/SCIP program did, verified edge-for-edge on the dense test movie (jaccard 1.0). This drops
the offline SCIP wheel stack for numpy and scipy already in the base image. Edge cost per candidate =
distance - bonus * P(s->t); an edge is taken only where the learned association beats its distance.

The kit-mount, pack-discovery and wheel-install ceremony lives in `celltrack.kernel_runtime`, and the mounting
plus per-GPU execution of the submission in `celltrack.multi_gpu_submission`; this file is the bootstrap plus
the one operating point it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Bootstrap (must run before celltrack's dependencies are installed): find the mounted kit by a known module,
# put it on the path, then hand off to celltrack.kernel_runtime for the rest.
_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/tracker.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import KernelRuntime  # noqa: E402

# kernel_runtime's steps are @staticmethods on KernelRuntime (the arch gate forbids top-level functions);
# alias them to the free names the kernel body uses so the bootstrap stays a flat script.
install_wheels = KernelRuntime.install_wheels
pack_source = KernelRuntime.pack_source
pilkwang_packs = KernelRuntime.pilkwang_packs
test_videos = KernelRuntime.test_videos

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

from celltrack.linkers.linkers import LinkerConfig  # noqa: E402
from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.tracker import TrackerConfig  # noqa: E402

# thr 0.97 is the measured LB best of the ladder probed so far (0.99/0.98/0.97 = 0.887/0.891/0.892).
_THRESHOLD = 0.97
_GATE_UM = 10.0
_EDGE_BONUS = 20.0
_EDGE_BLEND = (0.8, 0.2)
# The flow solver's track-boundary charge. At 0 it provably reduces to the assignment linker edge-for-edge;
# 3 is where it behaves globally and is its measured proxy peak.
_BOUNDARY_COST = 3.0
# CONSOLIDATES THE CONFIRMED WIN ONTO THE CONFIRMED BASE. v23 (global flow + bidirectional fusion) scored
# LB 0.895 — our best — but it was anchored on the min7+rescue base, which separately measured 0.891, i.e.
# 0.001 BELOW the min6 base. This is the same config with that handicap removed: min_track_length back to 6
# and no rescue, so the only difference from v23 is the base it stands on.
#
# The fusion is kept because the LB, not the proxy, arbitrated it: alone under the per-frame assignment linker
# it measured -0.0027, but a symmetric mutual-consistency affinity suits a GLOBAL solver that imposes
# one-parent/one-child itself rather than through the probability's normalisation. Proxy showed the sign flip;
# the leaderboard confirmed it at +0.004.
_CONFIG = TrackerConfig(
    threshold=_THRESHOLD,
    edge_blend=_EDGE_BLEND,
    linker=LinkerConfig(
        name="flow", gate_um=_GATE_UM, affinity_bonus=_EDGE_BONUS, disappearance_cost=_BOUNDARY_COST
    ),
    bidirectional_edges=True,
)
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    # Kaggle gives this kernel two T4s and the videos are independent, so MultiGpuSubmission shards them over
    # every visible device and runs the identical ephemeral-store tracker per shard — same weights, same config,
    # same per-video result, ~Nx the throughput. With one device it stays in-process.
    MultiGpuSubmission(packs=pilkwang_packs(), config=_CONFIG).run(test_videos(), _SUBMISSION)


# The per-GPU workers are spawned (CUDA cannot be forked). A spawn child that re-enters this module does so
# under the name `__mp_main__`, so excluding exactly that name stops a child from launching a submission of its
# own without assuming the host runs this file as `__main__` (a notebook host may not).
if __name__ != "__mp_main__":
    main()

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

# 0.97: stack the two independent recall levers the leaderboard rewards. Threshold is a recall lever the sparse
# proxy cannot see — public 0.99/0.98/0.97 = 0.887/0.891/0.892, matching the disclosed clean-baseline ~0.96875;
# the 83r NMS 3^3 window fix un-merges crowded cells (more raw-Jaccard recall on the dense bottleneck). Both are
# recall, so the NMS fix at 0.97 combines them; A/B is vs 55240719 (thr0.97 pre-NMS-fix = 0.892).
_THRESHOLD = 0.97
# The gate is the maximum single-frame cell travel: annotated-edge displacement maxes at 9.96um across the four
# movies (p99.9 = 9.78), so 10um admits every true successor; the proxy is flat 0.929-0.930 over gate 10-15
# (bead i0a). bonus = 2·gate: a certain association must overpower up to two gate-widths where the nearest-
# distance prior is anti-informative (the crowding-mislink); the proxy peaks broadly over 2-3·gate (bead 88x).
# Gate at the derived physical value 10um (max annotated single-frame travel 9.96um, bead i0a) — the confirmed
# best; the earlier gate14 LB probe tied at 0.892 (admission-widening is flat, transfer law), so it earns no
# keep and we restore the derived value.
_GATE_UM = 10.0
_EDGE_BONUS = 20.0
# Two edge-transformer seeds blended in logit space; 0.8·seed1 + 0.2·seed2 is the proxy peak (bead 88x).
_EDGE_BLEND = (0.8, 0.2)
# The CONFIRMED-BEST recipe (thr0.97, gate10, bonus20, min6, smooth0.8 = LB 0.892), restored deliberately:
# this run is an INFRASTRUCTURE control, not a score probe. It is the first submission carrying the fp16
# autocast (proxy-verified score-identical, 0.9334 either way) and the multi-GPU dispatcher (exact by
# construction — same code, same weights, one device per video). Holding the config at a known LB anchor is
# what makes the run interpretable: 0.892 confirms both are score-neutral on the hidden set and the kernel log
# gives the real T4 speedup, while any other number indicts the infrastructure rather than a tracking knob.
# The min7+rescue probe is already in flight on its own submission and is not duplicated here.
_CONFIG = TrackerConfig(
    threshold=_THRESHOLD,
    edge_blend=_EDGE_BLEND,
    linker=LinkerConfig(name="assignment", gate_um=_GATE_UM, affinity_bonus=_EDGE_BONUS),
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

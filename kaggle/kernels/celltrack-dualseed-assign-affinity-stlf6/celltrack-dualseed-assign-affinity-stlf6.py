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

# 0.96 EXTENDS the recall ladder past where we stopped probing. The measured LB curve is monotonic as the
# threshold FALLS — 0.99 -> 0.887, 0.98 -> 0.891, 0.97 -> 0.892 — and we halted at 0.97 only because that is
# where probing began, not because it turned over. The disclosed clean public baseline runs 0.96875, BELOW our
# best, so the frontier is operating further down this same curve. Threshold is the one lever confirmed to
# transfer here, and it is proxy-BLIND (the sparse proxy reads it flat-to-inverted, preferring 0.995 which the
# LB refuted at 0.880), so a slot is the only instrument that can answer it. Serves bead 26l.
_THRESHOLD = 0.96
# The gate is the maximum single-frame cell travel: annotated-edge displacement maxes at 9.96um across the four
# movies, so 10um admits every true successor; bonus = 2*gate is the crowding argument (bead 88x/i0a).
_GATE_UM = 10.0
_EDGE_BONUS = 20.0
# Two edge-transformer seeds blended in logit space; 0.8*seed1 + 0.2*seed2 is the proxy peak (bead 88x).
_EDGE_BLEND = (0.8, 0.2)
# The CONFIRMED-BEST base otherwise (min6, no rescue, assignment linker, smooth0.8 = LB 0.892), so the
# threshold is the only variable. min7+rescue has since scored 0.891, i.e. below this base — hence min6.
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

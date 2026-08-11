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
from dataclasses import replace
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

from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig  # noqa: E402

# THE DIVISION TERM, CONTESTED FOR THE FIRST TIME. score = adjusted_edge_jaccard + 0.1 * division_jaccard and
# we have banked exactly zero of the second term all campaign — the flow solver's capacity-1 node split emits no
# forks at all, so division_jaccard is 0 BY CONSTRUCTION rather than by any measured limit.
#
# The proxy cannot arbitrate this axis: it holds THREE annotated divisions, so any fork budget faces a hostile
# FP:TP ratio the hidden set does not (our corpus rate is 0.113% of annotated nodes, and a fully annotated dense
# movie would hold ~79-170 divisions rather than 3). What the proxy CAN read is the fork count, whether the one
# recoverable division survives, and the cost to the edge term — measured at -0.0019, inside the noise floor.
#
# Arm J of the sweep: rank by SPLIT SYMMETRY. |u+v| / (|u|+|v|) over the two daughter displacements is the
# half-angle form, 0 for a clean opposite split and 1 for two cells leaving together; the cost adds it to
# parent_dist / parent_gate so both terms are normalised by the constraint that decides their own admissibility
# and equal weighting is the only unfitted choice. A/B against the frontier's parent + 0.15*sister at IDENTICAL
# gates and an identical 490 candidates: symmetry lifts the recoverable division into the kept set and costs
# -0.0019; the frontier's geometry misses it and costs -0.0036.
#
# The probability floors are OFF (0.0), which is what the sweep found actually blocked the recovery — not the
# gates and not the cap. Under this pipeline the true divider's kept-child probability is below 0.5, so at the
# shipped floor it was never even PROPOSED. Candidacy is carried by geometry; the absolute cap bounds the bet.
_DIVISION = AffinityDivisionConfig(
    ranking="symmetry",
    min_second_prob=0.0,
    min_kept_prob=0.0,
    parent_gate_um=7.0,
    sister_gate_um=14.0,
    max_added_forks=300,
)
# The base is MOUNTED, not restated: `TrackerConfig.shipped()` is the leaderboard-arbitrated 0.895 recipe
# (thr 0.97, flow linker gate 10 / bonus 20 / boundary 3, bidirectional fusion, min6, smooth 0.8), and it lives
# in the library so the selector and the submission cannot drift apart. This kernel adds exactly one variable.
_CONFIG = replace(TrackerConfig.shipped(), division=_DIVISION)
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

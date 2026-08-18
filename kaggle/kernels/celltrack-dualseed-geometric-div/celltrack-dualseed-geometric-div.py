"""Dual-seed detector + global 1-to-1 assignment linker + geometric division candidacy.

Same 0.899 base as `celltrack-dualseed-assign-affinity-stlf6` (edge-transformer affinities, assignment linker, no
SCIP), changing ONE field: the division stage's candidacy from head-gated `runner_up` to `geometric`.

The shipped `runner_up` candidacy proposes a fork only where the orphan is the edge head's SECOND-best target and
its best is the kept child. That is a confident-parent test — it recovers a sister only when the softmax already
ranks her second. In dense tissue the true sister is frequently NOT the head's #2: a nearer false neighbour
outranks her, so she is never proposed and the `+0.1 * division_jaccard` term is forfeited on exactly the movies
that have the most divisions. `geometric` (the public frontier's `DivisionAwareLinker` candidacy) proposes EVERY
unparented target near a single-child parent instead, so a true sister the dense softmax mis-ranks is still handed
to the SAME symmetry ranking and the SAME derived gates to judge. Candidacy widens the proposal set; nothing about
how a fork is scored or admitted changes.

The proxy CANNOT arbitrate this axis — it holds three annotated divisions, an FP:TP ratio 30-50x the hidden
denser set's, so a wider candidacy reads as pure false positives there while the hidden set's ~79-170 divisions
per dense movie are where the recovered sisters live. This is a MECHANISM change (which orphans are candidates),
not a threshold knob, so it is the one kind of division move the frozen 0.899 model still has room to make. LB
arbitrates; the proxy can only confirm the flag landed (geometric proposes a superset of runner_up's forks).

The kit-mount, pack-discovery and wheel-install ceremony lives in `celltrack.kernel_runtime`, and the mounting
plus per-GPU execution of the submission in `celltrack.multi_gpu_submission`; this file is the bootstrap plus
the one operating point it tests.
"""

import dataclasses
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

from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402

# The one changed field. `AffinityDivisionConfig` is pydantic-frozen, so the override is `model_copy(update=...)`,
# not `dataclasses.replace` (which reaches only the outer `TrackerConfig` dataclass). Everything else — ranking,
# gates, budget, threshold 0.97 — is the proven 0.899 shipped default; candidacy alone moves.
_shipped = TrackerConfig.shipped()
_CONFIG = dataclasses.replace(_shipped, division=_shipped.division.model_copy(update={"candidacy": "geometric"}))
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

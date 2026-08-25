"""FRONTIER-REPLICA: the genuine 0.926 cell-tracking stack wired into one kernel-generated submission.

This is the assembly of the five arms that reproduce the published 0.926 pipeline, each already built in the
`celltrack` package and turned ON together by `TrackerConfig.kernel_faithful_replica()` plus the mounted packs.
The faithful (not variant) factory is used deliberately: the six audited divergences from the published kernel
are all flipped to the kernel's own behaviour (per-column edge z-cal, continuous consensus ramp, the 0.48 edge
admission floor, and the per-frame detection retention guard), so the published-0.926 precedent and the
fidelity oracle's byte-exact divsub/motion match both apply to exactly the code this kernel runs:

  1. Calibrated dual-seed FUSION — the frontier's #1 edge lever: the primary head's logits fused with an
     independently-seeded secondary (`seed314159`) only on low-margin, both-heads-agree targets, and the
     detector blend std-ratio-aligned to give that secondary the frontier's 0.475 weight.
  2. DeepCenter VETO — the confirmed-recovery centre prior, mounted via `center_pack`, gating synthetic
     insertion so the recall stack adds back only cells a second opinion confirms.
  3. divsub C3 safe divisions — the recovery stage hardened with the public kernel's mutual-nearest and t+2
     post-mitotic divergence gates and its global fork ceiling, turning hundreds of speculative forks into the
     few real ones.
  4. Motion RELINK — the two-pass tight/loose motion-Hungarian linker at the annotated one-frame displacement
     bounds, in place of the shipped global-flow linker.
  5. A detection THRESHOLD (0.96875) between the shipped 0.97 and the high-precision 0.99.

The local 4-movie proxy is BLIND to this stack (it recovers unlabelled-tissue recall it cannot score and
inverts above 0.90), so this ships gated on the fidelity oracle (the divsub + motion arms must reproduce the
kernel's own decisions) and on the published 0.926 precedent — never on a proxy number. The secondary and
DeepCenter datasets are located by their own marker files; absent either, the runtime falls back to the
single-seed / no-confirmed-recovery path rather than failing on a mount layout.
"""

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
secondary_pack = KernelRuntime.secondary_pack
test_videos = KernelRuntime.test_videos

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

from celltrack.multi_gpu_submission import MultiGpuSubmission  # noqa: E402
from celltrack.operating_point import TrackerConfig  # noqa: E402

_CONFIG = TrackerConfig.kernel_faithful_replica()
_SUBMISSION = Path("/kaggle/working/submission.csv")


def main() -> None:
    MultiGpuSubmission(
        packs=pilkwang_packs(),
        config=_CONFIG,
        center_pack=center_prior_pack(),
        secondary_pack=secondary_pack(),
    ).run(test_videos(), _SUBMISSION)


if __name__ != "__mp_main__":
    main()

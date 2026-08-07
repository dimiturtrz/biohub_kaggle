"""Dual-seed detector + global 1-to-1 assignment linker (edge-transformer affinities) + stlf6 + linefit.

The frontier's 0.913 linker core, ours, with no SCIP. The prob-driven ILP's shipped configuration —
MaxParents(1) + MaxChildren(1) — decomposes into an independent min-cost bipartite matching per frame gap, so
`AssignmentLinker` (pure scipy `linear_sum_assignment` + a zero-cost skip per node) selects the identical edge
set the motile/ilpy/SCIP program did, verified edge-for-edge on the dense test movie (jaccard 1.0). This drops
the offline SCIP wheel stack for numpy and scipy already in the base image. Edge cost per candidate =
distance - bonus * P(s->t); an edge is taken only where the learned association beats its distance.

The kit-mount, pack-discovery, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`;
this file is the bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Bootstrap (must run before celltrack's dependencies are installed): find the mounted kit by a known module,
# put it on the path, then hand off to celltrack.kernel_runtime for the rest.
_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/motion_linking.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import (  # noqa: E402
    install_wheels,
    pack_source,
    pilkwang_packs,
    run_submission,
    test_videos,
)

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

import torch  # noqa: E402

from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer  # noqa: E402
from celltrack.linkers.linkers import LinkerConfig  # noqa: E402
from celltrack.detectors.pipeline import BlendDetectorScorer  # noqa: E402
from celltrack.detectors.response_cache import EphemeralResponseStore  # noqa: E402
from celltrack.tracker import CellTracker, TrackerConfig  # noqa: E402
from celltrack.detectors.tunet import TemporalUNetDetector  # noqa: E402

# 0.97: stack the two independent recall levers the leaderboard rewards. Threshold is a recall lever the sparse
# proxy cannot see — public 0.99/0.98/0.97 = 0.887/0.891/0.892, matching the disclosed clean-baseline ~0.96875;
# the 83r NMS 3^3 window fix un-merges crowded cells (more raw-Jaccard recall on the dense bottleneck). Both are
# recall, so the NMS fix at 0.97 combines them; A/B is vs 55240719 (thr0.97 pre-NMS-fix = 0.892).
_THRESHOLD = 0.97
# The gate is the maximum single-frame cell travel: annotated-edge displacement maxes at 9.96um across the four
# movies (p99.9 = 9.78), so 10um admits every true successor; the proxy is flat 0.929-0.930 over gate 10-15
# (bead i0a). bonus = 2·gate: a certain association must overpower up to two gate-widths where the nearest-
# distance prior is anti-informative (the crowding-mislink); the proxy peaks broadly over 2-3·gate (bead 88x).
# Single-variable LB probe on the confirmed-best base (thr0.97, bonus20, min6, smooth0.8 = LB 0.892): the one
# changed knob is the linker admission GATE, 10 -> 14um. It admits division/fast-mover edges beyond the proxy
# GT's 9.96um max displacement that the denser hidden set may annotate. Yusuke's LB-0.897 used a 14um edge
# gate -> a real prior; the proxy is flat over 10-15 (no proxy GT edge exceeds 10) so only the LB can judge.
# This is a LINKING-side change (distinct from the refuted recall-side tweaks). bonus HELD at 20 (= 2·10, the
# swept peak) so the A/B isolates the gate alone; smooth_strength defaults to the LB-confirmed 0.8.
_GATE_UM = 14.0
_EDGE_BONUS = 20.0
# Two edge-transformer seeds blended in logit space; 0.8·seed1 + 0.2·seed2 is the proxy peak (bead 88x).
_EDGE_BLEND = (0.8, 0.2)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CONFIG = TrackerConfig(
    threshold=_THRESHOLD,
    edge_blend=_EDGE_BLEND,
    linker=LinkerConfig(name="assignment", gate_um=_GATE_UM, affinity_bonus=_EDGE_BONUS),
    min_track_length=6,
)


def main() -> None:
    pack1, pack2 = pilkwang_packs()
    det1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=_DEVICE)
    det2, _ = TemporalUNetDetector.from_pack(pack2, map_location=_DEVICE)
    # A submission forwards each video once and reads it out once; an EphemeralResponseStore keeps the fp32
    # logits in memory for that single NMS read and drops them, so the gigabyte-per-seed volumes never touch
    # the kernel's bounded /kaggle/working disk (the disk-full failure the caching store caused here).
    detector = BlendDetectorScorer(
        detectors=(
            (det1.to(_DEVICE).eval(), EphemeralResponseStore()),
            (det2.to(_DEVICE).eval(), EphemeralResponseStore()),
        ),
        recipe=recipe,
        device=_DEVICE,
    )
    edge_scorer = BlendedEdgeTransformerScorer.from_packs((pack1, pack2), _EDGE_BLEND, _DEVICE)
    tracker = CellTracker(detector=detector, edge_scorer=edge_scorer, device=_DEVICE, config=_CONFIG)
    run_submission(tracker.run, test_videos())


main()

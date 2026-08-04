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

from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer  # noqa: E402
from celltrack.linkers import LinkerConfig  # noqa: E402
from celltrack.pipeline import BlendDetectorScorer  # noqa: E402
from celltrack.response_cache import ResponseCache  # noqa: E402
from celltrack.tracker import CellTracker, TrackerConfig  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402

# 0.99: the known-good threshold (this exact config at 0.99 = LB 0.887, sub 55218808), held fixed so this
# submission is a clean A/B isolating the reference NMS-window fix (bead 83r: 3^3 suppression, not 5^3, which
# un-merges crowded cells — proxy 0.9227 -> 0.9344, all raw-Jaccard recall on the dense movie). Threshold
# recall (0.98/0.97) is a separate LB-only lever probed by other subs; keep one variable moving at a time.
_THRESHOLD = 0.99
# The gate is the maximum single-frame cell travel: annotated-edge displacement maxes at 9.96um across the four
# movies (p99.9 = 9.78), so 10um admits every true successor; the proxy is flat 0.929-0.930 over gate 10-15
# (bead i0a). bonus = 2·gate: a certain association must overpower up to two gate-widths where the nearest-
# distance prior is anti-informative (the crowding-mislink); the proxy peaks broadly over 2-3·gate (bead 88x).
_GATE_UM = 10.0
_EDGE_BONUS = 2.0 * _GATE_UM
# Two edge-transformer seeds blended in logit space; 0.8·seed1 + 0.2·seed2 is the proxy peak (bead 88x).
_EDGE_BLEND = (0.8, 0.2)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CONFIG = TrackerConfig(
    threshold=_THRESHOLD,
    edge_blend=_EDGE_BLEND,
    linker=LinkerConfig(name="assignment", gate_um=_GATE_UM, affinity_bonus=_EDGE_BONUS),
)


def main() -> None:
    pack1, pack2 = pilkwang_packs()
    det1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=_DEVICE)
    det2, _ = TemporalUNetDetector.from_pack(pack2, map_location=_DEVICE)
    detector = BlendDetectorScorer(
        detectors=(
            (det1.to(_DEVICE).eval(), ResponseCache(Path("/kaggle/working/cache"), "seed1")),
            (det2.to(_DEVICE).eval(), ResponseCache(Path("/kaggle/working/cache"), "seed2")),
        ),
        recipe=recipe,
        device=_DEVICE,
    )
    edge_scorer = BlendedEdgeTransformerScorer.from_packs((pack1, pack2), _EDGE_BLEND, _DEVICE)
    tracker = CellTracker(detector=detector, edge_scorer=edge_scorer, device=_DEVICE, config=_CONFIG)
    run_submission(tracker.run, test_videos())


main()

"""Dual-seed detector + learned edge-transformer bonus in the motion linker (+ stlf6 + linefit).

Extends `celltrack-dualseed-t99-motion-stlf6` with the one change that moved the number most this campaign:
the pilkwang `SimpleNodeTransformer` edge head (the unused `transformer.*` weights already in the support
pack) scores every source->target candidate, and the motion linker's within-gate cost becomes
`distance - bonus * P(s->t)` — geometry still gates, the learned association breaks the crowding ties raw
distance mis-picks. bonus = loose gate (10 um): a certain learned link overrides any within-gate distance.

Dual-seed 4-test-movie proxy (the LB proxy): geometry 0.8876 -> +edge-bonus 0.8989 (+0.0113), all on the
crowded movies (6bba_05db0fb1 0.816->0.828, 44b6_0b24845f +0.057). Detector blend, linker, stlf6 and linefit
are the shipped fold-0 code path; only the linker cost gains the learned term.

The kit-mount, pack-discovery, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`;
this file is the bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from celltrack.edges.edge_scoring import EdgeTransformerScorer  # noqa: E402
from celltrack.postproc.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linkers.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.detectors.tunet import TemporalUNetDetector  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_THRESHOLD = 0.99
_TIGHT_GATE_UM, _LOOSE_GATE_UM = 6.0, 10.0
_EDGE_BONUS = _LOOSE_GATE_UM  # a certain learned link overrides any within-gate distance
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _spacing(path: Path) -> Spacing:
    """The video's physical voxel spacing, read from its OME coordinate transform."""
    scale = zarr.open_group(path, mode="r").attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    return Spacing(z=float(scale[1]), y=float(scale[2]), x=float(scale[3]))


def _blend_nodes(det1: TemporalUNetDetector, det2: TemporalUNetDetector, recipe: object, path: Path) -> TrackGraph:
    """Average the two seeds' per-frame logits on the device, sigmoid, and read the peaks into an edge-free graph."""
    logits1 = det1.logit_volumes(path, recipe, _DEVICE)
    logits2 = det2.logit_volumes(path, recipe, _DEVICE)
    blended = [
        torch.sigmoid(torch.as_tensor(np.stack([a, b]), device=_DEVICE).mean(dim=0)).cpu().numpy()
        for a, b in zip(logits1, logits2, strict=True)
    ]
    return TemporalUNetDetector.graph_from_volumes(blended, det1.voxel_scale(path), _THRESHOLD, recipe, _DEVICE)


def main() -> None:
    pack1, pack2 = pilkwang_packs()
    det1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=_DEVICE)
    det2, _ = TemporalUNetDetector.from_pack(pack2, map_location=_DEVICE)
    det1, det2 = det1.to(_DEVICE).eval(), det2.to(_DEVICE).eval()
    edge_scorer = EdgeTransformerScorer.from_pack(pack1, _DEVICE)
    short, smooth = ShortTrackFilter(min_length=6), LinefitSmoother(strength=0.8)

    def predict(_name: str, path: Path) -> TrackGraph:
        nodes = _blend_nodes(det1, det2, recipe, path)
        affinity = edge_scorer.affinities(path, nodes, _DEVICE)
        linker = MotionHungarianLinker(
            spacing=_spacing(path),
            tight_gate_um=_TIGHT_GATE_UM,
            loose_gate_um=_LOOSE_GATE_UM,
            affinity=affinity,
            affinity_bonus=_EDGE_BONUS,
        )
        return smooth.transform(short.transform(linker.link(nodes)))

    run_submission(predict, test_videos())


main()

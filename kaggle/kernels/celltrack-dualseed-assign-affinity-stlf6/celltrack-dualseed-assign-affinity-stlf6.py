"""Dual-seed detector + global 1-to-1 assignment linker (edge-transformer affinities) + stlf6 + linefit.

The frontier's 0.913 linker core, ours, with no SCIP. The prob-driven ILP's shipped configuration —
MaxParents(1) + MaxChildren(1) — decomposes into an independent min-cost bipartite matching per frame gap, so
`AssignmentLinker` (pure scipy `linear_sum_assignment` + a zero-cost skip per node) selects the identical edge
set the motile/ilpy/SCIP program did, verified edge-for-edge on the dense test movie (jaccard 1.0). This drops
the offline SCIP wheel stack — whose solver threw `SCIP: unspecified error!` inside the kernel — for numpy and
scipy already in the base image. Edge cost per candidate = distance - bonus * P(s->t); an edge is taken only
where the learned association beats its distance.

Dual-seed 4-test-movie proxy: 0.9125 (assignment) and 0.9134 (+ density gap-bridge), matching the ILP exactly.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

_INPUTS = os.listdir("/kaggle/input")
print("INPUT DIRS:", _INPUTS, flush=True)


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_CONFIGS = glob.glob("/kaggle/input/**/weights/unet_transformer/split_0/config.json", recursive=True)
_PACK1 = next(Path(c).parent for c in _CONFIGS if "support-pack" in c or "50ep" in c)
_PACK2 = next(Path(c).parent for c in _CONFIGS if "seed314159" in c)
_PACK_SRC = Path(_find("repo/src/biohub_tracking/models/__init__.py")).parents[2]
_MARKER = _find("celltrack/motion_linking.py")

_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
_WHEELS = sorted(glob.glob(f"{Path(_KIT_ROOT).parent}/**/*.whl", recursive=True))
print("PACKS:", _PACK1, _PACK2, flush=True)
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

sys.path.insert(0, _KIT_ROOT)
sys.path.insert(0, str(_PACK_SRC))

import torch  # noqa: E402

from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer  # noqa: E402
from celltrack.champion import ChampionConfig, ChampionPipeline  # noqa: E402
from celltrack.pipeline import BlendDetectorScorer  # noqa: E402
from celltrack.response_cache import ResponseCache  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
# 0.99: the known-good threshold (this exact config at 0.99 = LB 0.887, sub 55218808), held fixed so this
# submission is a clean A/B isolating the reference NMS-window fix (bead 83r: 3^3 suppression, not 5^3, which
# un-merges crowded cells — proxy 0.9227 -> 0.9344, all raw-Jaccard recall on the dense movie). Threshold
# recall (0.98/0.97) is a separate LB-only lever probed by other subs; keep one variable moving at a time.
_THRESHOLD = 0.99
# The gate is the maximum single-frame cell travel, not a free constant: across all four movies' annotated
# edges the displacement maxes at 9.96um (p99.9 = 9.78), so 10um admits every true successor with ~no margin
# waste; the proxy is flat 0.929-0.930 over gate 10-15 (bead i0a), 10 is the physical floor of that plateau.
_GATE_UM = 10.0
# A certain learned association (P=1) must overpower up to two gate-widths of distance: in dense tissue the
# true successor's displacement routinely exceeds one gate, so the nearest-distance prior is actively
# misleading (the crowding-mislink), and the edge transformer — the discriminative signal — has to win against
# it. bonus = 2·gate; the proxy peaks broadly over 2–3·gate (0.9154 at 1·gate → 0.9227 at 2·gate), declining
# past it as pure-association over-trusts confident-but-wrong links. A median-scale autobalance is bead 88x.
_EDGE_BONUS = 2.0 * _GATE_UM
# Two edge-transformer seeds blended in logit space; 0.8·seed1 + 0.2·seed2 is the proxy peak (0.9134→0.9154
# at bonus=1·gate; the sweep extends both together, bead 88x).
_EDGE_BLEND = (0.8, 0.2)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# The shipped recipe's operating point, carried on the one config object ChampionPipeline consumes; the
# remaining knobs (min_track_length=6, bridge_reach_um=10, smooth_strength=0.8) are its defaults.
_CHAMPION = ChampionConfig(threshold=_THRESHOLD, gate_um=_GATE_UM, edge_bonus=_EDGE_BONUS, edge_blend=_EDGE_BLEND)


def main() -> None:
    tests = sorted(glob.glob(_TEST_GLOB, recursive=True))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    det1, recipe = TemporalUNetDetector.from_pack(_PACK1, map_location=_DEVICE)
    det2, _ = TemporalUNetDetector.from_pack(_PACK2, map_location=_DEVICE)
    detector = BlendDetectorScorer(
        detectors=(
            (det1.to(_DEVICE).eval(), ResponseCache(Path("/kaggle/working/cache"), "seed1")),
            (det2.to(_DEVICE).eval(), ResponseCache(Path("/kaggle/working/cache"), "seed2")),
        ),
        recipe=recipe,
        device=_DEVICE,
    )
    edge_scorer = BlendedEdgeTransformerScorer.from_packs((_PACK1, _PACK2), _EDGE_BLEND, _DEVICE)
    pipeline = ChampionPipeline(detector=detector, edge_scorer=edge_scorer, device=_DEVICE, config=_CHAMPION)
    graphs = {}
    for path in tests:
        name = os.path.basename(path)[: -len(".zarr")]
        graph = pipeline.run(name, Path(path))
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()

"""Does the generator actually POSE the association confusor? — the mandatory pre-train gate.

The synthetic-training bet has one load-bearing precondition: the generated scenes must CONTAIN the case the
head fails on — a next-frame node sitting nearer the source than its true successor, so proximity points at the
wrong target (the `Scene.hard_fraction` definition). A generator that renders dense-LOOKING but separable scenes
trains the head on trivially-easy edges and P_true never moves; that is what `joint_confusor_synth_v1` did, with
a uniform-placement confusor rate of ~4% (card-free audit 2026-08-24).

This measures that rate on the EXACT batches the trainer consumes — `GpuScenes.batch`, the `--gpu-scene-fraction`
path — over a sweep of `confusor_rate`, so a run is never launched on an unaudited generator again. Run before
every synth-detector train; PASS = the rate at the run's `confusor_rate` clears the target (~0.30).

    python -m celltrack.data.confusor_audit --rates 0.0 0.15 0.3 0.45 --faithful
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace

import torch

from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.joint_dataset import PairSample
from celltrack.data.synthetic_scene import _SCENE_DOWNSAMPLE, FAITHFUL_APPEARANCE, SceneConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfusorAudit:
    """Measures the source-vs-target confusor rate on the exact `GpuScenes.batch` output over a `confusor_rate`
    sweep — the pre-train gate. `base` is the appearance/motion preset; each swept rate replaces its
    `confusor_rate` and nothing else, so the sweep isolates construction from every other scene knob."""

    base: SceneConfig
    gate_um: float
    batch: int
    seed: int

    @staticmethod
    def hard_fraction(sample: PairSample, spacing_um: float, gate_um: float) -> tuple[float, float]:
        """(confusor rate, median own step um) for one pair — source-vs-target, the `hard_fraction` definition:
        an in-gate edge is hard when some target j sits nearer source i than i's own successor does."""
        lift = torch.tensor([1, *_SCENE_DOWNSAMPLE], dtype=torch.float32)[1:]  # (z, y, x); undo the raw-grid lift
        source = sample.source_centres.float() / lift  # back to 64^3 isotropic voxels
        target = sample.target_centres.float() / lift
        distances = torch.cdist(source, target) * spacing_um  # (n, n) um, source i vs target j
        own = distances.diagonal()
        nearest = distances.argmin(dim=1)
        in_gate = own <= gate_um
        hard = in_gate & (nearest != torch.arange(len(own)))
        rate = hard.sum().item() / in_gate.sum().clamp_min(1).item()
        return rate, own[in_gate].median().item()

    def sweep(self, rates: list[float]) -> list[tuple[float, float, float]]:
        """(rate, confusor %, mean median-own-step um) per requested `confusor_rate`, over one shared generator
        so the sweep reads like a controlled experiment (only `confusor_rate` moves between rows)."""
        generator = torch.Generator().manual_seed(self.seed)
        rows: list[tuple[float, float, float]] = []
        for rate in rates:
            scenes = GpuScenes(config=replace(self.base, confusor_rate=rate), device="cpu")
            stats = [
                self.hard_fraction(s, self.base.spacing_um, self.gate_um) for s in scenes.batch(self.batch, generator)
            ]
            confusor = 100.0 * sum(s[0] for s in stats) / len(stats)
            own = sum(s[1] for s in stats) / len(stats)
            rows.append((rate, confusor, own))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.45])
    parser.add_argument("--faithful", action="store_true", help="use the FAITHFUL_APPEARANCE preset")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--gate-um", type=float, default=10.0, help="linkable-edge gate (max annotated step)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = FAITHFUL_APPEARANCE if args.faithful else SceneConfig()
    audit = ConfusorAudit(base=base, gate_um=args.gate_um, batch=args.batch, seed=args.seed)
    logger.info(
        "config: n_cells=%d  spacing=%sum  gate=%sum  batch=%d",
        base.n_cells,
        base.spacing_um,
        args.gate_um,
        args.batch,
    )
    logger.info("%6s  %10s  %11s", "rate", "confusor%", "own_med_um")
    for rate, confusor, own in audit.sweep(args.rates):
        logger.info("%6.2f  %8.2f%%  %11.2f", rate, confusor, own)


if __name__ == "__main__":
    main()

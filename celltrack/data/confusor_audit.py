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

import numpy as np
import torch

from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.joint_dataset import PairSample
from celltrack.data.synthetic_scene import _SCENE_DOWNSAMPLE, FAITHFUL_APPEARANCE, Scene, SceneConfig

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

    @staticmethod
    def last_step_alignment(scene: Scene, config: SceneConfig) -> float:
        """Mean cos(true_step_{t->t+1}, incoming_{t-1->t}) over the confusor SOURCES at the last transition.

        This is the axis the head/velocity-prior actually keys on: the default construction poses it ~ +1
        (successor departs ALONG the heading, velocity SOLVES it), the real dense confusor is ~ -0.17 (successor
        departs OFF the heading, velocity DEFEATS it). Sources are identified as the cells forced to the long
        `confusor_min_step_um` jump; distractors (near-static) and plain cells (own speed) fall below the cut.
        """
        cells, frames = config.n_cells, config.n_frames
        positions = scene.positions_vox.reshape(frames, cells, 3)  # row f*cells + c -> [f, c]
        t = frames - 2
        incoming = positions[t] - positions[t - 1]
        step = positions[t + 1] - positions[t]
        step_um = np.linalg.norm(step, axis=1) * config.spacing_um
        source = step_um >= 0.5 * config.confusor_min_step_um  # the forced long jump marks a confusor source
        if not source.any():
            return float("nan")
        inc, stp = incoming[source], step[source]
        cos = (inc * stp).sum(axis=1) / (np.linalg.norm(inc, axis=1) * np.linalg.norm(stp, axis=1) + 1e-9)
        return float(cos.mean())

    def alignment(self, rates: list[float], *, invert: bool) -> list[tuple[float, float]]:
        """(rate, mean source alignment) per `confusor_rate` for one construction mode — the faithfulness check."""
        rows: list[tuple[float, float]] = []
        for rate in rates:
            config = replace(self.base, confusor_rate=rate, confusor_invert_velocity=invert)
            scenes = (Scene.generate(config, self.seed + i) for i in range(self.batch))
            cosines = [self.last_step_alignment(scene, config) for scene in scenes]
            finite = [c for c in cosines if not np.isnan(c)]
            rows.append((rate, sum(finite) / len(finite) if finite else float("nan")))
        return rows

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
    parser.add_argument(
        "--align", action="store_true", help="report source alignment for continuous vs inversion instead of rate"
    )
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
    if args.align:
        # Faithfulness check: real dense confusor keys on align ~ -0.17; continuous poses ~ +1, inversion ~ -0.17.
        continuous = dict(audit.alignment(args.rates, invert=False))
        inversion = dict(audit.alignment(args.rates, invert=True))
        logger.info("%6s  %12s  %12s", "rate", "continuous", "inversion")
        for rate in args.rates:
            logger.info("%6.2f  %12.3f  %12.3f", rate, continuous[rate], inversion[rate])
        return
    logger.info("%6s  %10s  %11s", "rate", "confusor%", "own_med_um")
    for rate, confusor, own in audit.sweep(args.rates):
        logger.info("%6.2f  %8.2f%%  %11.2f", rate, confusor, own)


if __name__ == "__main__":
    main()

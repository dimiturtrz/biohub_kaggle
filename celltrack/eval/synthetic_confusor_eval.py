"""CLI: rank a saved joint checkpoint's affinity on a LARGE generated INVERTED-confusor set.

The real faithful confusor axis is tiny — the four annotated validation movies hold ~48 mislinked edges
between them, so a per-window top1 read off them resolves to a single case and any bottleneck-B fix reads
as noise. This entrypoint measures the SAME quantities (top1, the affinity-inverted share, mean P_true) on
the CPU multi-frame generator's velocity-DEFEATING confusor at high rate, where the case count runs to the
thousands — the ruler that can SEE an axis-supervision effect the annotated floor cannot.

It is a SYNTHETIC ruler. A move here says axis-supervision shifts the faithful axis ON ITS OWN distribution
(a clean go/kill for the whole slack / confusor-margin line); it does NOT prove transfer to the 48 real
cases — `joint_eval --checkpoint ... --movies <dense>` is that (noise-floored) number.

The forward is `_edge_auc`'s exact shape (softmax over sources per target); the metrics are
`dense_diagnosis.MislinkSignal`'s (rank 0 = top1, P_chosen > P_true = inverted), read off the generator's GT
`edge_matrix` directly since a placed scene needs no tracker or matching. The HARD subset is the confusor
axis isolated arm-independently: a target whose geometrically NEAREST source is not its true parent is
exactly the velocity-defeating case a distance or velocity prior mislinks.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int

from celltrack.data.joint_dataset import PairDataset, PairOptions, PairTarget
from celltrack.data.synthetic_scene import FAITHFUL_APPEARANCE, SceneConfig, SceneCorpus
from celltrack.eval.prior_velocity_source import PriorVelocitySource
from celltrack.models.joint_model import JointModel
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_DATASET = "biohub_cell_tracking"
# Generated scenes render on the (1,4,4)-equivalent 64^3 grid with raw-lifted nodes; PairDataset must read
# them on that grid or the frame reader raises (synthetic_scene.SceneFrames.frame).
_SCENE_DOWNSAMPLE = (1, 4, 4)


@dataclass(frozen=True)
class ConfusorScore:
    """The mislink read-out over one population of scored target columns."""

    top1: float
    inverted_fraction: float
    mean_p_true: float
    n_cases: int

    @classmethod
    def of(
        cls,
        ranks: Int[np.ndarray, " n"],
        p_true: Float[np.ndarray, " n"],
        p_chosen: Float[np.ndarray, " n"],
    ) -> "ConfusorScore":
        """The pooled statistics, or an all-NaN empty score when no column carried a GT parent."""
        if not len(ranks):
            return cls(float("nan"), float("nan"), float("nan"), 0)
        return cls(
            top1=float(np.mean(ranks == 0)),
            inverted_fraction=float(np.mean(p_chosen > p_true)),
            mean_p_true=float(np.mean(p_true)),
            n_cases=len(ranks),
        )


@dataclass(frozen=True)
class ConfusorEvalResult:
    """The whole-set and confusor-only reads side by side — the confusor axis is `hard`."""

    every: ConfusorScore
    hard: ConfusorScore


class _RankPool:
    """Accumulates (rank, P_true, P_chosen) per scored target column, split into all vs confusor-hard."""

    def __init__(self) -> None:
        self._all: list[tuple[int, float, float]] = []
        self._hard: list[tuple[int, float, float]] = []

    def add(
        self,
        posterior: Float[np.ndarray, " s"],
        true_source: int,
        *,
        is_hard: bool,
    ) -> None:
        """Record one target's ranking of its true parent against the rival sources."""
        true = float(posterior[true_source])
        row = (int(np.sum(posterior > true)), true, float(posterior.max()))
        self._all.append(row)
        if is_hard:
            self._hard.append(row)

    def result(self) -> ConfusorEvalResult:
        """Pool both populations into their `ConfusorScore`s."""
        return ConfusorEvalResult(self._score(self._all), self._score(self._hard))

    @staticmethod
    def _score(rows: list[tuple[int, float, float]]) -> ConfusorScore:
        if not rows:
            return ConfusorScore.of(np.array([]), np.array([]), np.array([]))
        ranks, p_true, p_chosen = (np.asarray(column) for column in zip(*rows, strict=True))
        return ConfusorScore.of(ranks, p_true, p_chosen)


@dataclass(frozen=True)
class SyntheticConfusorEval:
    """A large generated inverted-confusor set and the machinery to rank a model's affinity over it."""

    targets: list[PairTarget]
    velocity: PriorVelocitySource
    prior_velocity: bool

    @classmethod
    def build(
        cls,
        n_scenes: int,
        confusor_rate: float,
        seed: int,
        *,
        faithful: bool,
        prior_velocity: bool,
    ) -> "SyntheticConfusorEval":
        """Generate `n_scenes` velocity-DEFEATING scenes at `confusor_rate` and mount the velocity feed."""
        base = FAITHFUL_APPEARANCE if faithful else SceneConfig()
        config = replace(base, confusor_rate=confusor_rate, confusor_invert_velocity=True)
        targets = SceneCorpus.generate(config, n_scenes, seed)
        velocity = PriorVelocitySource(enabled=prior_velocity, gt_warmup_steps=0)
        return cls(targets, velocity, prior_velocity)

    @staticmethod
    def _autocast(device: str) -> torch.autocast:
        """The run's precision context — bf16 on CUDA, a no-op elsewhere, matching the trainer's forwards."""
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    @torch.no_grad()
    def score(self, model: JointModel, seed: int) -> ConfusorEvalResult:  # devtools-ignore: test-mirror
        """Rank the model's affinity over every generated target, pooled and confusor-only."""
        model.eval()
        device = str(next(model.parameters()).device)
        pool = _RankPool()
        for target in self.targets:
            corpus = PairDataset([target], 1, _SCENE_DOWNSAMPLE, seed, PairOptions(with_previous=self.prior_velocity))
            sample = corpus.pair(0).to(device)
            velocity = self.velocity.of(model, sample, self._autocast(device))
            with self._autocast(device):
                logits = model.forward(
                    sample.frame_t, sample.frame_t1, sample.source_centres, sample.target_centres, velocity
                ).edge_logits
            posteriors = torch.softmax(logits.float(), dim=0).cpu().numpy()  # (s, u) over sources per target
            self._pool_columns(pool, posteriors, target)
        return pool.result()

    @staticmethod
    def _pool_columns(pool: _RankPool, posteriors: Float[np.ndarray, "s u"], target: PairTarget) -> None:
        """Score each target column that has a GT parent, flagging the geometry-hard confusor cases."""
        sources = target.source_centres.astype(np.float64)
        targets = target.target_centres.astype(np.float64)
        for column_index in range(target.edge_matrix.shape[1]):
            column = target.edge_matrix[:, column_index]
            if column.sum() == 0:  # a target with no GT parent (scene entry / division child) has no rank
                continue
            true_source = int(np.argmax(column))
            nearest_source = int(np.argmin(np.linalg.norm(sources - targets[column_index], axis=1)))
            pool.add(posteriors[:, column_index], true_source, is_hard=nearest_source != true_source)


def main() -> None:
    """Rank one joint checkpoint on a large generated inverted-confusor set, logging the whole and hard reads."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Rank a joint checkpoint on a generated inverted-confusor set.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the joint checkpoint, under the processed root")
    parser.add_argument(
        "--n-scenes", type=int, default=400, help="scenes to generate; each holds ~confusor_rate*n_cells hard cases"
    )
    parser.add_argument("--confusor-rate", type=float, default=0.45, help="fraction of cells made confusors")
    parser.add_argument(
        "--seed", type=int, default=90210, help="held-out generator seed, kept disjoint from any training seed"
    )
    parser.add_argument("--faithful", action="store_true", help="render with the appearance-faithful preset")
    parser.add_argument(
        "--no-prior-velocity",
        dest="prior_velocity",
        action="store_false",
        help="score without the velocity feed (default: on, matching a --prior-velocity arm)",
    )
    args = parser.parse_args()

    root = DataRoot.from_config(args.config)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root.processed(_DATASET) / args.checkpoint
    model = JointModel.from_checkpoint(checkpoint, args.device)
    harness = SyntheticConfusorEval.build(
        args.n_scenes,
        args.confusor_rate,
        args.seed,
        faithful=args.faithful,
        prior_velocity=args.prior_velocity,
    )
    result = harness.score(model, args.seed)
    logger.info(
        "checkpoint=%s n_scenes=%d confusor_rate=%.2f invert=on faithful=%s prior_velocity=%s",
        checkpoint.name,
        args.n_scenes,
        args.confusor_rate,
        args.faithful,
        args.prior_velocity,
    )
    logger.info(
        "ALL   n=%d top1 %.4f inverted %.4f mean_P_true %.4f",
        result.every.n_cases,
        result.every.top1,
        result.every.inverted_fraction,
        result.every.mean_p_true,
    )
    logger.info(
        "HARD  n=%d top1 %.4f inverted %.4f mean_P_true %.4f  (confusor axis: nearest source != true parent)",
        result.hard.n_cases,
        result.hard.top1,
        result.hard.inverted_fraction,
        result.hard.mean_p_true,
    )


if __name__ == "__main__":
    main()

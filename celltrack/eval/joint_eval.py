"""CLI: score a SAVED joint checkpoint through the shipped tracker on a chosen movie set.

The joint trainer scores its own weights every window, but only on the movies it selected on. Reading a
finished run against the untouched TEST four — or diagnosing it on the dense movie — needs the same assembly
from the outside, and there was no entrypoint for it: `proxy_eval` and `dense_diagnosis` both mount the
PUBLISHED packs, so neither can see our own trained heads at all. That is the difference this module exists
for; it is not a flag either of them could have carried.

`ModelEvaluator.evaluate_joint` does the scoring unchanged, so the number here is the number the run's own log
reported — the faithful competition score, the clamped selection score, node recall and the node-count ratio,
and the association read-out (mislink count, the affinity-inverted share, mean `P_true` vs `P_chosen`). Pointed
at the dense movie alone, those last four ARE the mislink mechanism check, measured on the arm's own weights.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, TestMovieProxy
from celltrack.eval.proxy_eval import ConfigOverride
from celltrack.models.joint_model import JointModel
from celltrack.operating_point import TrackerConfig
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_DATASET = "biohub_cell_tracking"
# The published packs, spelled as `dense_diagnosis` and `proxy_eval` spell them. A joint checkpoint supplies
# BOTH heads, so nothing of these reaches the score — `ModelEvaluator` simply holds a pack-based affinity for
# the detector-only path it also serves, and mounting one is the price of reusing that assembly unchanged.
_PACKS = (Path("reference/pilkwang/split_0"), Path("reference/pilkwang/seed2/weights/unet_transformer/split_0"))


class JointCheckpointEval:
    """The shipped-pipeline evaluator, mounted over an arbitrary movie set for scoring saved joint weights."""

    @staticmethod
    def mounted(
        root: DataRoot, stems: tuple[str, ...], downsample: tuple[int, int, int], device: str, config: TrackerConfig
    ) -> ModelEvaluator:
        """An evaluator over `stems`, its detector recipe taking the DOWNSAMPLE the checkpoint was trained at.

        Read off the checkpoint rather than defaulted: a model scored through a different input grid than it
        trained on is a different model, and the mismatch would show up as a quality result.
        """
        proc = root.processed(_DATASET)
        recipe = DetectorRecipe(downsample=downsample, tta=False)
        packs = (proc / _PACKS[0], proc / _PACKS[1])
        return ModelEvaluator.mount(TestMovieProxy.load(root, stems), packs, recipe, device, config)


def main() -> None:
    """Score one joint checkpoint on the requested movies and log the score beside the association read-out."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Score a saved joint checkpoint through the shipped tracker.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the joint checkpoint, under the processed root")
    parser.add_argument(
        "--movies",
        default=",".join(TEST_MOVIES),
        help="comma-separated movie stems to score (default: the untouched test four)",
    )
    parser.add_argument(
        "--threshold", type=float, default=TrackerConfig.shipped().threshold, help="detection threshold"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any tracker knob, exactly as `proxy_eval --set` does",
    )
    args = parser.parse_args()

    root = DataRoot.from_config(args.config)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root.processed(_DATASET) / args.checkpoint
    stems = tuple(args.movies.split(","))
    model = JointModel.from_checkpoint(checkpoint, args.device)
    # The base is the SHIPPED operating point, not TrackerConfig's neutral defaults: those default to the
    # per-frame assignment linker with no fusion, which is the 0.892 pipeline rather than the 0.895 one we
    # submit — and scoring our own heads under a linker we do not deploy is how a better model gets discarded.
    config = replace(TrackerConfig.shipped(), threshold=args.threshold)
    for assignment in args.overrides:
        config = ConfigOverride.apply(config, assignment)
    evaluator = JointCheckpointEval.mounted(root, stems, model.downsample, args.device, config)
    result = evaluator.evaluate_joint(model)
    logger.info(
        "checkpoint=%s movies=%s threshold=%.3f linker=%s boundary=%.1f bidirectional=%s",
        checkpoint.name,
        list(stems),
        config.threshold,
        config.linker.name,
        config.linker.disappearance_cost,
        config.bidirectional_edges,
    )
    logger.info(
        "faithful %.4f | clamped %.4f | node recall %.4f ratio %+.3f | mislinks %d inverted %.3f "
        "P_true %.3f vs P_chosen %.3f",
        result.score,
        result.selection_score,
        result.node_recall,
        result.node_ratio,
        result.mislinks,
        result.inverted_fraction,
        result.mean_p_true,
        result.mean_p_chosen,
    )


if __name__ == "__main__":
    main()

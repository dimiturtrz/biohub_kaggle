"""Calibration-drift test: does a COLLAPSED warm-finetune checkpoint recover proxy under a re-derived
detection threshold? If a higher threshold climbs the proxy back toward the 0.847 warm baseline, the
collapse is calibration (recipe salvageable); if the proxy stays low at every threshold, it is real
association-quality loss. Legit re-threshold: the model CHANGED, its operating point moved — not a knob
on a frozen model."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import VALIDATION_MOVIES, TestMovieProxy
from celltrack.models.joint_model import JointModel
from celltrack.training.joint_cli import JointCli
from celltrack.training.joint_config import WARM_PACKS, JointTrainConfig
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[0] / "paths.yaml"  # unused; real path below
DATASET = "biohub_cell_tracking"
DEVICE = "cuda"
THRESHOLDS = (0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998)


def main() -> None:
    ckpt_name = sys.argv[1]  # e.g. joint_plainctrl_v1.resume.pt
    root = DataRoot.from_config(Path("paths.yaml"))
    proc = root.processed(DATASET)

    args = JointCli.build_parser().parse_args(["--eval-threshold", "0.97"])
    config = JointTrainConfig.from_args(args)
    recipe = DetectorRecipe(downsample=config.data.downsample, tta=config.eval.tta)
    tracker_config = config.eval.tracker

    print(f"mounting evaluator (VALIDATION_MOVIES={VALIDATION_MOVIES})", flush=True)
    validation = TestMovieProxy.load(root, VALIDATION_MOVIES)
    packs = (proc / WARM_PACKS["seed1"], proc / WARM_PACKS["seed2"])
    evaluator = ModelEvaluator.mount(validation, packs, recipe, DEVICE, tracker_config)

    print(f"loading collapsed checkpoint {ckpt_name}", flush=True)
    model = JointModel.from_checkpoint(proc / ckpt_name, DEVICE)

    print(f"\n{'thr':>6} {'score':>8} {'sel':>8} {'nodeR':>7} {'ratio':>7} {'inv':>6} {'pTrue':>7} {'pChos':>7}")
    for thr in THRESHOLDS:
        new_cfg = dataclasses.replace(tracker_config, threshold=thr)
        ev = dataclasses.replace(evaluator, config=new_cfg)
        r = ev.evaluate_joint(model)
        print(
            f"{thr:>6.3f} {r.score:>8.4f} {r.selection_score:>8.4f} {r.node_recall:>7.3f} "
            f"{r.node_ratio:>+7.2f} {r.inverted_fraction:>6.2f} {r.mean_p_true:>7.3f} {r.mean_p_chosen:>7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""The joint trainer's command line — every flag it takes, kept apart from the training logic.

The flags are the interface; `joint_detector.main` fans them out into the config objects each concern owns.
Split out so the trainer module carries the training logic and this carries the (long, flat) argument surface.
"""

from __future__ import annotations

import argparse

from celltrack.operating_point import TrackerConfig
from celltrack.training.joint_config import WARM_PACKS, ContrastiveSite


class JointCli:
    """The joint run's argument surface — a namespace for the parser, kept off `main` so wiring reads short."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Every flag a joint run takes — kept apart from `main` so the run's WIRING reads as its own short surface."""
        parser = argparse.ArgumentParser(description="Train the joint detector+edge model on the non-held-out videos.")
        parser.add_argument("--steps", type=int, default=1500)
        parser.add_argument("--eval-every", type=int, default=500, help="steps per eval window")
        # The step scale above is a probe-run unit: one pair per step means the ~18000-pair corpus makes an epoch,
        # so `--steps 3000` is 17% of ONE pass against the published pack's 50 epochs, and `--patience` counts
        # EVAL WINDOWS — five of them is 14% of an epoch at eval-every 500 and five whole epochs at 9000. The
        # flags below say the same things in passes over the corpus, and win over their raw twin when given.
        parser.add_argument("--epochs", type=float, default=None, help="passes over the GT pairs; overrides --steps")
        parser.add_argument(
            "--evals-per-epoch", type=int, default=None, help="eval windows per epoch; overrides --eval-every"
        )
        parser.add_argument(
            "--patience-epochs",
            type=float,
            default=None,
            help="epochs without a gain before stopping; overrides --patience",
        )
        parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
        # Chain a staged curriculum: load a checkpoint THIS trainer wrote (both heads, any width it saved at) as the
        # init, fresh optimiser — so synth-pretrain -> train-real -> finetune each warm from the last and the slow
        # early stages are done ONCE. Distinct from --warm-start (the pilkwang pack); the two are mutually exclusive.
        parser.add_argument(
            "--init-weights",
            type=str,
            default=None,
            help="init from a joint checkpoint this trainer saved (path relative to processed/)",
        )
        # Stage-3 corpus: fold the four TEST movies into the train set (competitors do, and the hidden set overlaps
        # them). Selection still lands on the validation four, so the LB estimate stays honest — only the TRAIN
        # corpus grows. Off by default: test stays held out as an estimate.
        parser.add_argument(
            "--include-test-in-train",
            action="store_true",
            help="fold the four test movies into the TRAIN corpus (selection stays on the validation four)",
        )
        parser.add_argument(
            "--det-weight", type=float, default=1.0, help="weight on the detection term vs the edge term"
        )
        parser.add_argument(
            "--contrastive-weight", type=float, default=0.0, help="weight on the InfoNCE term over the node features"
        )
        # WHERE that weight acts. On the features it is measured to trade detection away for discrimination
        # (proxy 0.8414 -> 0.7292, node recall 0.966 -> 0.867); the projecting sites give the term its own
        # embedding to shape, `detached_projection` keeping the backbone out of its reach entirely.
        parser.add_argument(
            "--contrastive-site",
            choices=tuple(ContrastiveSite),
            default=ContrastiveSite.FEATURES,
            type=ContrastiveSite,
            help="where the InfoNCE term acts: on the shared features, or through its own projection head",
        )
        parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE softmax temperature over targets")
        # The zni probe's mining under a MARGIN objective and an UNFROZEN backbone — the one combination never run.
        # Its absolute form (decoy probability -> 0, UNet frozen) flattened the head; a weight here is matched to the
        # term's own logged magnitude against the edge term, so the arm measures the mechanism and not a rescaling.
        parser.add_argument(
            "--hard-negative-weight",
            type=float,
            default=0.0,
            help="weight on the ranking term holding each source's true logit above its nearest wrong targets'",
        )
        parser.add_argument(
            "--hard-negatives", type=int, default=4, help="nearest wrong targets mined per annotated source"
        )
        parser.add_argument(
            "--symmetric-links",
            action="store_true",
            help="normalise the link loss across TARGETS too (one child per source), not sources alone",
        )
        parser.add_argument(
            "--reliability-weighting",
            action="store_true",
            help="weight each pair's loss by SNR/(1+crowd): trust clean supervision, ease off ambiguous cells",
        )
        parser.add_argument(
            "--balanced-links",
            action="store_true",
            help="count the link loss once per decision, balancing the true candidate against its rivals",
        )
        # Fine-tuning a CONVERGED model at its original training rate is the classic way to walk off its optimum,
        # which is what 3000 warm-start steps at 1e-4 did (never beat the init). Exposed so the rate is a
        # variable of the experiment rather than an inherited constant.
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate; lower it when warm-starting")
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="pairs forwarded together in one backbone pass, averaged into one step (1 = per-pair reference)",
        )
        parser.add_argument("--cosine-lr", action="store_true", help="decay the rate to zero over the run")
        parser.add_argument(
            "--warm-pack", choices=tuple(WARM_PACKS), default="seed1", help="which published pack to continue"
        )
        # Frame decompression prefetches the GPU step (uniform path): more threads = the GPU stops stalling on
        # zarr reads. Defaults are conservative; a 32-core box saturates the GPU at ~16 threads / 24 in flight.
        # KEEP THESE LOW. Each loader thread holds a concurrent memory-mapped zarr reader, and `prefetch` in-flight
        # decompressed pairs sit in host RAM; 24 threads / 32 prefetch OOM'd a 94GB box (Windows commit limit) in
        # ~10s under concurrent load. The 2x throughput that config bought is NOT worth crashing the machine — a
        # safe fast config must be RE-MEASURED with RSS + commit-charge monitoring (bead) before the default rises.
        parser.add_argument("--loader-threads", type=int, default=4, help="frame-decompression threads (mmap readers)")
        parser.add_argument("--loader-prefetch", type=int, default=8, help="pairs kept in flight ahead of the step")
        # The corpus whose candidate set is the one the tracker deploys against: 3.86 in-gate candidates per
        # source with 97.1% contested, where the annotated pairs carry 0.99 and 1.9%. These REPLACE the
        # annotated pairs — mixing re-introduces the uncontested rows the corpus exists to escape.
        parser.add_argument(
            "--detected-videos",
            type=int,
            default=0,
            help="train videos to build the DETECTED-neighbourhood corpus over (0 = the annotated pairs)",
        )
        parser.add_argument(
            "--difficulty-sampling",
            action="store_true",
            help="draw pairs in proportion to their measured top-1 defect instead of uniformly with replacement",
        )
        # The one property our own corpus cannot have: every cell of a synthetic frame is labelled, so the crowded
        # near-neighbours enter the edge matrix as TRUE NEGATIVES instead of sitting outside it as unannotated
        # detections — which is what all 38 of the dense movie's mislinked partners are.
        parser.add_argument(
            "--synthetic-fraction",
            type=float,
            default=0.0,
            help="share of steps drawn from the fully-labelled synthetic corpus (0 = the real corpus alone)",
        )
        parser.add_argument(
            "--synthetic-sequences", type=int, default=None, help="synthetic sequences to enumerate (default: all)"
        )
        parser.add_argument(
            "--gpu-scene-fraction", type=float, default=0.0, help="fraction of each batch generated fresh on GPU"
        )
        parser.add_argument(
            "--gpu-scene-detection",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="generated scenes also train detection (honest 100%% labels); --no- for association only",
        )
        parser.add_argument(
            "--synthetic-scenes", type=int, default=0, help="hard scenes to GENERATE dynamically (0 = static npz)"
        )
        parser.add_argument(
            "--paced-curriculum",
            action="store_true",
            help="replace the fixed synthetic share with the EMA'd, loss-weighting, hard-early feedback controller",
        )
        parser.add_argument(
            "--prior-velocity",
            action="store_true",
            help="feed each source its t-1 -> t displacement as head input (~50%% slower per step)",
        )
        parser.add_argument(
            "--velocity-gt-warmup-steps",
            type=int,
            default=0,
            help="bootstrap velocity off GT t-1 -> t links for N steps (linear anneal), breaking the from-scratch"
            " chicken-egg; N > 0 IMPLIES --prior-velocity (warmup 0 is the refuted from-scratch dead-end); 0 = off",
        )
        JointCli._add_augmentation_args(parser)
        parser.add_argument("--temporal-position", action="store_true", help="frame-position embedding (motion sight)")
        # LoRA: freeze the pilkwang base, train low-rank adapters on the association locus. Detection is anchored
        # by the frozen base; the adapters give association the feature reshape the frozen backbone cannot.
        parser.add_argument("--lora", action="store_true", help="freeze the base, train low-rank adapters only")
        parser.add_argument("--lora-rank", type=int, default=16, help="adapter rank (per-layer capacity)")
        parser.add_argument("--lora-alpha", type=float, default=16.0, help="adapter scale (delta = alpha/rank * BA)")
        parser.add_argument(
            "--lora-targets",
            nargs="+",
            default=["transformer", "decoder_blocks"],
            help="dotted-path substrings of the leaves to adapt (default: transformer + U-Net decoder)",
        )
        # The threshold is DERIVED, not swept: given bare, the flag takes the pipeline's own inference operating
        # point — the response at which the shipped tracker would already have called the voxel a cell, so every
        # voxel the mask spares is one the deployed model detects and our sparse annotation cannot adjudicate.
        parser.add_argument(
            "--ignore-ambiguous-above",
            type=float,
            nargs="?",
            const=TrackerConfig.shipped().threshold,
            default=None,
            help="leave unannotated voxels above this sigmoid response unsupervised (bare = the tracker threshold)",
        )
        parser.add_argument("--patience", type=int, default=5, help="stop after N non-improving evals (<1 disables)")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--val-videos", type=int, default=2, help="validation movies the edge AUC is measured over")
        parser.add_argument(
            "--eval-threshold",
            type=float,
            default=TrackerConfig.shipped().threshold,
            help="detection threshold of the selector eval (defaults to the SHIPPED operating point's)",
        )
        parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
        parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
        return parser

    @staticmethod
    def _add_augmentation_args(parser: argparse.ArgumentParser) -> None:
        """The label-preserving augmentation flags — one on/off decision plus per-component overrides.

        The pilkwang recipe augments (intensity jitter + axis flips) and every from-scratch joint run did not,
        which is the from-scratch recall gap: warm weights recover cells our own weights miss from identical
        pixels. A pair shares one draw across t-1/t/t+1. ON by default at the moderate xy preset
        (`DEFAULT_AUGMENTATION`); `--no-augment` is the byte-identical un-augmented baseline. Each `--aug-*` flag
        overrides one preset component (an unset one keeps the preset's value).
        """
        parser.add_argument(
            "--augment",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="augment with the moderate xy preset; --no-augment restores the byte-identical un-augmented baseline",
        )
        parser.add_argument(
            "--aug-brightness", type=float, default=None, help="override the preset multiplicative jitter half-range"
        )
        parser.add_argument(
            "--aug-offset", type=float, default=None, help="override the preset additive jitter half-range"
        )
        parser.add_argument(
            "--aug-flip-axes",
            nargs="*",
            type=int,
            default=None,
            help="override the preset flip axes (0=z 1=y 2=x), each flipped with p=0.5, centres mirrored to match",
        )

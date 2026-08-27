"""The joint trainer's command line — every flag it takes, kept apart from the training logic.

The flags are the interface; `joint_detector.main` fans them out into the config objects each concern owns.
Split out so the trainer module carries the training logic and this carries the (long, flat) argument surface.
"""

from __future__ import annotations

import argparse

from celltrack.operating_point import TrackerConfig
from celltrack.training.joint_config import WARM_PACKS, ContrastiveSite, SelectionObjective


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
        # Fast head-only validation: warm ONLY the detector from a converged checkpoint, fresh-init the configured
        # head, freeze the detector, train the head alone. Isolates the edge-head mechanism on identical detector
        # features (both arms share one detector) and skips the multi-hour detector pretrain. Unlike --init-weights
        # (whole model, same head) this crosses head classes: a pack-detector ckpt warms a --head hoct run.
        parser.add_argument(
            "--detector-from",
            type=str,
            default=None,
            help="warm the detector from a joint checkpoint, fresh-init + train ONLY the head (path under processed/)",
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
        JointCli._add_contrastive_flags(parser)
        JointCli._add_link_flags(parser)
        parser.add_argument(
            "--reliability-weighting",
            action="store_true",
            help="weight each pair's loss by SNR/(1+crowd): trust clean supervision, ease off ambiguous cells",
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
        parser.add_argument(
            "--downsample",
            type=int,
            nargs=3,
            default=(1, 4, 4),
            metavar=("DZ", "DY", "DX"),
            help="(z, y, x) pooling grid; the published default is (1, 4, 4). Finer y/x resolves the confusor.",
        )
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
        JointCli._add_synthetic_flags(parser)
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
        parser.add_argument(
            "--relative-position",
            action="store_true",
            help="add a learned per-head distance bias to the edge transformer's cross-attention (geometry sight)",
        )
        parser.add_argument(
            "--relative-position-directional",
            action="store_true",
            help="extend --relative-position with a per-head offset-DIRECTION term (the confusor's directional axis)",
        )
        JointCli._add_architecture_flags(parser)
        parser.add_argument(
            "--freeze-backbone-norm",
            action="store_true",
            help="hold the warm pack's BatchNorm running stats frozen (eval-mode BN through train); warm-start only",
        )
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
        parser.add_argument(
            "--selection-objective",
            choices=tuple(o.value for o in SelectionObjective),
            default=SelectionObjective.PROXY.value,
            help="scalar save-best maximises: 'proxy' (shipped clamped metric) or 'confusor_margin' "
            "(proxy-independent mean_p_true - mean_p_chosen, for a mechanism run above the proxy ceiling)",
        )
        parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
        parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
        return parser

    @staticmethod
    def _add_architecture_flags(parser: argparse.ArgumentParser) -> None:
        """The from-scratch backbone/head selectors — the two axes a warm-start pack cannot vary.

        `--norm` picks the backbone normalisation and `--head` the edge-head architecture; the warm-start guard
        rejects any non-default here, since the pilkwang pack is BatchNorm + the pack head byte for byte.
        """
        parser.add_argument(
            "--norm",
            choices=("batch", "group"),
            default="batch",
            help="backbone norm: 'batch' (published BN) or 'group' (domain-robust GroupNorm, from-scratch only)",
        )
        parser.add_argument(
            "--head",
            choices=("pack", "hoct"),
            default="pack",
            help="edge-head architecture (from-scratch only): 'pack' (pilkwang SimpleNodeTransformer) or 'hoct'"
            " (two-stage 3D-RoPE node + line-to-line edge attention)",
        )
        parser.add_argument(
            "--edge-hidden-dim",
            type=int,
            default=128,
            help="edge-head width (from-scratch only). Default 128 = pilkwang shape; the #1-CTC HOCT arch is 288"
            " (must stay divisible by 4 heads). Read back off the checkpoint on reload, so a widened run reloads.",
        )
        parser.add_argument(
            "--edge-blocks",
            type=int,
            default=4,
            help="edge-head depth per stage (from-scratch only). Default 4 = pilkwang shape.",
        )

    @staticmethod
    def _add_synthetic_flags(parser: argparse.ArgumentParser) -> None:
        """The fully-labelled synthetic/GPU-scene corpus flags — the source of TRUE-NEGATIVE crowded neighbours.

        The one property our own corpus cannot have: every cell of a synthetic frame is labelled, so the crowded
        near-neighbours enter the edge matrix as TRUE NEGATIVES instead of sitting outside it as unannotated
        detections — which is what all 38 of the dense movie's mislinked partners are.
        """
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
            "--faithful-scenes",
            action="store_true",
            help="render generated scenes with the appearance-faithful preset (size spread + z-shape + noise)",
        )
        parser.add_argument(
            "--confusor-rate",
            type=float,
            default=0.0,
            help="fraction of generated cells made CONSTRUCTED confusors (fast source + slow near-rival); 0.3~=32%%",
        )
        parser.add_argument(
            "--confusor-invert-velocity",
            action="store_true",
            help="build the confusor velocity-DEFEATING (successor departs OFF the source heading) — the real dense "
            "mislink, vs the default velocity-continuous shape a head learns to trust and then inverts on",
        )

    @staticmethod
    def _add_contrastive_flags(parser: argparse.ArgumentParser) -> None:
        """The representation-shaping loss terms — InfoNCE, its margin, and the CELLECT center embedding.

        Every weight is 0.0 by default (computed and logged as a diagnostic, an unasked run is yesterday's).
        `--contrastive-weight`/`--contrastive-site` act on the SHARED node features (features site measured to
        trade detection away for discrimination); `--center-embed-weight` acts instead on the dedicated 64-ch
        `embed_head` the backbone is free to reshape without disturbing the detection logit beside it.
        """
        parser.add_argument(
            "--contrastive-weight", type=float, default=0.0, help="weight on the InfoNCE term over the node features"
        )
        parser.add_argument(
            "--contrastive-site",
            choices=tuple(ContrastiveSite),
            default=ContrastiveSite.FEATURES,
            type=ContrastiveSite,
            help="where the InfoNCE term acts: on the shared features, or through its own projection head",
        )
        parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE softmax temperature over targets")
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
            "--center-embed-weight",
            type=float,
            default=0.0,
            help="weight on the CELLECT center-point contrastive term over the dedicated 64-ch embed head",
        )

    @staticmethod
    def _add_link_flags(parser: argparse.ArgumentParser) -> None:
        """How the link loss normalises — the three interchangeable forms of the association objective.

        Each is off by default, so an unasked run is the frontier's plain source-axis focal BCE. `--symmetric-links`
        adds the target axis (one child per source); `--balanced-links` counts one decision at a time; `--slack-links`
        appends the learned no-parent row (parental softmax), which overrides the axes to source-only.
        """
        parser.add_argument(
            "--symmetric-links",
            action="store_true",
            help="normalise the link loss across TARGETS too (one child per source), not sources alone",
        )
        parser.add_argument(
            "--balanced-links",
            action="store_true",
            help="count the link loss once per decision, balancing the true candidate against its rivals",
        )
        parser.add_argument(
            "--slack-links",
            action="store_true",
            help="append a learned no-parent row (parental softmax) so unannotated rivals need no real parent",
        )

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

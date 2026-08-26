"""The tracker's operating point: every knob of the shipped recipe, as one object a sweep varies.

`TrackerConfig` is the vocabulary the whole repo speaks — the eval tools, the training entrypoints, the
submission dispatcher and the Kaggle kernels all name an operating point, and most of them never mount a
`CellTracker` at all. It therefore lives at the package root beside the runner rather than inside it, so
naming a recipe does not drag the mounted models' import graph along with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from celltrack.edges.blended_edge_scoring import EdgeBlendOptions
from celltrack.edges.calibrated_fusion import CalibratedFusionOptions
from celltrack.linkers.linkers import LinkerConfig
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig
from celltrack.postproc.division_recovery import DivisionRecoveryConfig
from celltrack.postproc.gap_closer import BridgeConfig, ReuseConfig
from celltrack.postproc.short_track_filter import ShortTrackRescueConfig
from celltrack.postproc.topology_repair import TopologyConfig

# The frontier_replica operating point's constants, each a MEASURED value of the public 0.926 stack, not a
# tuning surface (see `TrackerConfig.frontier_replica`). The detection threshold sits between the shipped 0.97
# and the high-precision 0.99; the secondary seed carries the frontier's 0.475 detector weight (so the primary,
# which `detector_blend` holds, is its complement); the division C3 gate and global fork fraction are the public
# divsub kernel's own; the motion linker gates tight-then-loose at the annotated one-frame displacement bounds.
_FRONTIER_THRESHOLD = 0.96875
_SECONDARY_DETECTOR_WEIGHT = 0.475
_MOTION_TIGHT_UM = 6.0
_MOTION_LOOSE_UM = 10.0


@dataclass(frozen=True)
class TrackerConfig:
    """Every operating-point knob of the tracker recipe, so a sweep varies one object, not a call site."""

    # 0.97 is the MEASURED best, not a round number: the leaderboard ladder is 0.99 -> 0.887,
    # 0.98 -> 0.891, 0.97 -> 0.892, a spread five times our noise floor. The default used to be 0.99,
    # so every caller that omitted a config silently got the worst of the three — the shipped kernel only
    # escaped it by hardcoding 0.97. Threshold is a RECALL lever the sparse proxy cannot see (it reads
    # flat-to-inverted there), so this value comes from the leaderboard and must not be re-tuned on the proxy.
    threshold: float = 0.97
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    edge_blend: tuple[float, float] = (0.8, 0.2)
    # The seed-1 fraction of the detector's logit blend; None is the equal mean the frontier ships. The edge
    # blend favours seed 1 (0.8), so the detector blend may too — a value w tilts the two seeds to (w, 1-w).
    detector_blend: float | None = None
    # Off by default: z-score / std-ratio align each secondary detector seed onto the primary before the logit
    # blend — the detection half of the calibrated dual-seed fusion, so an independently trained second seed
    # contributes its ranking rather than its own scale. Inert unless `detector_blend` tilts the two seeds.
    detector_align_moments: bool = False
    # Off by default: detect each frame inside its REAL (t, t+1) pair instead of the shipped (t, t) duplicate.
    # The net was TRAINED on real pairs but ships temporally blind; enable to A/B whether the temporal axis
    # carries the division / dense-detection signal the sparse proxy cannot see. A MOUNT-time knob (it changes
    # the detector's cached forward, keyed by the recipe fingerprint's `pc`), so it is threaded at mount, not
    # re-pointed by `with_config` — a pair_context A/B is two mounts, not two cells of one sweep.
    pair_context: bool = False
    # Off by default: recovering the edge head's second-daughter fork is a detection-limited lever (the second
    # daughter is often undetected, so the fork cannot be placed). Set it to A/B the division-jaccard term.
    division: AffinityDivisionConfig | None = None
    # Off by default: the geometry-only single-child division repair (a DIFFERENT mechanism from `division`
    # above — it joins an unparented cell to a single-child parent, so it does NOT need the second daughter
    # detected). The affinity fork is detection-limited; this one is placement-side, which the ceiling
    # diagnostic (CV8: 7/7 recoverable, all parts detected, shipped recovers 0) named as the real gap. Present
    # (non-None) mounts it as a post-link stage, before the short-track filter.
    division_recovery: DivisionRecoveryConfig | None = None
    min_track_length: int = 6
    # Off by default: the confidence rescue that keeps a short high-probability, tight-step component the length
    # rule would drop. It reads the edge head, so it needs an affinity; enable it (with an aggressive
    # min_track_length) to recover the true short tracks the blunt cut removes — a recall lever the sparse proxy
    # understates but the denser hidden annotation rewards. The frontier pairs min_track_length 7 with this.
    rescue: ShortTrackRescueConfig | None = None
    # Off by default: exempt components touching the first or last frame from the length cut. Their
    # brevity is the observation window's doing, not the detector's — a cell entering at frame 96 of 100
    # cannot reach min_track_length however real it is, yet the hidden annotation still scores its edges.
    keep_boundary_tracks: bool = False
    # The one-frame motion-synthetic gap bridge — always on (it helps, +0.0017 at its plateau cap).
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    # Off by default (present = on): reuse an existing isolated t+1 detection to bridge a one-frame gap (geometry
    # only, no invented node), complementary to the motion-synthetic bridge. Inert at thr0.99 (few isolated nodes
    # survive), but the lower-threshold recall regime leaves more isolated detections to reuse — enable it there
    # to A/B whether the precision-safe reuse recovers dropped edges the synthetic bridge misses.
    reuse: ReuseConfig | None = None
    # Off by default (present = on): on the shipped AssignmentLinker path this is a measured no-op (the linker
    # already guarantees consecutive-frame + single-parent edges and a tighter gate), so it is off; enable it
    # behind a linker (nearest-neighbour, motion) that lacks those invariants — the frontier's output filter.
    topology: TopologyConfig | None = None
    # Proxy favours 0.3 but a clean LB A/B refuted it (smooth0.8 = 0.892 vs smooth0.3 = 0.889): the dense
    # raw-Jaccard gain is recall-side and does not transfer, like min_track_length/node-count. Ship the LB
    # value 0.8; 0.3 stays a swept-available knob for the proxy regime.
    smooth_strength: float = 0.8
    # Off by default: score every gap a second time with the temporal pair swapped and fuse the two
    # normalisations by their harmonic mean (`BlendedEdgeTransformerScorer.fuse`). Costs a second affinity
    # forward and no retraining; it targets the affinity-inverted dense mislink, where a wrong near neighbour
    # wins forwards but loses backwards.
    bidirectional_edges: bool = False
    # The edge blend's two candidate transforms (`EdgeBlendOptions`), both off by default: seed-moment alignment
    # before the weighted sum, and four-view flip TTA on the edge head fused by a JS-reliability log pool. They
    # live here rather than at the mount because they are operating-point knobs a sweep re-points — reach them
    # with `--set edge_options.align_seed_moments=true` / `--set edge_options.view_tta=true`.
    edge_options: EdgeBlendOptions = field(default_factory=EdgeBlendOptions)
    # OFF by default (byte-identical when off): the two remaining kernel-fidelity divergences that live on the
    # tracker's own path rather than in a nested config — the linker's edge-candidate admission floor (E5, the
    # kernel's 0.48 `edge_candidate_threshold`, threaded to `LinkerConfig.build`) and the detection-stage
    # per-frame retention guard (E6, revert a frame's blended detection to primary when its high-threshold peak
    # count collapses, threaded to `pipeline.nodes`). Both are the published kernel's exact behaviour; the four
    # fusion divergences and the divsub/motion cost replicas carry their OWN faithful flags on their nested
    # configs. Turn all of them on together for the full byte-faithful replica (see `frontier_replica`).
    kernel_faithful: bool = False

    @classmethod
    def shipped(cls) -> "TrackerConfig":
        """The operating point we actually SUBMIT — the recipe's one home, mounted rather than restated.

        The field defaults above are a neutral baseline (per-frame assignment linker, no fusion); this is the
        configuration the leaderboard scored 0.895. They differ, and that gap was a live measurement bias: both
        trainers built their selector as `TrackerConfig(threshold=…)`, grafting the shipped THRESHOLD onto
        otherwise-default everything, so every checkpoint this campaign selected was ranked under a linker we do
        not deploy. That is not bookkeeping — the sign of an affinity-side change depends on its consumer
        (bidirectional fusion: -0.0027 under the assignment linker, +0.004 on the leaderboard under flow), and
        checkpoint selection IS an affinity-side judgement, so a head that is better under flow could lose
        selection under assignment and simply never be saved.

        Naming it here rather than in the kernel is the point: a caller MOUNTS the recipe instead of restating
        four literals, so the selector cannot silently drift from deployment again.
        """
        return cls(
            linker=LinkerConfig(name="flow", gate_um=10.0, affinity_bonus=20.0, disappearance_cost=3.0),
            bidirectional_edges=True,
            # Divisions are part of the SUBMITTED recipe (0.899 against the 0.895 base), so the selector runs
            # them too — the same argument as the linker above, and the reason this method exists. Every field
            # of `AffinityDivisionConfig` now defaults to the submitted value, so the recipe is the type rather
            # than a list of literals: symmetry ranking, both probability floors off, both gates derived from
            # the linker's own, and the budget derived from the measured division rate (a leaderboard-confirmed
            # removal — caps of ~1072, ~400 and the derived ~157 forks all scored 0.899).
            division=AffinityDivisionConfig(),
        )

    @classmethod
    def frontier_replica(cls) -> "TrackerConfig":
        """The genuine 0.926 stack as one operating point — the shipped recipe with all five frontier arms ON.

        The one home the kernel and any local eval both mount, so the replica is a single definition rather than
        a list of literals restated per call site (the same argument `shipped` makes). It is `shipped` plus:

        * the calibrated dual-seed FUSION — both heads flipped as `proxy_eval._with_calibrated_fusion` does: the
          edge blend to the margin-gated `CalibratedFusionOptions` and the detector blend to the std-ratio-aligned
          mix that gives the secondary seed the frontier's 0.475 weight (so `detector_blend`, seed 1's share, is
          its complement) with moment alignment on. The second seed is mounted by `MultiGpuSubmission.secondary_pack`;
        * a detection THRESHOLD between the shipped 0.97 and the high-precision 0.99;
        * the divsub recovery stage in its BYTE-FAITHFUL form (`kernel_faithful`) — the public kernel's own gate
          bracket, geometric candidacy and ranking, with C1 (mid-track parent), one-directional mutual-nearest,
          C3 single-successor t+2 divergence and the per-frame + global two-cap. This is what scored 0.926; the
          per-flag `require_*` bracket is our own looser recovery and diverges (the fidelity oracle proved it
          fabricates forks off track-starts and drops the cap floor), so the replica takes the faithful switch;
        * the MOTION relink linker, gating tight-then-loose at the annotated one-frame displacement bounds.

        DeepCenter (the fifth arm) is the mounted `center_pack`, orthogonal to this config. Gated on the fidelity
        oracle, and on the published 0.926 precedent — never on the saturated local proxy, which inverts above 0.89.
        """
        base = cls.shipped()
        # shipped always carries the division stage; the fallback keeps the type honest without an assert.
        division = base.division or AffinityDivisionConfig()
        return replace(
            base,
            threshold=_FRONTIER_THRESHOLD,
            detector_blend=1.0 - _SECONDARY_DETECTOR_WEIGHT,
            detector_align_moments=True,
            edge_options=base.edge_options.model_copy(update={"calibrated_fusion": CalibratedFusionOptions()}),
            division=division.model_copy(update={"kernel_faithful": True}),
            linker=LinkerConfig(name="motion", tight_um=_MOTION_TIGHT_UM, gate_um=_MOTION_LOOSE_UM),
        )

    @classmethod
    def kernel_faithful_replica(cls) -> "TrackerConfig":
        """The FULL byte-faithful replica: `frontier_replica` with every one of the six fidelity switches ON.

        `frontier_replica` is the genuine 0.926 recipe with the divsub stage already in its faithful form; this
        is the single incantation that also flips the other five, so the whole stack reproduces the published
        kernel exactly rather than our shipped approximations:

        * the FUSION divergences (E1-E4) — `edge_options.calibrated_fusion.kernel_faithful`;
        * the linker edge-candidate ADMISSION floor (E5) and the detection retention GUARD (E6) — this config's
          own `kernel_faithful`, threaded to the linker's `edge_admission_prob` and the detector's `nodes`;
        * the MOTION relink COST replica — `linker.kernel_faithful`;
        * the divsub recovery replica — `division.kernel_faithful`, already on from `frontier_replica`.
        """
        base = cls.frontier_replica()
        fusion = base.edge_options.calibrated_fusion
        if fusion is None:
            raise ValueError("frontier_replica must mount the calibrated fusion before it can be made faithful")
        return replace(
            base,
            kernel_faithful=True,
            edge_options=base.edge_options.model_copy(
                update={"calibrated_fusion": fusion.model_copy(update={"kernel_faithful": True})}
            ),
            linker=base.linker.model_copy(update={"kernel_faithful": True}),
            # evgendvorkin's published 0.926 kernel closes two-frame gaps (GAP_CLOSE_MAX_GAP=2); our replica
            # left the span at 1. The GT-free fire-check confirms it recovers +81 edges on the dense movie.
            bridge=base.bridge.model_copy(update={"max_gap": 2}),
        )

    def __post_init__(self) -> None:
        """Reject an operating point that cannot mean what it says — the one config level that had no checks.

        Every nested config here is a validated pydantic model; this dataclass was the exception, so a typo at
        the top of the hierarchy constructed fine and misbehaved quietly. The blend is the sharp case: the
        seeds' LOGITS are combined as a raw weighted sum before the softmax, so weights that do not sum to one
        rescale the logits and shift the softmax TEMPERATURE — a different pipeline, not a different mix.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be a probability in [0, 1], got {self.threshold}")
        if self.min_track_length < 1:
            raise ValueError(f"min_track_length must keep at least one node, got {self.min_track_length}")
        if not math.isclose(sum(self.edge_blend), 1.0, abs_tol=1e-6):
            raise ValueError(f"edge_blend must sum to 1 (it scales logits before the softmax), got {self.edge_blend}")
        if self.detector_blend is not None and not 0.0 <= self.detector_blend <= 1.0:
            raise ValueError(f"detector_blend is seed1's share in [0, 1], got {self.detector_blend}")
        if not 0.0 <= self.smooth_strength <= 1.0:
            raise ValueError(f"smooth_strength must be in [0, 1], got {self.smooth_strength}")

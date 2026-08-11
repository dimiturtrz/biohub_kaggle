"""The tracker's operating point: every knob of the shipped recipe, as one object a sweep varies.

`TrackerConfig` is the vocabulary the whole repo speaks — the eval tools, the training entrypoints, the
submission dispatcher and the Kaggle kernels all name an operating point, and most of them never mount a
`CellTracker` at all. It therefore lives at the package root beside the runner rather than inside it, so
naming a recipe does not drag the mounted models' import graph along with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from celltrack.edges.blended_edge_scoring import EdgeBlendOptions
from celltrack.linkers.linkers import LinkerConfig
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig
from celltrack.postproc.gap_closer import BridgeConfig, ReuseConfig
from celltrack.postproc.short_track_filter import ShortTrackRescueConfig
from celltrack.postproc.topology_repair import TopologyConfig


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
    # Off by default: recovering the edge head's second-daughter fork is a detection-limited lever (the second
    # daughter is often undetected, so the fork cannot be placed). Set it to A/B the division-jaccard term.
    division: AffinityDivisionConfig | None = None
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

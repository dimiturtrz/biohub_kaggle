"""Score a trained detector through the *shipped* tracker on a 4-movie proxy — the leaderboard's own pipeline.

The detector trainer used to select checkpoints on a fold-0 motion-linker eval, a metric that does not transfer
(same weights: 0.837 motion-fold vs 0.893 real-pipeline). This evaluator closes that gap: it mounts the real
`CellTracker` (AssignmentLinker + blended edge affinity + gap bridge + short-track filter + smoother) over a
held-out set of movies — the tightest local stand-in for the hidden leaderboard — and scores the trained
detector through exactly that pipeline. Which movies is the caller's choice: the trainers select on the
densely-annotated validation four so the untouched test four stay a clean estimate. The edge affinity is
pilkwang's (independent of the detector under test), so it is mounted once and reused across every eval;
only the detector changes, wrapped as a single-seed
`BlendDetectorScorer` over an `EphemeralResponseStore` (the trainer's weights move every window — there is no
forward to cache). The score is the same `SplitScore` the proxy and the leaderboard report; node recall and the
node-count ratio ride alongside, the detector-side levers that show *how* the number moved.

A JOINT run trains both heads, so a pilkwang affinity would hide half of what it learned: `evaluate_joint`
mounts the model's OWN transformer (as a one-member logit blend) behind its own detector instead. Both paths
share the tracker assembly — only the edge affinity differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import EphemeralResponseStore
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.dense_diagnosis import AffinityIndex, MislinkSignal
from celltrack.eval.proxy import TestMovieProxy
from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from core.metrics.detection import NodeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics


@dataclass(frozen=True)
class EvalResult:
    """One eval's pooled numbers — the faithful LB metric, the clamped score a run selects on, and the node levers.

    `score` is the competition metric exactly, so a log line stays comparable to the leaderboard;
    `selection_score` is `SplitScore.selection_score`, the same number with the node-count factor capped at its
    neutral 1.0 so under-detection earns nothing (see `VideoMetrics.clamped_edge_jaccard`). Both ride along
    because a run should report the honest score AND what its checkpoint choice actually maximised.

    Node *recall* (over annotated cells) is the real detector lever; node precision against the sparse annotation
    is ~meaningless (most true detections land on unannotated cells), so the over-detection signal is the
    node-count ratio `(predicted - estimated) / estimated` instead — the same term the leaderboard penalises.

    The association levers are the `MislinkSignal` numbers, pooled over the proxy's movies: `mislinks` is how
    many annotated edges the linker got wrong on REAL detections with both endpoints detected, `inverted_fraction`
    the share of those the affinity itself ranked wrong (`p_chosen > p_true`) — a training defect, as against the
    remainder, where a nearer wrong neighbour beat a correctly-ranked successor on the distance term (a linker
    defect) — and `mean_p_true` / `mean_p_chosen` how far apart the two are scored where it fails. This is
    association measured where it is HARD; an AUC over ground-truth pairs is not (it reads ~1.0 untrained).
    BASELINE, dense movie post-NMS-fix: 35 mislinks, inverted_fraction 1.00, mean p_true 0.134 vs p_chosen 0.686.
    A run genuinely improving association moves those; a run that only moves the AUC has not.
    """

    score: float
    selection_score: float
    node_recall: float
    node_ratio: float
    inverted_fraction: float
    mean_p_true: float
    mean_p_chosen: float
    top1_fraction: float
    mislinks: int


@dataclass(frozen=True)
class ModelEvaluator:
    """The shipped-pipeline checkpoint selector: score a trained detector through a real `CellTracker` on the proxy.

    The proxy (whichever movies `mount` was given — the trainers pass the denser-annotated validation four, the
    default is the test four) and pilkwang's blended edge affinity are mounted once — both are independent of
    the detector under test — so a per-window eval only wraps the current weights into a fresh single-seed
    `BlendDetectorScorer` and runs the full tracker. `TrackerConfig` fixes the shipped operating point (the
    AssignmentLinker at gate10/bonus20, the 0.8/0.2 edge blend, min_track_length 6, smooth 0.8) at the trainer's
    detection threshold; `recipe` is the detector's input pipeline (downsample / TTA), matched to how it trained.
    """

    proxy: TestMovieProxy
    edge_scorer: BlendedEdgeTransformerScorer
    recipe: DetectorRecipe
    device: str
    config: TrackerConfig

    @classmethod
    def mount(
        cls, proxy: TestMovieProxy, packs: tuple[Path, Path], recipe: DetectorRecipe, device: str, config: TrackerConfig
    ) -> "ModelEvaluator":
        """Mount pilkwang's two-seed edge affinity behind the given proxy once, ready to score any detector.

        The edge affinity is pilkwang's own (the trained detector supplies only the nodes), so it is mounted here
        and shared across every window's eval rather than rebuilt per checkpoint. `packs` are the two seed packs.
        The caller chooses which movies the score runs over — the trainers load the proxy over `VALIDATION_MOVIES`,
        so selection pressure never lands on the test four.
        """
        edge_scorer = BlendedEdgeTransformerScorer.from_packs(
            packs, config.edge_blend, device, bidirectional=config.bidirectional_edges, options=config.edge_options
        )
        return cls(proxy=proxy, edge_scorer=edge_scorer, recipe=recipe, device=device, config=config)

    def evaluate(self, detector: TemporalUNetDetector) -> EvalResult:
        """Score the trained detector through the shipped tracker on the proxy — the LB-aligned selector number.

        The detector is wrapped as a single-seed `BlendDetectorScorer` (ephemeral store — the weights move every
        window, so nothing is cached) and dropped into a `CellTracker` behind the pre-mounted edge affinity. The
        split score is the leaderboard's own metric; node recall and the node-count ratio come from the same
        per-movie run, so one tracker pass yields all three numbers the trainer logs and selects on.
        """
        return self._score(detector, self.edge_scorer)

    def evaluate_joint(self, model: JointModel) -> EvalResult:
        """Score a jointly-trained pair of heads — the trained detector AND its trained association head.

        `evaluate` mounts pilkwang's edge affinity, which makes the association half of a joint run invisible to
        the selector. Here both halves come from the model: the detector gets the peak read-out re-bound onto its
        live weights, and the transformer is mounted as a one-member logit blend (the type the tracker consumes),
        so the number the run selects on is the whole trained pipeline through the shipped tracker.
        """
        detector = TemporalUNetDetector.of(model.detector)
        scorer = EdgeTransformerScorer.of(detector, model.transformer, self.recipe)
        one_seed = BlendedEdgeTransformerScorer(
            (scorer,), (1.0,), bidirectional=self.config.bidirectional_edges, options=self.config.edge_options
        )
        return self._score(detector, one_seed)

    def evaluate_ensemble(
        self,
        model: JointModel,
        packs: tuple[Path, ...],
        weights: tuple[float, ...] | None = None,
        detector_packs: tuple[Path, ...] = (),
    ) -> EvalResult:
        """Score the model's association head BLENDED with published packs' heads — our weights as an extra seed.

        The detector stays the model's own; only the edge affinity becomes a blend, so what this measures is
        whether OUR head adds anything to theirs rather than whether our detector does. Each member computes its
        own node features from its own backbone, which is why a pack contributes a whole scorer rather than a
        tensor.

        `weights` defaults to EQUAL across members — the self-balancing choice, and the one that needs no
        argument. A tilt is the thing to measure, not to assume: the shipped two-seed 0.8/0.2 sits on a plateau
        so wide that 0.5/0.5 scores identically, so there is no precedent here for favouring anyone.

        The earlier null on this idea (a fine-tuned variant adding nothing to the two-seed blend) was measured
        under a bare `TrackerConfig` — the per-frame linker with no fusion — which is the instrument since shown
        to INVERT the ranking of our own checkpoints. It also used a checkpoint that loses under the shipped
        pipeline. Both are reasons the old number does not carry, not reasons to expect a win.
        """
        detector = TemporalUNetDetector.of(model.detector)
        ours = EdgeTransformerScorer.of(detector, model.transformer, self.recipe)
        published = tuple(EdgeTransformerScorer.from_pack(pack, self.device) for pack in packs)
        members = (*published, ours)
        blended = BlendedEdgeTransformerScorer(
            members,
            weights if weights is not None else tuple(1.0 / len(members) for _ in members),
            bidirectional=self.config.bidirectional_edges,
            options=self.config.edge_options,
        )
        return self._score(detector, blended, detector_packs)

    def _score(
        self,
        detector: TemporalUNetDetector,
        edge_scorer: BlendedEdgeTransformerScorer,
        detector_packs: tuple[Path, ...] = (),
    ) -> EvalResult:
        """One tracker pass per proxy movie, pooled into the split score and the detector's node levers.

        Scored with cudnn autotuning OFF, restoring it afterwards. The trainers enable `cudnn.benchmark` —
        correct for training, where the shape is static and the speedup is real — but it picks convolution
        algorithms by TIMING, so the same weights can take different kernels on different runs. Under fp16
        that shifts peak logits, which shifts which maxima survive the equality-NMS, which shifts node counts
        and the score: two evaluations of one untrained model measured 0.8474 and 0.8457. A selector that
        cannot reproduce itself cannot resolve a small improvement, and near a ceiling the improvements ARE
        small — so determinism is worth more here than the few seconds it costs.
        """
        autotune = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            return self._scored(detector, edge_scorer, detector_packs)
        finally:
            torch.backends.cudnn.benchmark = autotune

    def _scored(
        self,
        detector: TemporalUNetDetector,
        edge_scorer: BlendedEdgeTransformerScorer,
        detector_packs: tuple[Path, ...] = (),
    ) -> EvalResult:
        """The scoring pass itself — see `_score` for why it runs with autotuning disabled.

        The mislink signal rides this pass for free: `run_scored` hands back the affinity and the detections it
        was aligned to, both already computed for the linker, and the analysis over them is array algebra on the
        CPU. No second forward, no second tracker run — so it runs EVERY window, at every window's own weights.
        """
        tracker = self._tracker(detector, edge_scorer, detector_packs)
        matcher = DistanceMatcher(spacing=self.proxy.spacing)
        metrics: list[VideoMetrics] = []
        nodes: list[NodeCounts] = []
        signals: list[MislinkSignal] = []
        for path, truth in zip(self.proxy.paths, self.proxy.truths, strict=True):
            tracked = tracker.run_scored(path.name, path)
            matching = matcher.match(tracked.graph, truth.graph)
            metrics.append(VideoMetrics.of(tracked.graph, truth, matcher))
            nodes.append(NodeCounts.of(matching, len(truth.graph.node_ids)))
            index = AffinityIndex.of(tracked.detections, tracked.affinity)
            signals.append(MislinkSignal.of(tracked.graph, truth.graph, matching, index))
        predicted = sum(metric.predicted_nodes for metric in metrics)
        estimated = sum(metric.estimated_nodes for metric in metrics)
        ratio = (predicted - estimated) / estimated if estimated > 0 else float("nan")
        split = SplitScore.of(metrics)
        signal = MislinkSignal.pooled(signals)
        mean_true, mean_chosen = signal.means()
        return EvalResult(
            score=split.score,
            selection_score=split.selection_score,
            node_recall=NodeCounts.pooled(nodes).recall(),
            node_ratio=ratio,
            inverted_fraction=signal.inverted_fraction(),
            mean_p_true=mean_true,
            mean_p_chosen=mean_chosen,
            top1_fraction=signal.top1_fraction(),
            mislinks=len(signal.p_true),
        )

    def _tracker(
        self,
        detector: TemporalUNetDetector,
        edge_scorer: BlendedEdgeTransformerScorer,
        detector_packs: tuple[Path, ...] = (),
    ) -> CellTracker:
        """Mount the trained detector into the shipped tracker behind the given edge affinity — one assembly.

        `detector_packs` add PUBLISHED detectors beside ours, blended in logit space by `BlendDetectorScorer`
        exactly as the shipped dual-seed submission blends its two. Empty is the single-detector pipeline every
        number before this was measured under, so it is inert by default.
        """
        # `from_pack` hands back the detector AND the recipe it was published with; the recipe is dropped
        # deliberately, because every member of a blend must be read out on ONE grid — ours — or the logit
        # volumes would not be voxel-aligned to average.
        published = tuple(
            (TemporalUNetDetector.from_pack(pack, map_location=self.device)[0].to(self.device).eval(), store)
            for pack, store in ((pack, EphemeralResponseStore()) for pack in detector_packs)
        )
        scorer = BlendDetectorScorer(
            detectors=((detector, EphemeralResponseStore()), *published), recipe=self.recipe, device=self.device
        )
        return CellTracker(detector=scorer, edge_scorer=edge_scorer, device=self.device, config=self.config)

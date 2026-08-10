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

from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import EphemeralResponseStore
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.proxy import TestMovieProxy
from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.tracker import CellTracker, TrackerConfig
from core.metrics.detection import NodeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics


@dataclass(frozen=True)
class EvalResult:
    """One eval's pooled numbers — the LB-aligned split score the run selects on, plus the detector's own levers.

    Node *recall* (over annotated cells) is the real detector lever; node precision against the sparse annotation
    is ~meaningless (most true detections land on unannotated cells), so the over-detection signal is the
    node-count ratio `(predicted - estimated) / estimated` instead — the same term the leaderboard penalises.
    """

    score: float
    node_recall: float
    node_ratio: float


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
            packs, config.edge_blend, device, bidirectional=config.bidirectional_edges
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
        one_seed = BlendedEdgeTransformerScorer((scorer,), (1.0,), bidirectional=self.config.bidirectional_edges)
        return self._score(detector, one_seed)

    def _score(self, detector: TemporalUNetDetector, edge_scorer: BlendedEdgeTransformerScorer) -> EvalResult:
        """One tracker pass per proxy movie, pooled into the split score and the detector's node levers."""
        tracker = self._tracker(detector, edge_scorer)
        matcher = DistanceMatcher(spacing=self.proxy.spacing)
        metrics: list[VideoMetrics] = []
        nodes: list[NodeCounts] = []
        for path, truth in zip(self.proxy.paths, self.proxy.truths, strict=True):
            graph = tracker.run(path.name, path)
            metrics.append(VideoMetrics.of(graph, truth, matcher))
            nodes.append(NodeCounts.of(matcher.match(graph, truth.graph), len(truth.graph.node_ids)))
        predicted = sum(metric.predicted_nodes for metric in metrics)
        estimated = sum(metric.estimated_nodes for metric in metrics)
        ratio = (predicted - estimated) / estimated if estimated > 0 else float("nan")
        return EvalResult(SplitScore.of(metrics).score, NodeCounts.pooled(nodes).recall(), ratio)

    def _tracker(self, detector: TemporalUNetDetector, edge_scorer: BlendedEdgeTransformerScorer) -> CellTracker:
        """Mount the trained detector into the shipped tracker behind the given edge affinity — one assembly."""
        scorer = BlendDetectorScorer(
            detectors=((detector, EphemeralResponseStore()),), recipe=self.recipe, device=self.device
        )
        return CellTracker(detector=scorer, edge_scorer=edge_scorer, device=self.device, config=self.config)

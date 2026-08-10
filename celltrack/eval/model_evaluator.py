"""Score a trained detector through the *shipped* tracker on the 4-movie proxy — the leaderboard's own pipeline.

The detector trainer used to select checkpoints on a fold-0 motion-linker eval, a metric that does not transfer
(same weights: 0.837 motion-fold vs 0.893 real-pipeline). This evaluator closes that gap: it mounts the real
`CellTracker` (AssignmentLinker + blended edge affinity + gap bridge + short-track filter + smoother) over the
four competition test movies — the tightest local stand-in for the hidden leaderboard — and scores the trained
detector through exactly that pipeline. The edge affinity is pilkwang's (independent of the detector under test),
so it is mounted once and reused across every eval; only the detector changes, wrapped as a single-seed
`BlendDetectorScorer` over an `EphemeralResponseStore` (the trainer's weights move every window — there is no
forward to cache). The score is the same `SplitScore` the proxy and the leaderboard report; node recall and the
node-count ratio ride alongside, the detector-side levers that show *how* the number moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import EphemeralResponseStore
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.proxy import TEST_MOVIES, TestMovieProxy
from celltrack.tracker import CellTracker, TrackerConfig
from core.metrics.detection import NodeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot


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

    The proxy (four test movies) and pilkwang's blended edge affinity are mounted once — both are independent of
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
        cls, root: DataRoot, packs: tuple[Path, Path], recipe: DetectorRecipe, device: str, config: TrackerConfig
    ) -> "ModelEvaluator":
        """Load the four-movie proxy and pilkwang's two-seed edge affinity once, ready to score any detector.

        The edge affinity is pilkwang's own (the trained detector supplies only the nodes), so it is mounted here
        and shared across every window's eval rather than rebuilt per checkpoint. `packs` are the two seed packs.
        """
        proxy = TestMovieProxy.load(root, TEST_MOVIES)
        edge_scorer = BlendedEdgeTransformerScorer.from_packs(packs, config.edge_blend, device)
        return cls(proxy=proxy, edge_scorer=edge_scorer, recipe=recipe, device=device, config=config)

    def evaluate(self, detector: TemporalUNetDetector) -> EvalResult:
        """Score the trained detector through the shipped tracker on the proxy — the LB-aligned selector number.

        The detector is wrapped as a single-seed `BlendDetectorScorer` (ephemeral store — the weights move every
        window, so nothing is cached) and dropped into a `CellTracker` behind the pre-mounted edge affinity. The
        split score is the leaderboard's own metric; node recall and the node-count ratio come from the same
        per-movie run, so one tracker pass yields all three numbers the trainer logs and selects on.
        """
        tracker = self._tracker(detector)
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

    def _tracker(self, detector: TemporalUNetDetector) -> CellTracker:
        """Mount the trained detector into the shipped tracker behind the pre-mounted pilkwang edge affinity."""
        scorer = BlendDetectorScorer(
            detectors=((detector, EphemeralResponseStore()),), recipe=self.recipe, device=self.device
        )
        return CellTracker(detector=scorer, edge_scorer=self.edge_scorer, device=self.device, config=self.config)

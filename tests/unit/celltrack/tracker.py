from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from celltrack import tracker as tracker_module
from celltrack.detectors.pipeline import BlendDetectorScorer, DetectionFusionOptions
from celltrack.detectors.response_cache import EphemeralResponseStore
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import EdgeBlendOptions
from celltrack.linkers.linkers import LinkerConfig
from celltrack.linkers.linking import Linker
from celltrack.linkers.motion_linking import _KERNEL_EDGE_ADMISSION_THRESHOLD, MotionHungarianLinker
from celltrack.operating_point import TrackerConfig
from celltrack.postproc.topology_repair import TopologyConfig
from celltrack.tracker import CellTracker, LinkerStage
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_CHAIN = TrackGraph(
    node_ids=np.array([0, 1], dtype=np.int64),
    coordinates=np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int64),
    edges=np.empty((0, 2), dtype=np.int64),
)


@dataclass
class _Video:
    spacing: Spacing
    volume_shape: tuple[int, int, int] = (4, 4, 4)


class _StubDetector:
    """A detector whose forward is faked, so `from_packs` mounts without weights or a GPU."""

    def to(self, device: str) -> "_StubDetector":
        return self

    def eval(self) -> "_StubDetector":
        return self


class _StubBlend:
    """A blend detector that returns a fixed two-cell chain, so the linker + post-proc run without a forward."""

    def nodes(
        self,
        video_key: str,
        path: Path,
        threshold: float,
        weights: tuple[float, ...] | None = None,
        *,
        fusion: DetectionFusionOptions | None = None,
    ) -> TrackGraph:
        return _CHAIN


class _StubEdgeScorer:
    """A blended edge scorer supplying no learned affinity, leaving the linker on pure geometry.

    It carries the blend `options` it was re-pointed with, so a config-path test can assert which transforms
    the tracker asked its scorer for without mounting weights.
    """

    def __init__(self, options: EdgeBlendOptions | None = None, weights: tuple[float, ...] = ()) -> None:
        self.options = options if options is not None else EdgeBlendOptions()
        self.weights = weights

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> None:
        return None

    def with_bidirectional(self, *, bidirectional: bool) -> "_StubEdgeScorer":
        return self

    def with_options(self, options: EdgeBlendOptions) -> "_StubEdgeScorer":
        return _StubEdgeScorer(options, self.weights)

    def with_weights(self, weights: tuple[float, ...]) -> "_StubEdgeScorer":
        return _StubEdgeScorer(self.options, weights)


def _tracker(monkeypatch: pytest.MonkeyPatch, config: TrackerConfig) -> CellTracker:
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    return CellTracker(
        detector=cast(BlendDetectorScorer, _StubBlend()),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        device="cpu",
        config=config,
    )


def test_run(monkeypatch: pytest.MonkeyPatch):
    """`run` folds detection -> link -> post-processing into one linked track graph for a two-cell chain."""
    tracker = _tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0))
    graph = tracker.run("m.zarr", Path("m.zarr"))
    assert isinstance(graph, TrackGraph)
    assert graph.edges.tolist() == [[0, 1]]


def test_run_scored(monkeypatch: pytest.MonkeyPatch):
    """`run_scored` returns the same graph `run` does, beside the detections and affinity it was built from.

    The affinity and the DETECTION node set are what a post-hoc mislink diagnosis needs, and both are already
    computed for the linker — handing them back is what makes that diagnosis free rather than a second pass.
    """
    tracker = _tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0))
    tracked = tracker.run_scored("m.zarr", Path("m.zarr"))
    assert tracked.graph.edges.tolist() == [[0, 1]]
    assert tracked.detections.node_ids.tolist() == [0, 1]  # pre-post-processing, the rows the affinity is aligned to
    assert tracked.affinity is tracker.edge_scorer.affinities(Path("m.zarr"), _CHAIN, "cpu")


def test_run_with_topology_repair(monkeypatch: pytest.MonkeyPatch):
    """A present `topology` config folds the invariant-enforcing stage in without breaking a valid chain."""
    tracker = _tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0, topology=TopologyConfig()))
    graph = tracker.run("m.zarr", Path("m.zarr"))
    assert graph.edges.tolist() == [[0, 1]]  # the single consecutive, single-parent edge survives the repair


def test_run_hands_the_linker_the_volume_shape(monkeypatch: pytest.MonkeyPatch):
    """The imaged extent reaches the linker: a boundary prior refuses to build without it, so this run would fail.

    Where the volume ENDS is the one fact about the imaging a detection graph cannot carry, and the tracker is
    the only place that has already opened the video.
    """
    linker = LinkerConfig(name="flow", boundary_prior=True)
    tracker = _tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0, linker=linker))
    assert tracker.run("m.zarr", Path("m.zarr")).node_ids.tolist() == [0, 1]  # built and ran; no shape = ValueError


def test_kernel_faithful_threads_the_admission_floor_and_the_detection_guard(monkeypatch: pytest.MonkeyPatch):
    """`TrackerConfig.kernel_faithful` reaches BOTH ends: the linker's 0.48 admission floor (E5) and the

    detector's per-frame retention guard (E6). One toggle, threaded to the two divergences that live on the
    tracker's own path rather than in a nested config — the plumbing assertion that the flag actually lands.
    """
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    blend = _CountingBlend(_CHAIN)
    config = TrackerConfig(
        linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0),
        kernel_faithful=True,
        min_track_length=1,
        smooth_strength=0.0,
    )
    tracker = CellTracker(
        detector=cast(BlendDetectorScorer, blend),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        device="cpu",
        config=config,
    )
    tracker.run("m.zarr", Path("m.zarr"))

    assert blend.kernel_faithful_seen == [True]  # E6: the detection guard flag reached `nodes`
    link_stage = tracker._stages(Spacing(1.0, 1.0, 1.0), affinity=None)[0]
    assert isinstance(link_stage, LinkerStage)
    assert cast(MotionHungarianLinker, link_stage.linker).edge_admission_prob == _KERNEL_EDGE_ADMISSION_THRESHOLD


def test_kernel_faithful_off_leaves_both_paths_untouched(monkeypatch: pytest.MonkeyPatch):
    """With the toggle off the detector sees `False` and the linker carries no admission floor — byte-identical."""
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    blend = _CountingBlend(_CHAIN)
    config = TrackerConfig(
        linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0), min_track_length=1, smooth_strength=0.0
    )
    tracker = CellTracker(
        detector=cast(BlendDetectorScorer, blend),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        device="cpu",
        config=config,
    )
    tracker.run("m.zarr", Path("m.zarr"))

    assert blend.kernel_faithful_seen == [False]  # E6 off
    link_stage = tracker._stages(Spacing(1.0, 1.0, 1.0), affinity=None)[0]
    assert isinstance(link_stage, LinkerStage)
    assert cast(MotionHungarianLinker, link_stage.linker).edge_admission_prob is None  # E5 off


def test_with_config(monkeypatch: pytest.MonkeyPatch):
    """`with_config` swaps the operating point while keeping the same mounted models."""
    tracker = _tracker(monkeypatch, TrackerConfig())
    swapped = tracker.with_config(TrackerConfig(linker=LinkerConfig(disappearance_cost=5.0)))
    assert swapped.config.linker.disappearance_cost == 5.0
    assert swapped.detector is tracker.detector


def test_with_config_repoints_the_edge_blend_options(monkeypatch: pytest.MonkeyPatch):
    """The blend's candidate transforms reach the mounted scorer THROUGH the config — the sweep's only path.

    Both mechanisms (`align_seed_moments`, `view_tta`) were reachable only by constructing the scorer directly,
    so no measurement path could turn them on. This asserts the config route: default OFF, and a config that
    asks for them re-points the same mounted seeds rather than remounting.
    """
    tracker = _tracker(monkeypatch, TrackerConfig())
    assert cast(_StubEdgeScorer, tracker.with_config(TrackerConfig()).edge_scorer).options == EdgeBlendOptions()

    asked = TrackerConfig(edge_options=EdgeBlendOptions(align_seed_moments=True, view_tta=True))
    scorer = cast(_StubEdgeScorer, tracker.with_config(asked).edge_scorer)
    assert (scorer.options.align_seed_moments, scorer.options.view_tta) == (True, True)


def test_from_packs(monkeypatch: pytest.MonkeyPatch):
    """`from_packs` mounts both detector packs (cache-backed) and the blended edge scorer, carrying the config."""
    monkeypatch.setattr(
        tracker_module.TemporalUNetDetector,
        "from_pack",
        classmethod(lambda cls, pack, map_location: (_StubDetector(), DetectorRecipe())),
    )
    monkeypatch.setattr(
        tracker_module.BlendedEdgeTransformerScorer,
        "from_packs",
        staticmethod(lambda packs, weights, device, *, bidirectional, options: _StubEdgeScorer(options)),
    )
    config = TrackerConfig(
        linker=LinkerConfig(disappearance_cost=3.0), edge_options=EdgeBlendOptions(align_seed_moments=True)
    )
    tracker = CellTracker.from_packs(Path("p1"), Path("p2"), Path("cache"), "cpu", config)
    assert tracker.config.linker.disappearance_cost == 3.0
    assert isinstance(tracker.detector, BlendDetectorScorer)
    # The mount reads the config's blend options — the other half of the path a sweep drives them through.
    assert cast(_StubEdgeScorer, tracker.edge_scorer).options.align_seed_moments


def test_ephemeral(monkeypatch: pytest.MonkeyPatch):
    """`ephemeral` mounts the same pair of packs with a non-persisting store per seed and no responses dir."""
    monkeypatch.setattr(
        tracker_module.TemporalUNetDetector,
        "from_pack",
        classmethod(lambda cls, pack, map_location: (_StubDetector(), DetectorRecipe())),
    )
    monkeypatch.setattr(
        tracker_module.BlendedEdgeTransformerScorer,
        "from_packs",
        staticmethod(lambda packs, weights, device, *, bidirectional, options: _StubEdgeScorer(options)),
    )
    tracker = CellTracker.ephemeral(Path("p1"), Path("p2"), "cpu", TrackerConfig(threshold=0.97))

    assert tracker.config.threshold == 0.97
    assert [type(store) for _, store in tracker.detector.detectors] == [EphemeralResponseStore] * 2


def test_from_joint(monkeypatch: pytest.MonkeyPatch):
    """`from_joint` mounts both heads from ONE joint checkpoint as the single-seed one-member blend."""
    model = SimpleNamespace(detector=object(), transformer=object(), downsample=(1, 4, 4))
    monkeypatch.setattr(tracker_module.JointModel, "from_checkpoint", classmethod(lambda cls, path, device: model))
    monkeypatch.setattr(tracker_module.TemporalUNetDetector, "of", classmethod(lambda cls, net: _StubDetector()))
    monkeypatch.setattr(
        tracker_module.EdgeTransformerScorer,
        "of",
        classmethod(lambda cls, detector, transformer, recipe: object()),
    )
    captured: dict[str, object] = {}

    def _blend(
        members: tuple[object, ...],
        weights: tuple[float, ...],
        *,
        bidirectional: bool,
        options: EdgeBlendOptions,
    ) -> _StubEdgeScorer:
        captured["members"], captured["weights"] = members, weights
        return _StubEdgeScorer(options)

    monkeypatch.setattr(tracker_module, "BlendedEdgeTransformerScorer", _blend)

    tracker = CellTracker.from_joint(Path("dw0.pt"), DetectorRecipe(), "cpu", TrackerConfig(threshold=0.98))

    assert tracker.config.threshold == 0.98
    assert isinstance(tracker.detector, BlendDetectorScorer)
    assert [type(store) for _, store in tracker.detector.detectors] == [EphemeralResponseStore]
    assert captured["weights"] == (1.0,)
    assert len(cast(tuple[object, ...], captured["members"])) == 1


def test_spacing(monkeypatch: pytest.MonkeyPatch):
    """`spacing` reads the video's voxel spacing from its OME metadata."""
    tracker = _tracker(monkeypatch, TrackerConfig())
    assert tracker.spacing(Path("m.zarr")) == Spacing(1.0, 1.0, 1.0)


def test_transform():
    """`LinkerStage.transform` adapts a linker's `link` to the graph-stage contract."""

    @dataclass(frozen=True)
    class _Linker:
        def link(self, graph: TrackGraph) -> TrackGraph:
            return TrackGraph(graph.node_ids, graph.coordinates, np.array([[0, 1]], dtype=np.int64))

    stage = LinkerStage(cast(Linker, _Linker()))
    linked = stage.transform(_CHAIN)
    assert linked.edges.tolist() == [[0, 1]]


class _FusedAffinity:
    """A gap scored high in both directions — what the agreement floor admits."""

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return np.ones((1, 1)) if timepoint == 0 else None


class _RecordingEdgeScorer:
    """An edge scorer that records the direction setting of every scoring pass it is asked for."""

    def __init__(self, scored: list[bool], *, bidirectional: bool) -> None:
        self.scored = scored
        self.bidirectional = bidirectional

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> _FusedAffinity:
        self.scored.append(self.bidirectional)
        return _FusedAffinity()

    def with_bidirectional(self, *, bidirectional: bool) -> "_RecordingEdgeScorer":
        return _RecordingEdgeScorer(self.scored, bidirectional=bidirectional)

    def with_options(self, options: EdgeBlendOptions) -> "_RecordingEdgeScorer":
        return self


def _recording_tracker(monkeypatch: pytest.MonkeyPatch, config: TrackerConfig) -> tuple[CellTracker, list[bool]]:
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    scored: list[bool] = []
    scorer = _RecordingEdgeScorer(scored, bidirectional=config.bidirectional_edges)
    tracker = CellTracker(
        detector=cast(BlendDetectorScorer, _StubBlend()),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, scorer),
        device="cpu",
        config=config,
    )
    return tracker, scored


def test_run_scores_the_fused_affinity_for_an_agreement_floor(monkeypatch: pytest.MonkeyPatch):
    """A linker agreement floor makes the tracker score the gaps a second time, backwards — the gate's input.

    The cost still reads the directional pass; the reversed pass exists only to be gated on, which is why it is
    scored only when a floor is set.
    """
    config = TrackerConfig(min_track_length=1, smooth_strength=0.0, linker=LinkerConfig(agreement_floor=0.5))
    tracker, scored = _recording_tracker(monkeypatch, config)

    graph = tracker.run("m.zarr", Path("m.zarr"))
    assert scored == [False, True]  # the directional pass for the cost, then the fused one for the gate
    assert graph.edges.tolist() == [[0, 1]]  # a pair both directions agree on stays admissible


def test_run_reuses_the_bidirectional_affinity_for_the_gate(monkeypatch: pytest.MonkeyPatch):
    """Under `bidirectional_edges` the mounted affinity IS the fused one, so the gate reads it — no second pass."""
    config = TrackerConfig(
        min_track_length=1, smooth_strength=0.0, bidirectional_edges=True, linker=LinkerConfig(agreement_floor=0.5)
    )
    tracker, scored = _recording_tracker(monkeypatch, config)

    tracker.run("m.zarr", Path("m.zarr"))
    assert scored == [True]


def test_fused_affinities(monkeypatch: pytest.MonkeyPatch):
    """The fusion rule, called directly: off when nothing reads it, the mounted affinity under `bidirectional_edges`,
    and a second reversed pass otherwise. It is public because `celltrack.analysis.ranker_distribution` reconstructs
    the tracker's first linking pass and must resolve the fused affinity by exactly this rule, not a copy of it."""
    forward = _FusedAffinity()

    unused, _ = _recording_tracker(monkeypatch, TrackerConfig())
    assert unused.fused_affinities(Path("m.zarr"), _CHAIN, forward) is None

    reused, scored = _recording_tracker(
        monkeypatch, TrackerConfig(bidirectional_edges=True, linker=LinkerConfig(agreement_floor=0.5))
    )
    assert reused.fused_affinities(Path("m.zarr"), _CHAIN, forward) is forward
    assert scored == []  # the mounted affinity already IS the fused one

    rescored, passes = _recording_tracker(monkeypatch, TrackerConfig(linker=LinkerConfig(agreement_floor=0.5)))
    assert rescored.fused_affinities(Path("m.zarr"), _CHAIN, forward) is not forward
    assert passes == [True]


def test_run_without_a_floor_scores_once(monkeypatch: pytest.MonkeyPatch):
    """No floor, no fused pass: the default tracker pays for exactly the one scoring pass it always did."""
    tracker, scored = _recording_tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0))

    tracker.run("m.zarr", Path("m.zarr"))
    assert scored == [False]


# One source at the origin and two candidates 1 um and 4 um away, both inside the 10 um gate. Distance alone
# picks the near one; a re-ranker confident about the far one is the only thing that can move the link.
_FORK = TrackGraph(
    node_ids=np.array([0, 1, 2], dtype=np.int64),
    coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 4]], dtype=np.int64),
    edges=np.empty((0, 2), dtype=np.int64),
)


class _CountingBlend:
    """A blend detector returning the fork graph and counting its forwards — the two-pass reuse check."""

    def __init__(self, graph: TrackGraph = _FORK) -> None:
        self.graph = graph
        self.calls = 0
        self.kernel_faithful_seen: list[bool] = []

    def nodes(
        self,
        video_key: str,
        path: Path,
        threshold: float,
        weights: tuple[float, ...] | None = None,
        *,
        fusion: DetectionFusionOptions | None = None,
    ) -> TrackGraph:
        self.calls += 1
        self.kernel_faithful_seen.append(bool(fusion and fusion.kernel_faithful))
        return self.graph


class _CountingEdgeScorer:
    """An edge scorer supplying no affinity and counting its scoring passes."""

    def __init__(self) -> None:
        self.calls = 0

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> None:
        self.calls += 1
        return None

    def with_bidirectional(self, *, bidirectional: bool) -> "_CountingEdgeScorer":
        return self

    def with_options(self, options: EdgeBlendOptions) -> "_CountingEdgeScorer":
        return self


class _GapAffinity:
    """One gap's fixed probability matrix, in the ascending node order the linker's cost is aligned to."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self.matrix if timepoint == 0 else None


class _ContrarianRanker:
    """A re-ranker that reads the CONTEXT graph and backs whichever candidate the first pass did NOT take.

    Its probability is a function of the context links alone, so a run whose second pass changes is proof the
    features were computed against pass 1's emitted edge list — the whole reason the path is two-pass.
    """

    def __init__(self) -> None:
        self.contexts: list[list[list[int]]] = []

    def affinities(
        self,
        spacing: Spacing,
        detections: TrackGraph,
        links: TrackGraph,
        affinity: object,
        gate_um: float,
    ) -> _GapAffinity:
        self.contexts.append(links.edges.tolist())
        taken = {target for _, target in links.edges.tolist()}
        spurned = next(node for node in detections.node_ids.tolist() if node not in taken and node != 0)
        matrix = np.zeros((1, 2))
        matrix[0, spurned - 1] = 1.0
        return _GapAffinity(matrix)


def _forked_tracker(
    monkeypatch: pytest.MonkeyPatch, config: TrackerConfig, ranker: _ContrarianRanker | None
) -> tuple[CellTracker, _CountingBlend, _CountingEdgeScorer]:
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    detector, scorer = _CountingBlend(), _CountingEdgeScorer()
    tracker = CellTracker(
        detector=cast(BlendDetectorScorer, detector),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, scorer),
        device="cpu",
        config=config,
        ranker=cast(tracker_module.AssociationRanker, ranker),
    )
    return tracker, detector, scorer


def _fork_config(ranker_bonus: float | None) -> TrackerConfig:
    return TrackerConfig(
        min_track_length=1,
        smooth_strength=0.0,
        linker=LinkerConfig(gate_um=10.0, ranker_bonus=ranker_bonus),
    )


def test_run_without_a_ranker_is_unchanged(monkeypatch: pytest.MonkeyPatch):
    """A mounted re-ranker that no bonus prices changes NOTHING — same edges, byte for byte, and never called.

    The two-pass path is opt-in through `linker.ranker_bonus`, so the shipped operating point must be reachable
    with the artifact mounted; a byte comparison is what makes "unchanged" a check rather than a claim.
    """
    plain, _, _ = _forked_tracker(monkeypatch, _fork_config(None), None)
    ranker = _ContrarianRanker()
    mounted, _, _ = _forked_tracker(monkeypatch, _fork_config(None), ranker)

    shipped = plain.run("m.zarr", Path("m.zarr"))
    with_artifact = mounted.run("m.zarr", Path("m.zarr"))

    assert shipped.edges.tobytes() == with_artifact.edges.tobytes()
    assert shipped.edges.tolist() == [[0, 1]]  # distance alone takes the near candidate
    assert ranker.contexts == []  # an unpriced ranker is not scored at all


def test_run_relinks_against_the_context_graph(monkeypatch: pytest.MonkeyPatch):
    """Priced, the re-ranker re-links: pass 1's edge list is what it scores, and pass 2 follows the new cost."""
    ranker = _ContrarianRanker()
    tracker, _, _ = _forked_tracker(monkeypatch, _fork_config(10.0), ranker)

    graph = tracker.run("m.zarr", Path("m.zarr"))

    assert ranker.contexts == [[[0, 1]]]  # scored ONCE, against the raw links of the first pass
    assert graph.edges.tolist() == [[0, 2]]  # 4 um - 10*1.0 beats 1 um - 0, so the far candidate wins


class _ForkAffinity:
    """An edge scorer whose one gap prefers the FAR candidate — so the pass-1 bonus decides what pass 1 links."""

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> _GapAffinity:
        return _GapAffinity(np.array([[0.1, 0.9]]))

    def with_bidirectional(self, *, bidirectional: bool) -> "_ForkAffinity":
        return self


def test_the_context_pass_is_invariant_to_the_split_of_one_budget(monkeypatch: pytest.MonkeyPatch):
    """Two splits of ONE learned-evidence budget hand the re-ranker the IDENTICAL context graph (bead ic44).

    The ranker's features are facts about pass 1's emitted edge list, so a context that moved with the weight
    under test would make "the score is monotone in ranker_bonus" indistinguishable from "the score is monotone
    in how degraded the graph its own features were read off". Here the affinity prefers the far candidate
    (4 um vs 1 um at P 0.9 vs 0.1), so a pass-1 bonus of 20 links it and a bonus of 3 does not — the contexts
    are equal only because the budget, not the split, sets that bonus.
    """
    contexts = []
    for affinity_bonus, ranker_bonus in ((20.0, 0.0), (3.0, 17.0)):
        ranker = _ContrarianRanker()
        config = TrackerConfig(
            min_track_length=1,
            smooth_strength=0.0,
            linker=LinkerConfig(gate_um=10.0, affinity_bonus=affinity_bonus, ranker_bonus=ranker_bonus),
        )
        tracker, _, _ = _forked_tracker(monkeypatch, config, ranker)
        tracker = replace(tracker, edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, _ForkAffinity()))
        tracker.run("m.zarr", Path("m.zarr"))
        contexts.append(ranker.contexts)

    assert contexts[0] == [[[0, 2]]]  # the full budget links the far, high-probability candidate
    assert contexts[0] == contexts[1]


def test_run_reuses_the_detections_and_affinity_across_both_passes(monkeypatch: pytest.MonkeyPatch):
    """The second pass re-LINKS only: one detection forward and one affinity pass serve both.

    That is what makes the path affordable (linking is ~11% of a video, detection plus affinity ~88%) and what
    makes it correct: features computed against a different detection set would describe a different graph.
    """
    tracker, detector, scorer = _forked_tracker(monkeypatch, _fork_config(10.0), _ContrarianRanker())

    tracker.run("m.zarr", Path("m.zarr"))

    assert (detector.calls, scorer.calls) == (1, 1)


def test_with_ranker(monkeypatch: pytest.MonkeyPatch):
    """`with_ranker` mounts the artifact from the directory the CALLER names — no path is written into the code."""
    mounted: list[Path] = []

    def _from_artifact(cls: type, root: Path) -> _ContrarianRanker:
        mounted.append(root)
        return _ContrarianRanker()

    monkeypatch.setattr(tracker_module.AssociationRanker, "from_artifact", classmethod(_from_artifact))
    tracker = _tracker(monkeypatch, TrackerConfig())

    elsewhere = tracker.with_ranker(Path("mounted/somewhere/else"))

    assert mounted == [Path("mounted/somewhere/else")]
    assert elsewhere.ranker is not None
    assert tracker.ranker is None  # the original tracker is untouched
    assert elsewhere.with_config(TrackerConfig(threshold=0.9)).ranker is elsewhere.ranker  # survives a sweep step


def test_with_config_repoints_the_blend_weights(monkeypatch: pytest.MonkeyPatch):
    """The blend MIX reaches the mounted scorer through the config — the knob that silently did not.

    `with_config` carried the fusion flag and the options but left the weights at whatever the mount was built
    with, so `--set edge_blend=…` moved `config.edge_blend` and nothing else. A sweep over the blend therefore
    scored ONE scorer repeatedly and read as a flat curve, which is how "0.8/0.2 and 0.5/0.5 are identical to
    four decimals" was produced. Asserting the config route is what makes the knob measurable at all.
    """
    tracker = _tracker(monkeypatch, TrackerConfig())

    swapped = tracker.with_config(TrackerConfig(edge_blend=(0.5, 0.5)))

    assert cast(_StubEdgeScorer, swapped.edge_scorer).weights == (0.5, 0.5)


def test_stage_breakdown_orders_the_fold_by_cost():
    """The breakdown names the slowest stage FIRST — the line exists to point at an optimisation target.

    `link+post` is logged as one pooled number, which was 11% of a pass when the fold was linking alone and
    is now the majority of it. A breakdown in call order would leave the reader doing the arithmetic; ordering
    by cost puts the stage that owns the time at the front, where it is read.
    """
    assert CellTracker._stage_breakdown([("Cheap", 0.5), ("Costly", 12.0), ("Middling", 3.0)]) == (
        "Costly 12.0s | Middling 3.0s | Cheap 0.5s"
    )


def test_stage_breakdown_of_an_empty_fold():
    """A tracker configured with no post-proc stages logs an empty breakdown rather than raising."""
    assert CellTracker._stage_breakdown([]) == ""

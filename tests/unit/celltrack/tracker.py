from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack import tracker as tracker_module
from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import EphemeralResponseStore
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.linkers.linkers import LinkerConfig
from celltrack.linkers.linking import Linker
from celltrack.postproc.topology_repair import TopologyConfig
from celltrack.tracker import CellTracker, LinkerStage, TrackerConfig
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
        self, video_key: str, path: Path, threshold: float, weights: tuple[float, ...] | None = None
    ) -> TrackGraph:
        return _CHAIN


class _StubEdgeScorer:
    """A blended edge scorer supplying no learned affinity, leaving the linker on pure geometry."""

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> None:
        return None

    def with_bidirectional(self, *, bidirectional: bool) -> "_StubEdgeScorer":
        return self


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


def test_with_config(monkeypatch: pytest.MonkeyPatch):
    """`with_config` swaps the operating point while keeping the same mounted models."""
    tracker = _tracker(monkeypatch, TrackerConfig())
    swapped = tracker.with_config(TrackerConfig(linker=LinkerConfig(disappearance_cost=5.0)))
    assert swapped.config.linker.disappearance_cost == 5.0
    assert swapped.detector is tracker.detector


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
        staticmethod(lambda packs, weights, device, *, bidirectional: _StubEdgeScorer()),
    )
    config = TrackerConfig(linker=LinkerConfig(disappearance_cost=3.0))
    tracker = CellTracker.from_packs(Path("p1"), Path("p2"), Path("cache"), "cpu", config)
    assert tracker.config.linker.disappearance_cost == 3.0
    assert isinstance(tracker.detector, BlendDetectorScorer)


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
        staticmethod(lambda packs, weights, device, *, bidirectional: _StubEdgeScorer()),
    )
    tracker = CellTracker.ephemeral(Path("p1"), Path("p2"), "cpu", TrackerConfig(threshold=0.97))

    assert tracker.config.threshold == 0.97
    assert [type(store) for _, store in tracker.detector.detectors] == [EphemeralResponseStore] * 2


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


def test_post_init():
    """An operating point that cannot mean what it says is refused at construction, not obeyed quietly."""
    with pytest.raises(ValueError, match="edge_blend must sum to 1"):
        TrackerConfig(edge_blend=(0.8, 0.8))  # rescales the logits -> shifts the softmax temperature
    with pytest.raises(ValueError, match="threshold must be a probability"):
        TrackerConfig(threshold=5.0)
    with pytest.raises(ValueError, match="min_track_length"):
        TrackerConfig(min_track_length=0)

    assert TrackerConfig().threshold == 0.97  # the LB-measured best, not the round number


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


def test_run_without_a_floor_scores_once(monkeypatch: pytest.MonkeyPatch):
    """No floor, no fused pass: the default tracker pays for exactly the one scoring pass it always did."""
    tracker, scored = _recording_tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0))

    tracker.run("m.zarr", Path("m.zarr"))
    assert scored == [False]

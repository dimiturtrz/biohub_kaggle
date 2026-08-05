import numpy as np

from celltrack.topology_repair import TopologyConfig, TopologyRepair
from core.data.tracks import TrackGraph
from core.geometry import Spacing


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def test_transform():
    """All four repairs at once: a frame-skip and an over-long edge go, a cell keeps its nearest parent,
    and every node the surviving edges no longer touch is pruned."""
    messy = graph(
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 100], [0, 0, 0, 2], [1, 0, 0, 20]],
        [[0, 1], [4, 1], [0, 2], [1, 2], [0, 5]],
    )
    repaired = TopologyRepair(spacing=Spacing(1.0, 1.0, 1.0), edge_max_um=14.0).transform(messy)
    assert set(repaired.node_ids.tolist()) == {0, 10, 20}
    assert repaired.edges.tolist() == [[0, 10], [10, 20]]


def test_build():
    """`TopologyConfig.build` carries `edge_max_um` into a `TopologyRepair` at the given spacing."""
    stage = TopologyConfig(edge_max_um=12.0).build(Spacing(1.0, 1.0, 1.0))
    assert isinstance(stage, TopologyRepair)
    assert stage.edge_max_um == 12.0

"""Cell-track graphs — the ground truth's GEFF store and the same shape as a prediction.

GEFF stores a directed graph: nodes carry `t, z, y, x` voxel indices, edges carry source and target
node ids, and a cell division is one node with two outgoing edges. Ground truth adds one fact a
prediction cannot have — the estimated count of real cells, annotated or not — which the metric's
over-detection penalty is measured against, so it lives on its own type rather than as a hole in the
graph.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import scipy.sparse as sp
import zarr
from jaxtyping import Float, Int

from core.geometry import Spacing

_AXIS_ORDER = ("t", "z", "y", "x")
_DIVISION_OUT_DEGREE = 2


class GeffExtra(TypedDict):
    """The competition's addition to the GEFF spec: how many cells the video really holds."""

    estimated_number_of_nodes: int


class GeffAttributes(TypedDict):
    """The subset of the GEFF metadata this project reads."""

    extra: GeffExtra


@dataclass(frozen=True)
class TrackGraph:
    """Cell centres and their temporal links — voxel indices throughout, `t, z, y, x` column order."""

    node_ids: Int[np.ndarray, "n"]
    coordinates: Int[np.ndarray, "n 4"]
    edges: Int[np.ndarray, "e 2"]

    def timepoints(self) -> Int[np.ndarray, "n"]:
        """The timepoint of every node."""
        return self.coordinates[:, 0]

    def positions(self) -> Int[np.ndarray, "n 3"]:
        """The `(z, y, x)` voxel index of every node."""
        return self.coordinates[:, 1:]

    def division_parents(self) -> Int[np.ndarray, "d"]:
        """Node ids with two or more outgoing edges — the divisions this graph asserts."""
        sources, counts = np.unique(self.edges[:, 0], return_counts=True)
        return sources[counts >= _DIVISION_OUT_DEGREE]

    def link_displacements(self, spacing: Spacing) -> Float[np.ndarray, "e"]:
        """How far each link moves per timepoint, in micrometres.

        This is the physical quantity a linker's search radius has to cover, and it is measured against
        the same 7 um the matcher uses — if cells routinely move further than the matching cutoff, a
        nearest-neighbour linker cannot reach the right partner however good the detector is.
        """
        physical = spacing.to_micrometres(self.positions())
        sources, targets = self.rows_of(self.edges[:, 0]), self.rows_of(self.edges[:, 1])
        elapsed = np.maximum(self.timepoints()[targets] - self.timepoints()[sources], 1)
        return np.linalg.norm(physical[targets] - physical[sources], axis=1) / elapsed

    def rows_of(self, ids: Int[np.ndarray, "k"]) -> Int[np.ndarray, "k"]:
        """Row index of each node id. Ids are arbitrary sparse integers, so this is a search, not indexing."""
        order = np.argsort(self.node_ids)
        return order[np.searchsorted(self.node_ids, ids, sorter=order)]

    def edge_rows(self) -> Int[np.ndarray, "e 2"]:
        """The edges as row indices rather than node ids, so they can index the node arrays directly."""
        return self.rows_of(self.edges.reshape(-1)).reshape(-1, 2)

    def without_edges(self) -> "TrackGraph":
        """The same nodes with their links dropped — what a detector hands a linker."""
        return TrackGraph(node_ids=self.node_ids, coordinates=self.coordinates, edges=self.edges[:0])


@dataclass(frozen=True)
class Adjacency:
    """Neighbour lookup over a track graph, in row space.

    Duplicate edges are kept rather than collapsed: the metric treats a node with two identical outgoing
    links as a fork, so degrees have to count links, not distinct neighbours.
    """

    edges: Int[np.ndarray, "e 2"]
    successors: dict[int, tuple[int, ...]]
    predecessors: dict[int, tuple[int, ...]]
    out_degrees: Int[np.ndarray, "n"]
    in_degrees: Int[np.ndarray, "n"]

    @classmethod
    def of(cls, graph: TrackGraph) -> "Adjacency":
        """Index a graph's links in both directions."""
        edges = graph.edge_rows()
        count = len(graph.node_ids)
        return cls(
            edges=edges,
            successors=cls._grouped(edges[:, 0], edges[:, 1], count),
            predecessors=cls._grouped(edges[:, 1], edges[:, 0], count),
            out_degrees=np.bincount(edges[:, 0], minlength=count),
            in_degrees=np.bincount(edges[:, 1], minlength=count),
        )

    def dividing_rows(self) -> Int[np.ndarray, "d"]:
        """Rows with at least two outgoing links — what the metric treats as a fork."""
        return np.flatnonzero(self.out_degrees >= _DIVISION_OUT_DEGREE)

    def components(self) -> Int[np.ndarray, "n"]:
        """A weakly-connected-component label per row: which lineage fragment each node belongs to."""
        count = len(self.out_degrees)
        links = sp.coo_array(
            (np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])),
            shape=(count, count),
        )
        return sp.csgraph.connected_components(links, directed=False)[1]

    @staticmethod
    def _grouped(
        origin: Int[np.ndarray, "e"],
        destination: Int[np.ndarray, "e"],
        count: int,
    ) -> dict[int, tuple[int, ...]]:
        """Group each row's link endpoints, in edge order."""
        grouped: dict[int, list[int]] = {row: [] for row in range(count)}
        for source, target in zip(origin.tolist(), destination.tolist(), strict=True):
            grouped[source].append(target)
        return {row: tuple(targets) for row, targets in grouped.items()}


@dataclass(frozen=True)
class AnnotatedTracks:
    """A GEFF ground-truth store: a sparse track graph plus the estimated true cell count."""

    graph: TrackGraph
    estimated_node_count: int

    @classmethod
    def from_geff(cls, store: Path) -> "AnnotatedTracks":
        """Open a GEFF store, reading the node props in a fixed `t, z, y, x` order."""
        attributes = cast(GeffAttributes, zarr.open_group(store, mode="r").attrs["geff"])
        axes = [np.asarray(zarr.open_array(store / "nodes/props" / axis / "values")) for axis in _AXIS_ORDER]
        graph = TrackGraph(
            node_ids=np.asarray(zarr.open_array(store / "nodes/ids")),
            coordinates=np.stack(axes, axis=1),
            edges=np.asarray(zarr.open_array(store / "edges/ids")).reshape(-1, 2),
        )
        return cls(graph=graph, estimated_node_count=attributes["extra"]["estimated_number_of_nodes"])

    @property
    def annotated_fraction(self) -> float:
        """What share of the estimated real cells is labelled — how much of the detection signal is supervised."""
        return len(self.graph.node_ids) / self.estimated_node_count

"""Dropping track fragments too short to be a real lineage — a post-link cleanup.

A jittery detector produces spurious detections that a linker wires into tiny, isolated tracks: a couple of
nodes joined by an edge that no real cell backs. Each such fragment is a run of false-positive nodes and
edges the metric scores against, and there are many of them. Removing every weakly-connected component
shorter than a minimum length discards those fragments wholesale while leaving real lineages untouched — the
single highest-impact piece of the public solutions' post-processing.

The one exception is a component that contains a division (a node with two children): mitoses are rare and
valuable to the score, and a real division near the start or end of a track can sit in a short component, so
those are kept whatever their length rather than thrown out with the noise.
"""

from dataclasses import dataclass

import numpy as np

from core.data.tracks import Adjacency, TrackGraph


@dataclass(frozen=True)
class ShortTrackFilter:
    """Drop weakly-connected track components below a minimum length, keeping any that contain a division."""

    min_length: int

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with short, division-free lineage fragments removed."""
        adjacency = Adjacency.of(graph)
        labels = adjacency.components()
        sizes = np.bincount(labels, minlength=len(graph.node_ids))
        keep_label = sizes >= self.min_length
        keep_label[labels[adjacency.dividing_rows()]] = True
        keep_row = keep_label[labels]

        edge_rows = graph.edge_rows()
        edge_kept = keep_row[edge_rows[:, 0]] & keep_row[edge_rows[:, 1]]
        return TrackGraph(
            node_ids=graph.node_ids[keep_row],
            coordinates=graph.coordinates[keep_row],
            edges=graph.edges[edge_kept],
        )

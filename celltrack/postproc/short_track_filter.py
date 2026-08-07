"""Dropping track fragments too short to be a real lineage — a post-link cleanup.

A jittery detector produces spurious detections that a linker wires into tiny, isolated tracks: a couple of
nodes joined by an edge that no real cell backs. Each such fragment is a run of false-positive nodes and
edges the metric scores against, and there are many of them. Removing every weakly-connected component
shorter than a minimum length discards those fragments wholesale while leaving real lineages untouched — the
single highest-impact piece of the public solutions' post-processing.

Two carve-outs keep a short component the length rule would drop. A component that contains a division (a node
with two children) is always kept — mitoses are rare and valuable, and a real division near a track end can
sit in a short component. And an optional confidence RESCUE (`ShortTrackRescue`) keeps a short component whose
edges the transformer scores highly and whose steps are tight: a genuinely short true track (a cell that
enters or leaves the field, or is annotated only briefly) looks exactly like this, and the blunt length rule
throws it out with the noise. The rescue is the frontier's lever for pairing an aggressive length cut with a
high-confidence recovery — recall the sparse proxy barely sees but the denser hidden annotation rewards.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict, Field

from celltrack.affinity import EdgeAffinity
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing


def _edge_probabilities(graph: TrackGraph, affinity: EdgeAffinity) -> Float[np.ndarray, "e"]:
    """The transformer's association probability for each of the graph's edges, aligned to `graph.edges`.

    Each edge's probability is read from its source frame's `s x t` matrix at the source and target's
    within-frame positions, so a component's mean edge probability can gate whether it is a confident track.
    """
    edges = graph.edge_rows()
    timepoints = graph.timepoints()
    probabilities = np.zeros(len(edges), dtype=np.float64)
    for timepoint in np.unique(timepoints[edges[:, 0]]) if len(edges) else np.empty(0, dtype=timepoints.dtype):
        matrix = affinity.probabilities(int(timepoint))
        if matrix is None:
            continue
        sources = np.flatnonzero(timepoints == timepoint)
        targets = np.flatnonzero(timepoints == timepoint + 1)
        for edge in np.flatnonzero(timepoints[edges[:, 0]] == timepoint):
            source_local = int(np.searchsorted(sources, edges[edge, 0]))
            target_local = int(np.searchsorted(targets, edges[edge, 1]))
            probabilities[edge] = matrix[source_local, target_local]
    return probabilities


@dataclass(frozen=True)
class ShortTrackRescue:
    """The confidence carve-out: which short components to keep because the head is sure and the steps are tight."""

    spacing: Spacing
    affinity: EdgeAffinity
    min_mean_probability: float
    max_mean_step_um: float
    max_length: int
    budget_fraction: float
    budget_cap: int

    def protected(
        self,
        graph: TrackGraph,
        labels: Int[np.ndarray, "n"],
        sizes: Int[np.ndarray, "c"],
        kept: Bool[np.ndarray, "c"],
    ) -> Bool[np.ndarray, "c"]:
        """The component labels to rescue: short, not already kept, confident, and tight — the tightest first."""
        edge_labels = labels[graph.edge_rows()[:, 0]]
        probabilities = _edge_probabilities(graph, self.affinity)
        steps = graph.link_displacements(self.spacing)
        eligible = (~kept) & (sizes > 0) & (sizes <= self.max_length)
        proposals: list[tuple[float, int]] = []
        for label in np.flatnonzero(eligible).tolist():
            in_component = edge_labels == label
            if not in_component.any():
                continue
            mean_probability = float(probabilities[in_component].mean())
            mean_step = float(steps[in_component].mean())
            if mean_probability >= self.min_mean_probability and mean_step <= self.max_mean_step_um:
                proposals.append((mean_probability, label))
        budget = min(int(self.budget_fraction * len(graph.node_ids)), self.budget_cap)
        protected = np.zeros(len(kept), dtype=bool)
        for _, label in sorted(proposals, reverse=True)[:budget]:
            protected[label] = True
        return protected


class ShortTrackRescueConfig(BaseModel):
    """The rescue's gates, resolved to a `ShortTrackRescue` once a video's edge affinity is in hand.

    Defaults follow the frontier's disclosed recovery (mean edge probability >= 0.90, mean step <= 2.75 um,
    length up to 5, budget min(0.6% of nodes, 60)) — an aggressive length cut earns back only the short tracks
    the head is confident about."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_mean_probability: float = Field(0.90, ge=0, le=1)
    max_mean_step_um: float = Field(2.75, gt=0)
    max_length: int = Field(5, gt=0)
    budget_fraction: float = Field(0.006, ge=0)
    budget_cap: int = Field(60, ge=0)

    def build(self, spacing: Spacing, affinity: EdgeAffinity) -> ShortTrackRescue:
        """The rescue stage wired with these gates and the video's edge-head probabilities."""
        return ShortTrackRescue(
            spacing=spacing,
            affinity=affinity,
            min_mean_probability=self.min_mean_probability,
            max_mean_step_um=self.max_mean_step_um,
            max_length=self.max_length,
            budget_fraction=self.budget_fraction,
            budget_cap=self.budget_cap,
        )


@dataclass(frozen=True)
class ShortTrackFilter:
    """Drop weakly-connected track components below a minimum length, keeping divisions and rescued short tracks."""

    min_length: int
    # Optional confidence carve-out: a short component the head scores highly and links tightly is kept rather
    # than dropped. None is the original blunt length rule (division carve-out only).
    rescue: ShortTrackRescue | None = None

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with short, division-free, low-confidence lineage fragments removed."""
        adjacency = Adjacency.of(graph)
        labels = adjacency.components()
        sizes = np.bincount(labels, minlength=len(graph.node_ids))
        keep_label = sizes >= self.min_length
        keep_label[labels[adjacency.dividing_rows()]] = True
        if self.rescue is not None:
            keep_label |= self.rescue.protected(graph, labels, sizes, keep_label)
        keep_row = keep_label[labels]

        edge_rows = graph.edge_rows()
        edge_kept = keep_row[edge_rows[:, 0]] & keep_row[edge_rows[:, 1]]
        return TrackGraph(
            node_ids=graph.node_ids[keep_row],
            coordinates=graph.coordinates[keep_row],
            edges=graph.edges[edge_kept],
        )

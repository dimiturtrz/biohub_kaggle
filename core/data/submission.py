"""The competition submission CSV, and its round-trip back to track graphs.

One flat table across all videos: `row_type` is `node` or `edge`, and every column a row does not use
holds `-1`. Node ids are per-video, so the graph identity is `(dataset, node_id)`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from core.data.tracks import TrackGraph

_UNUSED = -1
_NODE = "node"
_EDGE = "edge"
_ROW_SCHEMA = pl.Schema(
    {
        "dataset": pl.Utf8,
        "row_type": pl.Utf8,
        "node_id": pl.Int64,
        "t": pl.Int64,
        "z": pl.Int64,
        "y": pl.Int64,
        "x": pl.Int64,
        "source_id": pl.Int64,
        "target_id": pl.Int64,
    }
)
_SCHEMA = pl.Schema({"id": pl.Int64, **_ROW_SCHEMA})


@dataclass(frozen=True)
class Submission:
    """Predicted track graphs for a set of videos, keyed by the competition's dataset name."""

    graphs: dict[str, TrackGraph]

    def to_frame(self) -> pl.DataFrame:
        """Flatten to the submission table — nodes first, then edges, per video, with a global row id."""
        rows = pl.concat([self._video_rows(dataset, graph) for dataset, graph in sorted(self.graphs.items())])
        return rows.with_row_index("id").cast(_SCHEMA)

    def write_csv(self, destination: Path) -> Path:
        """Write the submission table where the competition expects it."""
        self.to_frame().write_csv(destination)
        return destination

    @classmethod
    def read_csv(cls, source: Path) -> "Submission":
        """Read a submission table back into per-video graphs — the inverse of `write_csv`."""
        table = pl.read_csv(source, schema_overrides=_SCHEMA)
        graphs = {
            dataset: cls._video_graph(rows) for (dataset,), rows in table.group_by(["dataset"], maintain_order=True)
        }
        return cls(graphs=graphs)

    @staticmethod
    def _video_rows(dataset: str, graph: TrackGraph) -> pl.DataFrame:
        nodes = pl.DataFrame(
            {
                "dataset": dataset,
                "row_type": _NODE,
                "node_id": graph.node_ids,
                "t": graph.coordinates[:, 0],
                "z": graph.coordinates[:, 1],
                "y": graph.coordinates[:, 2],
                "x": graph.coordinates[:, 3],
                "source_id": _UNUSED,
                "target_id": _UNUSED,
            }
        )
        edges = pl.DataFrame(
            {
                "dataset": dataset,
                "row_type": _EDGE,
                "node_id": _UNUSED,
                "t": _UNUSED,
                "z": _UNUSED,
                "y": _UNUSED,
                "x": _UNUSED,
                "source_id": graph.edges[:, 0],
                "target_id": graph.edges[:, 1],
            }
        )
        return pl.concat([nodes.cast(_ROW_SCHEMA), edges.cast(_ROW_SCHEMA)])

    @staticmethod
    def _video_graph(rows: pl.DataFrame) -> TrackGraph:
        nodes = rows.filter(pl.col("row_type") == _NODE)
        edges = rows.filter(pl.col("row_type") == _EDGE)
        return TrackGraph(
            node_ids=nodes["node_id"].to_numpy(),
            coordinates=np.stack([nodes[axis].to_numpy() for axis in ("t", "z", "y", "x")], axis=1),
            edges=np.stack([edges["source_id"].to_numpy(), edges["target_id"].to_numpy()], axis=1),
        )

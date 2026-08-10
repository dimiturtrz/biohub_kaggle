"""The consecutive-frame pairs a JOINT detector+edge trainer learns on.

Detection-only training (`tunet_dataset`) needs one frame at a time; training association needs the *real*
pair `(t, t+1)` — the two frames' ground-truth centres for the detection heads, plus the edge matrix between
them for the transformer. Unlike `GapSupervision` (which matches detections to ground truth), these are the
annotated nodes directly, so the edge matrix is read straight off the GT graph's edges: a `1` wherever an
annotated link joins a node at `t` to one at `t + 1`.

Each item reads both frames lazily (quantile-normalised, downsampled — the same recipe as the detector) and
returns them with the two centre sets and the `s x u` edge matrix. Node counts vary per pair, so the matrix
is ragged across items — the trainer processes a pair (or a padded batch), not a stacked tensor.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from core.data.tracks import TrackGraph


@dataclass(frozen=True)
class PairTarget:
    """One training pair: where to read frames `t` and `t + 1`, their GT centres, and the edge matrix between."""

    zarr_path: Path
    timepoint: int  # the source frame; the target frame is timepoint + 1
    q_low: float
    q_high: float
    source_centres: Int[np.ndarray, "s 3"]  # downsampled (z, y', x') at t
    target_centres: Int[np.ndarray, "u 3"]  # downsampled (z, y', x') at t + 1
    edge_matrix: Float[np.ndarray, "s u"]  # 1 where an annotated link joins a source to a target

    @classmethod
    def enumerate(cls, graph: TrackGraph, zarr_path: Path, q_low: float, q_high: float) -> list["PairTarget"]:
        """Every consecutive-frame pair a video's GT graph supports, with centres and the edge matrix precomputed.

        Nodes are grouped by timepoint; a pair exists for each `t` that has annotated nodes at both `t` and
        `t + 1`. The edge matrix is read off the GT edges directly — no matching — so `[i, j] = 1` exactly when
        the annotation links the `i`-th source node to the `j`-th target node. Centres are full-resolution voxels
        (the model divides by the downsample, mirroring the inference scorer's `_gap_logits`).
        """
        timepoints = graph.timepoints()
        positions = graph.positions()
        edge_rows = graph.edge_rows()
        targets: list[PairTarget] = []
        for timepoint in np.unique(timepoints).tolist():
            source_rows = np.flatnonzero(timepoints == timepoint)
            target_rows = np.flatnonzero(timepoints == timepoint + 1)
            if not (len(source_rows) and len(target_rows)):
                continue
            targets.append(
                cls(
                    zarr_path=zarr_path,
                    timepoint=int(timepoint),
                    q_low=q_low,
                    q_high=q_high,
                    source_centres=positions[source_rows].astype(np.int64),
                    target_centres=positions[target_rows].astype(np.int64),
                    edge_matrix=cls._edge_matrix(edge_rows, source_rows, target_rows),
                )
            )
        return targets

    @staticmethod
    def _edge_matrix(
        edge_rows: Int[np.ndarray, "e 2"], source_rows: Int[np.ndarray, "s"], target_rows: Int[np.ndarray, "u"]
    ) -> Float[np.ndarray, "s u"]:
        """The `s x u` matrix marking which source→target GT edges connect this pair's nodes, by row lookup."""
        source_index = {int(row): i for i, row in enumerate(source_rows)}
        target_index = {int(row): j for j, row in enumerate(target_rows)}
        matrix = np.zeros((len(source_rows), len(target_rows)), dtype=np.float32)
        for source, target in edge_rows.tolist():
            i, j = source_index.get(source), target_index.get(target)
            if i is not None and j is not None:
                matrix[i, j] = 1.0
        return matrix


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    """A stream of `steps` GT pairs sampled uniformly, both frames read lazily per access."""

    def __init__(self, targets: list[PairTarget], steps: int, downsample: tuple[int, int, int], seed: int) -> None:
        self._targets = targets
        self._steps = steps
        self._downsample = downsample
        self._seed = seed

    def __len__(self) -> int:
        return self._steps

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample one pair (seeded by index): both normalised frames, both centre sets, and the edge matrix."""
        rng = np.random.default_rng(self._seed + index)
        target = self._targets[int(rng.integers(len(self._targets)))]
        return (
            self._frame(target, target.timepoint),
            self._frame(target, target.timepoint + 1),
            torch.from_numpy(target.source_centres),
            torch.from_numpy(target.target_centres),
            torch.from_numpy(target.edge_matrix),
        )

    def _frame(self, target: PairTarget, timepoint: int) -> Float[Tensor, "z y x"]:
        """One quantile-normalised, downsampled, non-negative frame of the pair."""
        dz, dy, dx = self._downsample
        array = zarr.open_array(target.zarr_path / "0")
        raw = np.asarray(array[timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        normed = (torch.from_numpy(raw) - target.q_low) / (target.q_high - target.q_low + 1e-6)
        return normed.clamp(0.0)

    def stream(self) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Yield every pair once, in index order — reproducible; batching is the trainer's job (ragged matrices)."""
        for index in range(self._steps):
            yield self[index]

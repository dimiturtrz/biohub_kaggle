"""The consecutive-frame pairs a JOINT detector+edge trainer learns on.

Detection-only training (`tunet_dataset`) needs one frame at a time; training association needs the *real*
pair `(t, t+1)` — the two frames' ground-truth centres for the detection heads, plus the edge matrix between
them for the transformer. Unlike `GapSupervision` (which matches detections to ground truth), these are the
annotated nodes directly, so the edge matrix is read straight off the GT graph's edges: a `1` wherever an
annotated link joins a node at `t` to one at `t + 1`.

Each item reads both frames lazily (quantile-normalised, downsampled — the same recipe as the detector) and
returns them with the two centre sets and the `s x u` edge matrix. Node counts vary per pair, so the matrix
is ragged across items — the trainer processes a pair (or a padded batch), not a stacked tensor.

A pair can also carry its PREDECESSOR — frame `t - 1` and that frame's annotated centres — which is what the
prior-velocity feature needs to ask where each source was already heading (see `celltrack.models.prior_velocity`).
It is served only when asked for (`with_previous`), because it costs a third frame read per item; the first
annotated pair of a video has no predecessor at all, and says so with `None` rather than with an empty array.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
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
    # The annotated centres at `t - 1`, or None when this is the video's first annotated frame — the prior
    # gap the sources' velocity is read off. `None` is the explicit "no history" case, not an empty array.
    previous_centres: Int[np.ndarray, "p 3"] | None = None

    @classmethod
    def enumerate(cls, graph: TrackGraph, zarr_path: Path, q_low: float, q_high: float) -> list["PairTarget"]:
        """Every consecutive-frame pair a video's GT graph supports, with centres and the edge matrix precomputed.

        Nodes are grouped by timepoint; a pair exists for each `t` that has annotated nodes at both `t` and
        `t + 1`. The edge matrix is read off the GT edges directly — no matching — so `[i, j] = 1` exactly when
        the annotation links the `i`-th source node to the `j`-th target node. Centres are full-resolution voxels
        (the model divides by the downsample, mirroring the inference scorer's `_gap_logits`).

        `t - 1`'s annotated centres ride along when that frame has any, so a pair knows its own history without
        a second pass over the graph; the video's first annotated frame keeps `None`.
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
            previous_rows = np.flatnonzero(timepoints == timepoint - 1)
            targets.append(
                cls(
                    zarr_path=zarr_path,
                    timepoint=int(timepoint),
                    q_low=q_low,
                    q_high=q_high,
                    source_centres=positions[source_rows].astype(np.int64),
                    target_centres=positions[target_rows].astype(np.int64),
                    edge_matrix=cls._edge_matrix(edge_rows, source_rows, target_rows),
                    previous_centres=positions[previous_rows].astype(np.int64) if len(previous_rows) else None,
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


@dataclass(frozen=True)
class PairSample:
    """One pair as tensors — both frames, both centre sets, the edge matrix, and the predecessor when served.

    Named rather than a positional tuple because the predecessor half is optional: a seven-slot tuple whose
    last two entries are sometimes `None` is exactly the shape a caller unpacks wrong once.
    """

    frame_t: Float[Tensor, "z y x"]
    frame_t1: Float[Tensor, "z y x"]
    source_centres: Int[Tensor, "s 3"]
    target_centres: Int[Tensor, "u 3"]
    edge_matrix: Float[Tensor, "s u"]
    previous_frame: Float[Tensor, "z y x"] | None = None
    previous_centres: Int[Tensor, "p 3"] | None = None

    @property
    def has_previous(self) -> bool:
        """Whether this pair carries the `t - 1` gap — false for a video's first pair, and whenever unasked for."""
        return self.previous_frame is not None and self.previous_centres is not None

    def to(self, device: str) -> "PairSample":
        """The same sample with every tensor it holds moved to `device`; absent predecessor stays absent."""
        moved = {name: value.to(device) for name, value in vars(self).items() if isinstance(value, Tensor)}
        return replace(self, **moved)


class PairDataset(Dataset[PairSample]):
    """A stream of `steps` GT pairs sampled uniformly, both frames read lazily per access.

    Iteration stays uniform-with-replacement; `pair` exists for a caller that has ALREADY chosen which pair it
    wants (a difficulty sampler picks the index, the dataset still owns how a pair becomes tensors).
    """

    def __init__(
        self,
        targets: list[PairTarget],
        steps: int,
        downsample: tuple[int, int, int],
        seed: int,
        *,
        with_previous: bool = False,
    ) -> None:
        self._targets = targets
        self._steps = steps
        self._downsample = downsample
        self._seed = seed
        self._with_previous = with_previous

    def __len__(self) -> int:
        return self._steps

    @property
    def target_count(self) -> int:
        """How many GT pairs the corpus holds — the population a non-uniform sampler draws indices over."""
        return len(self._targets)

    @override
    def __getitem__(self, index: int) -> PairSample:
        """Sample one pair (seeded by index): both normalised frames, both centre sets, and the edge matrix."""
        rng = np.random.default_rng(self._seed + index)
        return self.pair(int(rng.integers(len(self._targets))))

    def pair(self, target_index: int) -> PairSample:
        """One SPECIFIC pair by corpus index: both normalised frames, both centre sets, and the edge matrix."""
        target = self._targets[target_index]
        previous_frame, previous_centres = self._previous(target)
        return PairSample(
            frame_t=self._frame(target, target.timepoint),
            frame_t1=self._frame(target, target.timepoint + 1),
            source_centres=torch.from_numpy(target.source_centres),
            target_centres=torch.from_numpy(target.target_centres),
            edge_matrix=torch.from_numpy(target.edge_matrix),
            previous_frame=previous_frame,
            previous_centres=previous_centres,
        )

    def _previous(self, target: PairTarget) -> tuple[Float[Tensor, "z y x"] | None, Int[Tensor, "p 3"] | None]:
        """Frame `t - 1` and its annotated centres — `(None, None)` when unasked for, or when there is no `t - 1`.

        The extra frame read is a third of this item's decode cost, so it happens only for a run that consumes
        it; a pair whose predecessor carries no annotation is indistinguishable here from one that has no
        predecessor at all, and both degrade to the same zero velocity downstream.
        """
        if not self._with_previous or target.previous_centres is None:
            return None, None
        return self._frame(target, target.timepoint - 1), torch.from_numpy(target.previous_centres)

    def _frame(self, target: PairTarget, timepoint: int) -> Float[Tensor, "z y x"]:
        """One quantile-normalised, downsampled, non-negative frame of the pair."""
        dz, dy, dx = self._downsample
        array = zarr.open_array(target.zarr_path / "0")
        raw = np.asarray(array[timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        normed = (torch.from_numpy(raw) - target.q_low) / (target.q_high - target.q_low + 1e-6)
        return normed.clamp(0.0)

    def stream(self) -> Iterator[PairSample]:
        """Yield every pair once, in index order — reproducible; batching is the trainer's job (ragged matrices)."""
        for index in range(self._steps):
            yield self[index]

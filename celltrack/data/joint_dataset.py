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

WHERE the two frames come from is a strategy (`celltrack.data.frame_source.FrameSource`), not a zarr path
baked into the target: the pair shape the trainer consumes says nothing about storage, so a fully-labelled
corpus stored some other way reaches the same loop unchanged.
"""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import islice
from typing import override

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from celltrack.data.augmentation import _NO_AUGMENTATION, Augmentation, FrameTransform
from celltrack.data.frame_source import FrameSource
from core.data.tracks import TrackGraph


@dataclass(frozen=True)
class PairTarget:
    """One training pair: where to read frames `t` and `t + 1`, their GT centres, and the edge matrix between."""

    frames: FrameSource
    timepoint: int  # the source frame; the target frame is timepoint + 1
    # FULL-RESOLUTION voxels (z, y, x), not downsampled: `JointModel.forward` divides them by the downsample
    # for the feature grid, exactly as the inference scorer's `_gap_logits` does. These comments used to say
    # "downsampled", which is the one mistake a new corpus producer would make silently — the centres would be
    # right by a factor of the downsample and every distance read off them wrong.
    source_centres: Int[np.ndarray, "s 3"]  # at t
    target_centres: Int[np.ndarray, "u 3"]  # at t + 1
    edge_matrix: Float[np.ndarray, "s u"]  # 1 where an annotated link joins a source to a target
    # The annotated centres at `t - 1`, or None when this is the video's first annotated frame — the prior
    # gap the sources' velocity is read off. `None` is the explicit "no history" case, not an empty array.
    previous_centres: Int[np.ndarray, "p 3"] | None = None
    # The GT `t-1 -> t` links (`[p, s] = 1` when previous node p is the annotated parent of source s), or None
    # in lockstep with `previous_centres`. A velocity WARMUP reads the exact incoming displacement off these
    # one-hot columns while the model's own affinity is still garbage from-scratch; a column with no parent
    # sums to zero and degrades to zero velocity, the same "no history" value an unmatched node takes.
    previous_edge_matrix: Float[np.ndarray, "p s"] | None = None

    @classmethod
    def enumerate(cls, graph: TrackGraph, frames: FrameSource) -> list["PairTarget"]:
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
                    frames=frames,
                    timepoint=int(timepoint),
                    source_centres=positions[source_rows].astype(np.int64),
                    target_centres=positions[target_rows].astype(np.int64),
                    edge_matrix=cls._edge_matrix(edge_rows, source_rows, target_rows),
                    previous_centres=positions[previous_rows].astype(np.int64) if len(previous_rows) else None,
                    previous_edge_matrix=(
                        cls._edge_matrix(edge_rows, previous_rows, source_rows) if len(previous_rows) else None
                    ),
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
    previous_edge_matrix: Float[Tensor, "p s"] | None = None

    @property
    def has_previous(self) -> bool:
        """Whether this pair carries the `t - 1` gap — false for a video's first pair, and whenever unasked for."""
        return self.previous_frame is not None and self.previous_centres is not None

    def to(self, device: str) -> "PairSample":
        """The same sample with every tensor it holds moved to `device`; absent predecessor stays absent."""
        moved = {name: value.to(device) for name, value in vars(self).items() if isinstance(value, Tensor)}
        return replace(self, **moved)


@dataclass(frozen=True)
class PairOptions:
    """What a stream wants beyond the base pair: the predecessor frame (velocity) and the augmentation policy.

    Both are training-run toggles orthogonal to the sampling knobs, so they travel together rather than swelling
    the constructor — a non-training reader takes the default (no predecessor, no augmentation = yesterday's pair).
    """

    with_previous: bool = False
    augmentation: Augmentation = _NO_AUGMENTATION


_DEFAULT_OPTIONS = PairOptions()


class PairDataset(Dataset[PairSample]):
    """A stream of `steps` GT pairs sampled uniformly, both frames read lazily per access.

    Iteration stays uniform-with-replacement; `pair` exists for a caller that has ALREADY chosen which pair it
    wants (a difficulty sampler picks the index, the dataset still owns how a pair becomes tensors).
    """

    # Aug draws off an INDEPENDENT stream from the target-selection rng: the same salted seed at line-level, so a
    # window's augmentation is reproducible per step yet uncorrelated with which target each step happened to draw.
    _AUGMENT_SALT = 0x5EED

    def __init__(
        self,
        targets: list[PairTarget],
        steps: int,
        downsample: tuple[int, int, int],
        seed: int,
        options: PairOptions = _DEFAULT_OPTIONS,
    ) -> None:
        self._targets = targets
        self._steps = steps
        self._downsample = downsample
        self._seed = seed
        self._with_previous = options.with_previous
        self._augmentation = options.augmentation

    def __len__(self) -> int:
        return self._steps

    @override
    def __getitem__(self, index: int) -> PairSample:
        """Sample one pair (seeded by index): both normalised frames, both centre sets, and the edge matrix."""
        rng = np.random.default_rng(self._seed + index)
        return self.pair(int(rng.integers(len(self._targets))), augment_step=index)

    def pair(self, target_index: int, *, augment_step: int | None = None) -> PairSample:
        """One SPECIFIC pair by corpus index: both normalised frames, both centre sets, and the edge matrix.

        `augment_step` seeds this pair's shared augmentation; `None` (a diagnostic or sanity read) takes the
        un-augmented frames verbatim, as does an identity policy — so a non-training caller sees yesterday's pair.
        """
        target = self._targets[target_index]
        transform = self._transform(augment_step)
        previous_frame, previous_centres = self._previous(target, transform)
        previous_edges = (
            torch.from_numpy(target.previous_edge_matrix)
            if previous_centres is not None and target.previous_edge_matrix is not None
            else None
        )
        frame_t = target.frames.frame(target.timepoint, self._downsample)
        frame_t1 = target.frames.frame(target.timepoint + 1, self._downsample)
        source_centres = torch.from_numpy(target.source_centres)
        target_centres = torch.from_numpy(target.target_centres)
        return PairSample(
            frame_t=transform.frame(frame_t),
            frame_t1=transform.frame(frame_t1),
            source_centres=transform.coords(source_centres, frame_t.shape),
            target_centres=transform.coords(target_centres, frame_t1.shape),
            edge_matrix=torch.from_numpy(target.edge_matrix),
            previous_frame=previous_frame,
            previous_centres=previous_centres,
            previous_edge_matrix=previous_edges,
        )

    def _transform(self, augment_step: int | None) -> FrameTransform:
        """The ONE transform this pair's frames share — identity when unasked, off, or a genuine no-op draw."""
        if augment_step is None:
            return FrameTransform(gain=1.0, bias=0.0, flips=())
        return self._augmentation.for_pair(np.random.default_rng([self._seed + augment_step, self._AUGMENT_SALT]))

    def _previous(
        self, target: PairTarget, transform: FrameTransform
    ) -> tuple[Float[Tensor, "z y x"] | None, Int[Tensor, "p 3"] | None]:
        """Frame `t - 1` and its annotated centres under the pair's SHARED transform — `(None, None)` when absent.

        The extra frame read is a third of this item's decode cost, so it happens only for a run that consumes
        it; a pair whose predecessor carries no annotation is indistinguishable here from one that has no
        predecessor at all, and both degrade to the same zero velocity downstream. The predecessor takes the
        SAME flips and jitter as `t`/`t+1`, or the velocity gap it feeds would be read off a differently-oriented
        frame than the sources it is compared against.
        """
        if not self._with_previous or target.previous_centres is None:
            return None, None
        frame = target.frames.frame(target.timepoint - 1, self._downsample)
        centres = torch.from_numpy(target.previous_centres)
        return transform.frame(frame), transform.coords(centres, frame.shape)

    def stream(self, threads: int = 1, prefetch: int = 0) -> Iterator[PairSample]:
        """Yield every pair once, in index order — reproducible; batching is the trainer's job (ragged matrices).

        With `threads > 1` the frames are decompressed by a thread pool with `prefetch` pairs in flight, so the
        GPU step overlaps the reads instead of stalling on them. zarr's blosc/zstd decompression releases the
        GIL, so threads parallelise it across cores with no process spawn — the throughput of DataLoader workers
        without the Windows worker-pool deadlock. This is the same device `TunetFrameDataset.stream` already
        uses; the joint loop was reading its two frames INLINE, at a measured ~16 ms each against a step of
        160-440 ms, so it idled the GPU for roughly a tenth to a fifth of every step.

        Items are yielded in INDEX ORDER even though they finish decompressing out of order, so a run stays
        reproducible. `threads = 1` is the serial path, byte-identical to before, and it is what the sampler
        path uses: a curriculum chooses the next index from feedback on the previous step, so reading ahead
        would draw against a staler distribution than the run's own semantics promise.
        """
        if threads <= 1:
            for index in range(self._steps):
                yield self[index]
            return
        with ThreadPoolExecutor(max_workers=threads) as pool:
            indices = iter(range(self._steps))
            pending: deque[Future[PairSample]] = deque()
            for index in islice(indices, max(prefetch, 1)):
                pending.append(pool.submit(self.__getitem__, index))
            for index in indices:
                item = pending.popleft().result()
                pending.append(pool.submit(self.__getitem__, index))
                yield item
            while pending:
                yield pending.popleft().result()

"""The fully-labelled pair corpus — the one thing our own annotation can never be.

Our association failure is measured and specific: on the dense movie every one of the mislinked partners is an
UNANNOTATED detection. `PairTarget.enumerate` reads its `s x u` edge matrix off the ground-truth graph, so a
pair's candidate set IS the annotated node set, and with ~1-2% of the dense movie's cells annotated the
detections that actually beat us can never enter a batch at all. A hard-negative term confirmed the same thing
from the other side: mining decoys among annotated nodes put the gradient on cases already won by ~4 logits.

This corpus fixes exactly that, and NOT by being more data. Every cell of a synthetic frame is labelled, so
the candidate set is COMPLETE BY CONSTRUCTION — the crowded near-neighbours are IN the matrix, as true
negatives. That is the single property being borrowed here; the images are not claimed to be better, and a
prior probe that fine-tuned the edge head on this corpus with the backbone FROZEN and no masked detection
loss measured monotonic dose-damage (0.9273 -> 0.9105 synth-only -> 0.9058 curriculum). A run using this
module differs on all four counts and claims something narrower.

TWO PROPERTIES OF THE CORPUS THAT MUST NOT BE FORGOTTEN:

- THE COORDINATE TRAP. `volumes` are POOLED (`vol[:, ::4, ::4]`, the official evaluator's own pooling) to
  64^3, while `nodes` stay in the RAW `(64, 256, 256)` grid. The trainer hands the model FULL-resolution
  centres and divides by the run's downsample, so the raw coordinates are exactly right AS THEY ARE — but the
  volume is pre-pooled and must NOT be strided again. `SyntheticFrames.frame` therefore refuses a downsample
  that is not the pooling already applied, rather than silently returning a 16x16 slab: a prior probe that
  missed this measured a spurious recall of 0.025.
- DIVISIONS ARE ~13x OVER-SAMPLED (47 per 1352 nodes = 3.5%, against a real rate near 0.26%, per the corpus's
  own metadata note). Nothing learned about division RATE from this corpus is calibrated.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from celltrack.data.joint_dataset import PairTarget
from core.data.tracks import TrackGraph

# The pooling the corpus ships already applied — `vol[:, ::4, ::4]`, identical to the official evaluator's.
# A run at any other downsample is reading a different grid than the node coordinates live on, so it is
# rejected rather than resampled: the mismatch has already cost one wrong measurement.
POOLED_BY: tuple[int, int, int] = (1, 4, 4)

# The same intensity window the competition videos are normalised through, so a synthetic frame reaches the
# backbone on the scale the real ones do. Per SEQUENCE, as the real recipe is per video.
_Q_LOW, _Q_HIGH = 0.001, 0.999

# How many decoded sequences stay resident. Each is 6 x 64^3 uint16 (3 MB), so 64 of them is ~200 MB of host
# RAM and spares the loop a re-decode per frame — a pair reads two frames of the same sequence by construction.
_RESIDENT_SEQUENCES = 64


@dataclass(frozen=True)
class _Sequence:
    """One decoded sequence: its pooled volumes and the intensity window they are normalised through."""

    volumes: np.ndarray
    q_low: float
    q_high: float


@dataclass(frozen=True)
class SyntheticFrames:
    """One synthetic sequence's frames, normalised the way a competition video's are — a `FrameSource`."""

    sequence_path: Path

    @staticmethod
    @lru_cache(maxsize=_RESIDENT_SEQUENCES)
    def _decoded(path: Path) -> _Sequence:
        """The sequence's pooled volumes plus its own quantile window, decoded once and kept resident."""
        volumes = np.asarray(np.load(path)["volumes"], dtype=np.float32)
        low, high = np.quantile(volumes, [_Q_LOW, _Q_HIGH])
        return _Sequence(volumes=volumes, q_low=float(low), q_high=float(high))

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """That timepoint's pooled, quantile-normalised, non-negative volume — never strided a second time."""
        if tuple(downsample) != POOLED_BY:
            raise ValueError(
                f"synthetic volumes ship pooled by {POOLED_BY} while their nodes stay in the raw grid; "
                f"a run at downsample {tuple(downsample)} would read a grid the coordinates do not live on"
            )
        sequence = SyntheticFrames._decoded(self.sequence_path)
        normed = (torch.from_numpy(sequence.volumes[timepoint]) - sequence.q_low) / (
            sequence.q_high - sequence.q_low + 1e-6
        )
        return normed.clamp(0.0)


class SyntheticPairs:
    """The synthetic corpus as the trainer's own `PairTarget`s — same shape, complete candidate sets."""

    @staticmethod
    def sequences(root: Path, count: int | None = None, seed: int = 0) -> list[Path]:
        """`count` sequence files drawn without replacement (all of them when `count` is None), seeded."""
        available = sorted(root.glob("*.npz"))
        if count is None or count >= len(available):
            return available
        chosen = np.random.default_rng(seed).choice(len(available), size=count, replace=False)
        return [available[index] for index in sorted(chosen.tolist())]

    @staticmethod
    def graph(sequence_path: Path) -> TrackGraph:
        """One sequence's lineage as a `TrackGraph` — nodes in the RAW grid, edges already row indices.

        The corpus stores `nodes` as `(t, z, y, x, id)` floats and `edges` as pairs of ROW indices into that
        array (verified: every stored edge spans exactly one frame under the row reading). Row indices are
        what `TrackGraph.edge_rows` yields when the ids are `arange`, so naming the rows as the ids is what
        makes the shared enumeration read this corpus with no special case at all.
        """
        stored = np.load(sequence_path)
        nodes, edges = stored["nodes"], stored["edges"]
        return TrackGraph(
            node_ids=np.arange(len(nodes), dtype=np.int64),
            coordinates=np.rint(nodes[:, :4]).astype(np.int64),
            edges=edges.astype(np.int64),
        )

    @classmethod
    def enumerate(cls, sequences: list[Path]) -> list[PairTarget]:
        """Every consecutive-frame pair of every sequence, as the pairs the joint trainer already consumes."""
        targets: list[PairTarget] = []
        for sequence_path in sequences:
            targets.extend(PairTarget.enumerate(cls.graph(sequence_path), SyntheticFrames(sequence_path)))
        return targets

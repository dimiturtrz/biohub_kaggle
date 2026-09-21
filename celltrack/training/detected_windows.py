"""Training windows built from DETECTED candidates rather than ground-truth nodes.

The champion's edge head only ever sees GT nodes, so it never learns the one thing inference asks of it:
that a real source must REJECT the ~1000 false-positive candidates per frame that crowd it. These windows
put that crowd in front of the head at training time — every detection is a column, and the ones that match
no GT node are supervised to link to nothing.

Selection is K-capped because edge attention is O(K^2) in both VRAM and time.

`CandidateSelection` caps by keeping every GT-matched detection and filling the remainder with the
highest-scoring unmatched ones, on the argument that edge attention is permutation-equivariant and
set-size-agnostic, so an exclusion prior trained on the top-K hardest negatives carries to the full crowd.
MEASURED, and it does not: keeping all positives but only the top tenth of the negatives leaves ~4.6
distractors per real cell in training against ~58 at inference on a dense movie. A head trained that way
reads as near-perfect on its own windows (acc 0.9990, recall 0.9967) and then rejects almost everything
at inference -- edge_jaccard 0.06 on the densest movie, with small edge-FP and huge edge-FN. Capping the
candidates at inference instead does not rescue it, because the detector's score does not rank real cells
above the crowd: the top 96 of a 962-candidate frame hold only 26% of the GT cells.

`RatioMatchedSelection` caps the POSITIVES rather than the negatives, so the crowd a window shows is the
crowd inference shows. What it gives up is supervision density per window, not the ratio.

The external trainer's window type and positional-feature function are passed in rather than imported: this
module owns the construction, not the donor's schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import override

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.spatial import KDTree

GtEdges = set[tuple[int, int]]


@dataclass(frozen=True)
class CandidateFrame:
    """One frame's kept detections, in the downsampled grid the UNet features are indexed in."""

    coords: Float[np.ndarray, "n 3"]
    matched_gt: Int[np.ndarray, " n"]

    def __len__(self) -> int:
        return len(self.coords)


@dataclass(frozen=True)
class CandidateSelection:
    """Which of a frame's detections a window carries, and which GT node each one stands for."""

    keep: int
    gate_um: float

    def frame(
        self,
        coords: Float[np.ndarray, "n 3"],
        score: Float[np.ndarray, " n"],
        detections_um: Float[np.ndarray, "n 3"],
        gt_um: Float[np.ndarray, "m 3"],
        gt_ids: Int[np.ndarray, " m"],
    ) -> CandidateFrame:
        matched = self.match_to_ground_truth(detections_um, gt_um, gt_ids)
        kept = self.select(score, matched)
        return CandidateFrame(coords=coords[kept], matched_gt=matched[kept])

    def match_to_ground_truth(
        self,
        detections_um: Float[np.ndarray, "n 3"],
        gt_um: Float[np.ndarray, "m 3"],
        gt_ids: Int[np.ndarray, " m"],
    ) -> Int[np.ndarray, " n"]:
        """GT node id per detection, or -1 beyond the gate. Physical µm, so a grid change cannot shift it."""
        distance, nearest = KDTree(gt_um).query(detections_um, k=1)
        return np.where(distance <= self.gate_um, gt_ids[nearest], -1).astype(np.int64)

    def select(
        self,
        score: Float[np.ndarray, " n"],
        matched_gt: Int[np.ndarray, " n"],
    ) -> Int[np.ndarray, " k"]:
        """Indices kept: every GT-matched detection, then the hardest negatives.

        A matched detection is never dropped for a higher-scoring unmatched one — the targets are defined
        over matched pairs, so dropping one silently deletes a positive. Only when the matched set alone
        overflows the cap (rare, and a sign the cap is too low for the density) does score decide among them.
        """
        matched = np.nonzero(matched_gt >= 0)[0]
        if len(matched) >= self.keep:
            return np.sort(matched[np.argsort(-score[matched])][: self.keep])
        unmatched = np.nonzero(matched_gt < 0)[0]
        fill = self.keep - len(matched)
        if len(unmatched) > fill:
            unmatched = unmatched[np.argsort(-score[unmatched])[:fill]]
        return np.sort(np.concatenate([matched, unmatched]))


@dataclass(frozen=True)
class RatioMatchedSelection(CandidateSelection):
    """A K-capped selection that keeps the crowd-to-cell ratio inference will actually present.

    The cap is spent on the negatives in `CandidateSelection`, which is what distorts the ratio. Here it is
    spent on the positives instead: a fixed handful of GT cells is carried per window and the rest of the
    cap goes to the crowd, so a window at K=96 carrying 2 cells shows ~47 distractors each -- the dense
    regime -- rather than ~4.6.

    Which cells are carried is chosen once per video, not once per frame. A cell dropped from one frame but
    kept in the next would leave a source whose true target is absent from the window, and the loss would
    teach that source to link to nothing; holding the choice fixed across the window makes a dropped cell
    simply a cell outside the field of view, which is the situation the head already handles. Detections
    matching a dropped cell are dropped too rather than demoted -- they are real cells, and supervising
    them as negatives would teach the head to reject exactly what it is meant to find.
    """

    carried_gt_ids: frozenset[int] = frozenset()

    @override
    def select(
        self,
        score: Float[np.ndarray, " n"],
        matched_gt: Int[np.ndarray, " n"],
    ) -> Int[np.ndarray, " k"]:
        carried = np.array([gt in self.carried_gt_ids for gt in matched_gt], dtype=bool)
        positives = np.nonzero(carried)[0]
        dropped_cells = (matched_gt >= 0) & ~carried
        negatives = np.nonzero(~dropped_cells & ~carried)[0]
        fill = max(self.keep - len(positives), 0)
        if len(negatives) > fill:
            negatives = negatives[np.argsort(-score[negatives])[:fill]]
        return np.sort(np.concatenate([positives, negatives]))

    def carrying(
        self,
        gt_ids: Int[np.ndarray, " m"],
        gt_edges: GtEdges,
        tracks_carried: int,
        seed: int,
    ) -> RatioMatchedSelection:
        """Sample whole TRACKS this video's windows will carry. Seeded so a rebuild is the same dataset.

        Sampling nodes would not survive a window: a node is one cell at one frame, so a node-level sample
        carries a cell into frame t and drops it from t+1, which is the incoherence this strategy exists to
        avoid. The GT graph stores no track id, so a track is taken as what it is -- a connected component
        of the GT edges, divisions included, which keeps a mother and her daughters together.
        """
        component_of = self.track_components(gt_ids, gt_edges)
        components = np.unique(list(component_of.values()))
        generator = np.random.default_rng(seed)
        chosen = set(generator.choice(components, size=min(tracks_carried, len(components)), replace=False).tolist())
        carried = {node for node, component in component_of.items() if component in chosen}
        return replace(self, carried_gt_ids=frozenset(carried))

    def track_components(self, gt_ids: Int[np.ndarray, " m"], gt_edges: GtEdges) -> dict[int, int]:
        """Component label per GT node id, by union-find over the GT edges."""
        parent = {int(node): int(node) for node in gt_ids}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for source, target in gt_edges:
            if source in parent and target in parent:
                root_source, root_target = find(int(source)), find(int(target))
                if root_source != root_target:
                    parent[root_target] = root_source
        return {node: find(node) for node in parent}


class SparseEdgeTargets(Sequence[torch.Tensor]):
    """A window's edge target is a dense (K,K) float32 that is ~99.9% zeros.

    Holding one per window for every video is what drove the trainer's resident set past 40GB and got it
    reaped mid-run. The true pairs are kept instead and densified on access — the loss only ever reads one
    window's targets at a time, so nothing downstream can tell the difference.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[Int[np.ndarray, " e"], Int[np.ndarray, " e"]]],
        shapes: Sequence[tuple[int, int]],
    ) -> None:
        self._pairs = list(pairs)
        self._shapes = list(shapes)

    @override
    def __len__(self) -> int:
        return len(self._pairs)

    @override
    def __getitem__(self, index: int) -> Float[torch.Tensor, "rows cols"]:  # type: ignore[override]
        rows, cols = self._pairs[index]
        target = torch.zeros(self._shapes[index], dtype=torch.float32)
        if len(rows):
            target[rows, cols] = 1.0
        return target

    @override
    def __iter__(self) -> Iterator[Float[torch.Tensor, "rows cols"]]:
        return (self[i] for i in range(len(self)))

    @staticmethod
    def pairs(
        source_gt: Int[np.ndarray, " r"],
        target_gt: Int[np.ndarray, " c"],
        gt_edges: GtEdges,
    ) -> tuple[Int[np.ndarray, " e"], Int[np.ndarray, " e"]]:
        """Row/column indices of the true links between two frames' kept detections.

        Indexed by GT id rather than scanned over `gt_edges`, so cost follows the kept detections instead
        of the whole video's edge list.
        """
        rows_by_id = SparseEdgeTargets._by_gt_id(source_gt)
        cols_by_id = SparseEdgeTargets._by_gt_id(target_gt)

        rows: list[int] = []
        cols: list[int] = []
        for source_id, rows_here in rows_by_id.items():
            for target_id, cols_here in cols_by_id.items():
                if (source_id, target_id) not in gt_edges:
                    continue
                for row in rows_here:
                    for col in cols_here:
                        rows.append(row)
                        cols.append(col)
        return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)

    @staticmethod
    def _by_gt_id(matched_gt: Int[np.ndarray, " n"]) -> dict[int, list[int]]:
        by_id: dict[int, list[int]] = {}
        for index, gt in enumerate(matched_gt):
            if gt >= 0:
                by_id.setdefault(int(gt), []).append(index)
        return by_id


@dataclass(frozen=True)
class DetectedWindows[W]:
    """Assembles one video's windows from its per-frame candidates.

    Generic in the window type so the caller keeps its own -- the trainer's `FrameWindowData` here, a
    plain record in a test -- without this module importing the donor's schema.

    A window carries its frames' `[t, z, y, x]` rows, NOT their positional embedding. The embedding is
    ~64x the coordinates it is computed from (four axes x 32 sinusoidal dims against three float32
    coordinates), so holding one per window for every video is what puts the trainer's resident set in
    tens of GB. The window type decides when to embed; the cheapest moment is inside `__getitem__`,
    where it costs a fraction of a millisecond against a 15-second training step.
    """

    window_size: int
    window_factory: Callable[..., W]

    def build(self, frames: Sequence[CandidateFrame | None], gt_edges: GtEdges) -> list[W]:
        """A window whose frames are not all populated is skipped, not padded."""
        windows: list[W] = []
        for start in range(len(frames) - self.window_size + 1):
            span = frames[start : start + self.window_size]
            if any(frame is None or len(frame) == 0 for frame in span):
                continue
            populated = [frame for frame in span if frame is not None]
            windows.append(self._window(start, populated, gt_edges))
        return windows

    def _window(self, start: int, populated: list[CandidateFrame], gt_edges: GtEdges) -> W:
        stamped = []
        coords = []
        for offset, frame in enumerate(populated):
            timepoint = np.full(len(frame), start + offset, dtype=np.float32)
            stamped.append(np.column_stack([timepoint, frame.coords]).astype(np.float32))
            coords.append(torch.from_numpy(frame.coords.astype(np.float32)))

        pairs = []
        shapes = []
        for i in range(self.window_size - 1):
            source, target = populated[i], populated[i + 1]
            pairs.append(SparseEdgeTargets.pairs(source.matched_gt, target.matched_gt, gt_edges))
            shapes.append((len(source), len(target)))

        return self.window_factory(
            t_start=start,
            n_frames=self.window_size,
            stamped=stamped,
            coords=coords,
            node_counts=[len(frame) for frame in populated],
            targets=SparseEdgeTargets(pairs, shapes),
        )

"""Training pairs whose candidate set is the DETECTED neighbourhood — the population our failures live in.

Every association arm this campaign ran was trained on `PairTarget.enumerate`, whose rows and columns are
ANNOTATED nodes. Annotation covers ~1.8% of the cells in a frame, so the corpus carries 0.99 in-gate candidates
per source and only 1.9% of sources face a rival at all, while the tracker chooses among 3.80 candidates with
95.4% of sources contested. The head is therefore trained on uncontested choices and deployed on contested
ones, and the wrong partner in all 38 dense mislinks is an unannotated detection that no batch ever contained.

This corpus is the same pairs read off the DETECTOR's output instead. It is the only substrate that is both

- COMPLETE: every real rival is present as a column, so a crowding neighbour is a true NEGATIVE in the edge
  matrix rather than an unlabelled node outside it, which annotated pairs can never express; and
- HARD: those rivals are the ones the features cannot separate — that is what makes them our failures.

Synthetic pairs (`celltrack.data.synthetic_pairs`) buy the first property and not the second: their edge loss
sits at 0.004 against 0.20-0.53 on real pairs, i.e. the warm-started head already solves them, and mixing them
in measured as pure dilution.

THE SUPERVISION RULE, which is the whole design. A source row is supervised only when the answer is KNOWN:
the detection matches an annotated cell, that cell has an annotated successor, and the successor was itself
detected. Then the successor's column is 1 and every other column is 0 — including the unannotated detections,
which is the point. A row failing any of those is UNDECIDABLE and is simply left out; rows are sources, so
dropping one removes its supervision while leaving every target in place as a candidate for the rows that
remain. That is why no mask field is needed and `PairTarget` is reused unchanged.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int

from celltrack.data.frame_source import FrameSource
from celltrack.data.joint_dataset import PairTarget
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher


class DetectionPairs:
    """Pairs read off the detector's own output, supervised only where the annotation can adjudicate."""

    @staticmethod
    def of(
        detections: TrackGraph, truth: TrackGraph, frames: FrameSource, spacing: Spacing, gate_um: float
    ) -> list[PairTarget]:
        """Every consecutive-frame pair the detections support, with the decidable source rows supervised.

        Matching is the metric's own `DistanceMatcher`, so "this detection is that annotated cell" means here
        exactly what it means when the leaderboard scores us — one matching rule, not a training-only variant.

        Candidates are GATED to `gate_um` of some supervised source. NOT because inference gates — the edge head
        scores the full source x target matrix there and the linker gates afterwards, when it builds the cost —
        but because an out-of-gate column can never change the score, so its gradient teaches the head to reject
        cells it will never be asked about. Those are also the trivial negatives: every one of the 38 dense
        mislinks is in-gate by construction, since the linker chose it. Ungated, a frame's ~743 detections make
        an edge matrix ~60x the annotated one, which exhausts 32GB of VRAM and spills into system memory.
        """
        gt_rows = DistanceMatcher(spacing=spacing).match(detections, truth).gt_rows
        successor = DetectionPairs._successor_of(truth)
        detection_of = DetectionPairs._detection_of(gt_rows)
        timepoints = detections.timepoints()
        positions = detections.positions()
        pairs: list[PairTarget] = []
        for timepoint in np.unique(timepoints).tolist():
            target_rows = np.flatnonzero(timepoints == timepoint + 1)
            if not len(target_rows):
                continue
            sources, columns = DetectionPairs._supervised(
                np.flatnonzero(timepoints == timepoint), target_rows, gt_rows, successor, detection_of
            )
            if not len(sources):
                continue
            target_rows, columns = DetectionPairs._gated(
                sources, target_rows, columns, spacing.to_micrometres(positions), gate_um
            )
            previous_rows = np.flatnonzero(timepoints == timepoint - 1)
            pairs.append(
                PairTarget(
                    frames=frames,
                    timepoint=int(timepoint),
                    source_centres=positions[sources].astype(np.int64),
                    target_centres=positions[target_rows].astype(np.int64),
                    edge_matrix=DetectionPairs._one_hot(columns, len(target_rows)),
                    previous_centres=positions[previous_rows].astype(np.int64) if len(previous_rows) else None,
                )
            )
        return pairs

    @staticmethod
    def _gated(
        sources: Int[np.ndarray, "k"],
        target_rows: Int[np.ndarray, "u"],
        columns: Int[np.ndarray, "k"],
        positions_um: Float[np.ndarray, "n 3"],
        gate_um: float,
    ) -> tuple[Int[np.ndarray, "g"], Int[np.ndarray, "k"]]:
        """The targets within `gate_um` of some supervised source, and the true columns re-indexed onto them.

        The true successor is kept whatever its distance: a supervised row must keep its positive, and a true
        step longer than the gate is exactly the case the linker gets wrong — dropping it would delete the
        hardest positives from the corpus while keeping their rivals.
        """
        distances = np.linalg.norm(positions_um[target_rows][None, :, :] - positions_um[sources][:, None, :], axis=2)
        keep = (distances <= gate_um).any(axis=0)
        keep[columns] = True
        kept = np.flatnonzero(keep)
        reindex = {int(column): index for index, column in enumerate(kept.tolist())}
        return target_rows[kept], np.array([reindex[int(column)] for column in columns], dtype=np.int64)

    @staticmethod
    def _successor_of(truth: TrackGraph) -> dict[int, int]:
        """Each annotated node's row -> its annotated successor's row, for the ground-truth graph.

        A dividing parent has two successors and would collide here. It is dropped rather than resolved: this
        corpus trains the ONE-TO-ONE association the linker's cost prices, and a fork's two children are a
        division-recovery question (`celltrack.postproc.affinity_division_recovery`), not an ambiguity to
        silently pick a winner from.
        """
        counts: dict[int, int] = {}
        successor: dict[int, int] = {}
        for source, target in truth.edge_rows().tolist():
            counts[source] = counts.get(source, 0) + 1
            successor[source] = target
        return {source: target for source, target in successor.items() if counts[source] == 1}

    @staticmethod
    def _detection_of(gt_rows: Int[np.ndarray, "n"]) -> dict[int, int]:
        """The inverse of the matching: each matched ground-truth row -> the detection row that claimed it."""
        return {int(gt_row): row for row, gt_row in enumerate(gt_rows.tolist()) if gt_row != UNMATCHED}

    @staticmethod
    def _supervised(
        source_rows: Int[np.ndarray, "s"],
        target_rows: Int[np.ndarray, "u"],
        gt_rows: Int[np.ndarray, "n"],
        successor: dict[int, int],
        detection_of: dict[int, int],
    ) -> tuple[Int[np.ndarray, "k"], Int[np.ndarray, "k"]]:
        """The decidable source rows and, for each, the column of the detection its true successor landed on.

        Decidable means all three links in the chain hold: detection -> annotated cell -> annotated successor
        -> a detection of that successor, and that detection is in THIS gap's target frame. The last clause is
        not redundant — a successor detected in a different frame than `t + 1` would index a column that does
        not exist here.
        """
        column_of = {int(row): column for column, row in enumerate(target_rows.tolist())}
        sources: list[int] = []
        columns: list[int] = []
        for row in source_rows.tolist():
            gt_row = int(gt_rows[row])
            target_gt = successor.get(gt_row, UNMATCHED) if gt_row != UNMATCHED else UNMATCHED
            column = column_of.get(detection_of.get(target_gt, UNMATCHED), UNMATCHED)
            if column != UNMATCHED:
                sources.append(row)
                columns.append(column)
        return np.array(sources, dtype=np.int64), np.array(columns, dtype=np.int64)

    @staticmethod
    def _one_hot(columns: Int[np.ndarray, "k"], width: int) -> Float[np.ndarray, "k u"]:
        """One row per supervised source, a single 1 at its true successor and 0 at every other detection.

        The zeros are the whole point of this corpus: on annotated pairs the crowding neighbours are absent
        from the matrix, so the loss never learns to reject them; here they are explicit negatives.
        """
        matrix = np.zeros((len(columns), width), dtype=np.float32)
        matrix[np.arange(len(columns)), columns] = 1.0
        return matrix

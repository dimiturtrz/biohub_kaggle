"""Classifying every ground-truth edge by *why* the tracker did or didn't reproduce it — the dense-loss decomposition.

The dense movie carries almost all of the residual edge-Jaccard loss, and that loss is not one failure but
several with different fixes. This module splits a movie's annotated edges into the fates a linker can land
them in, so the decision of *which* lever to build next reads off measured fractions rather than a guess:

- ``ENDPOINT_MISSING`` — an endpoint has no matched detection. Detection-limited; no linker recovers it.
- ``SKIP`` — both endpoints detected, but the source links to nothing. A dropped/too-costly edge; a lower
  link cost or a gap bridge recovers it.
- ``MISLINK_CONFLICT`` — the source links elsewhere *and* the true target was claimed by another source.
  The two assignments conflict, so a global-over-time optimiser that re-balances the whole frame can free it —
  this is the fraction a min-cost-flow linker (bead tf5) is aimed at.
- ``MISLINK_FREE`` — the source links elsewhere while the true target sits unclaimed. The learned P plus
  distance genuinely rank a wrong-but-real neighbour above the truth; reassignment can't fix what the signal
  gets wrong, only a better-trained affinity (bead zni) can.
- ``CORRECT`` — reproduced.

The classification is one pass of array algebra over the matched graphs, so it is cheap to re-run as the
pipeline changes (it re-measures itself after the flow linker lands, not just before).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

from celltrack.affinity import EdgeAffinity
from celltrack.eval.proxy import TestMovieProxy
from celltrack.tracker import CellTracker
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The dense movie that holds the residual — its annotations carry the four test divisions and dominate the
# pooled loss (raw Jaccard ~0.89 against ~0.96-1.0 on the other three), so its fate split is the one that matters.
DENSE_MOVIE = "6bba_05db0fb1"


class Fate(IntEnum):
    """Why one ground-truth edge was or was not reproduced — the mutually-exclusive outcomes of a link attempt."""

    CORRECT = 0
    SKIP = 1
    MISLINK_CONFLICT = 2
    MISLINK_FREE = 3
    ENDPOINT_MISSING = 4


@dataclass(frozen=True)
class DenseFateDiagnosis:
    """Each annotated edge tagged with the `Fate` the tracker landed it in — the dense loss decomposed."""

    fates: Int[np.ndarray, "e"]

    @staticmethod
    def _invert(matching: NodeMatching, truth_count: int) -> Int[np.ndarray, "g"]:
        """For each ground-truth node, the predicted row matched to it, or `UNMATCHED` — the matching reversed.

        The per-timepoint assignment is one-to-one, so no ground-truth node is claimed by two detections; the
        inverse is well defined and lets a ground-truth edge be looked up as the detections standing in for it.
        """
        predicted_of_truth = np.full(truth_count, UNMATCHED, dtype=np.int64)
        matched = np.flatnonzero(matching.is_matched())
        predicted_of_truth[matching.gt_rows[matched]] = matched
        return predicted_of_truth

    @classmethod
    def of(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching) -> "DenseFateDiagnosis":
        """Tag every ground-truth edge by comparing where its endpoints' matched detections actually linked.

        `matching` pairs predicted nodes to ground-truth nodes; inverting it (the per-timepoint assignment is
        one-to-one, so no ground-truth node is claimed twice) gives, for each ground-truth node, the predicted
        detection standing in for it — and the tracker's own edges then say whether that detection linked to the
        truth's target, to a wrong neighbour, or to nothing.
        """
        predicted_of_truth = cls._invert(matching, len(truth.node_ids))
        truth_edges = truth.edge_rows()
        source = predicted_of_truth[truth_edges[:, 0]]
        target = predicted_of_truth[truth_edges[:, 1]]
        both_detected = (source != UNMATCHED) & (target != UNMATCHED)

        predicted_edges = prediction.edge_rows()
        stride = max(len(prediction.node_ids), 1)
        predicted_codes = predicted_edges[:, 0] * stride + predicted_edges[:, 1]
        wanted = np.where(both_detected, source * stride + target, UNMATCHED)
        reproduced = both_detected & np.isin(wanted, predicted_codes)

        out_degree = np.bincount(predicted_edges[:, 0], minlength=stride)
        in_degree = np.bincount(predicted_edges[:, 1], minlength=stride)
        source_linked = both_detected & (out_degree[np.where(source == UNMATCHED, 0, source)] > 0)
        target_claimed = both_detected & (in_degree[np.where(target == UNMATCHED, 0, target)] > 0)

        return cls(fates=cls._label(both_detected, reproduced, source_linked, target_claimed))

    def counts(self) -> dict[Fate, int]:
        """How many annotated edges fell into each fate — the decomposition as absolute edge counts."""
        tally = np.bincount(self.fates, minlength=len(Fate))
        return {fate: int(tally[fate]) for fate in Fate}

    @staticmethod
    def _label(
        both: Bool[np.ndarray, "e"],
        reproduced: Bool[np.ndarray, "e"],
        source_linked: Bool[np.ndarray, "e"],
        target_claimed: Bool[np.ndarray, "e"],
    ) -> Int[np.ndarray, "e"]:
        """Fold the per-edge boolean masks into one exclusive `Fate` code, most-specific outcome first."""
        fates = np.full(len(both), Fate.ENDPOINT_MISSING, dtype=np.int64)
        mislink = both & ~reproduced & source_linked
        fates[both & ~reproduced & ~source_linked] = Fate.SKIP
        fates[mislink & ~target_claimed] = Fate.MISLINK_FREE
        fates[mislink & target_claimed] = Fate.MISLINK_CONFLICT
        fates[reproduced] = Fate.CORRECT
        return fates


@dataclass(frozen=True)
class AffinityIndex:
    """Where each node sits in an affinity's per-gap matrices — looked up by node ID, never by graph row.

    A matrix's rows follow `np.flatnonzero(timepoints == t)` over the graph the affinity was SCORED on, which
    is the detector's node set. Post-processing prunes short tracks and inserts synthetic bridge nodes, so the
    finished track graph the mislinks live in is a different row space, and indexing it into the matrices would
    silently read a neighbour's probability. Node IDs survive post-processing, so the lookup is by ID; a node
    the affinity never scored (a bridge's invented midpoint) is reported absent rather than guessed at.
    """

    affinity: EdgeAffinity
    slots: dict[int, tuple[int, int]]

    @classmethod
    def of(cls, scored: TrackGraph, affinity: EdgeAffinity) -> "AffinityIndex":
        """Index the graph the affinity was scored over — every node ID to its `(timepoint, per-gap row)` slot."""
        timepoints = scored.timepoints()
        slots: dict[int, tuple[int, int]] = {}
        for timepoint in np.unique(timepoints).tolist():
            rows = np.flatnonzero(timepoints == timepoint)
            for column, row in enumerate(rows.tolist()):
                slots[int(scored.node_ids[row])] = (int(timepoint), column)
        return cls(affinity=affinity, slots=slots)

    def pair(self, source_id: int, true_id: int, chosen_id: int) -> tuple[float, float] | None:
        """One source's P(true successor) and P(chosen neighbour), or `None` where this gap scored no such pair."""
        source = self.slots.get(source_id)
        if source is None:
            return None
        timepoint, row = source
        matrix = self.affinity.probabilities(timepoint)
        true_column = self._column(true_id, timepoint)
        chosen_column = self._column(chosen_id, timepoint)
        if matrix is None or true_column is None or chosen_column is None:
            return None
        return float(matrix[row, true_column]), float(matrix[row, chosen_column])

    def _column(self, node_id: int, timepoint: int) -> int | None:
        """The node's column in the `t -> t+1` matrix, or `None` when it is not a scored target of that gap."""
        slot = self.slots.get(node_id)
        return slot[1] if slot is not None and slot[0] == timepoint + 1 else None


@dataclass(frozen=True)
class MislinkSignal:
    """For each mislinked edge, the affinity's probability of the true successor vs the wrong one it chose.

    A mislink is either a *signal* error — the edge transformer itself scores the wrong neighbour above the
    true successor (`p_chosen > p_true`), which only a better-trained affinity (bead zni) can fix — or a
    *cost* error — the affinity ranks the true successor higher, but the nearer wrong neighbour won on the
    distance term of `distance - bonus*P` anyway (`p_true >= p_chosen`), which is the bonus/gate cost lever
    already swept and refuted (e9b). The inverted fraction is the share a retrain could even target.

    A mislink whose endpoints the affinity never scored — a bridge's invented midpoint, or a chosen target more
    than one gap ahead — carries no probability pair and is left OUT of the arrays rather than imputed.
    """

    p_true: Float[np.ndarray, "m"]
    p_chosen: Float[np.ndarray, "m"]

    @classmethod
    def of(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, index: AffinityIndex
    ) -> "MislinkSignal":
        """Read the affinity's true- and chosen-target probability for every mislinked annotated edge."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        predicted_of_truth = DenseFateDiagnosis._invert(matching, len(truth.node_ids))  # noqa: SLF001
        successors = Adjacency.of(prediction).successors
        node_ids = prediction.node_ids
        mislinked = np.flatnonzero((fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE))
        pairs: list[tuple[float, float]] = []
        for source, target in truth.edge_rows()[mislinked]:
            source_row, true_row = int(predicted_of_truth[source]), int(predicted_of_truth[target])
            chosen_row = successors[source_row][0]
            scored = index.pair(int(node_ids[source_row]), int(node_ids[true_row]), int(node_ids[chosen_row]))
            if scored is not None:
                pairs.append(scored)
        return cls(
            p_true=np.array([true for true, _ in pairs], dtype=np.float64),
            p_chosen=np.array([chosen for _, chosen in pairs], dtype=np.float64),
        )

    def inverted_fraction(self) -> float:
        """The share of mislinks the affinity itself gets wrong (chosen scored above true) — the retrain target.

        Zero, not NaN, on a graph with no mislinks: the quantity is how much of the residual a better-trained
        affinity could target, and a run that mislinks nothing leaves it nothing — a defined floor, so the
        series stays plottable across a training run instead of dropping out at its best window.
        """
        return float(np.mean(self.p_chosen > self.p_true)) if len(self.p_true) else 0.0

    def means(self) -> tuple[float, float]:
        """Mean true-successor and chosen-neighbour probability over the mislinks — NaN when there are none."""
        if not len(self.p_true):
            return float("nan"), float("nan")
        return float(np.mean(self.p_true)), float(np.mean(self.p_chosen))

    @classmethod
    def pooled(cls, signals: Sequence["MislinkSignal"]) -> "MislinkSignal":
        """Every movie's mislinks as one signal — an eval reads its inversion over the whole proxy, not per video."""
        empty = np.empty(0, dtype=np.float64)
        return cls(
            p_true=np.concatenate([signal.p_true for signal in signals]) if signals else empty,
            p_chosen=np.concatenate([signal.p_chosen for signal in signals]) if signals else empty,
        )


class DenseDiagnosis:
    """Running a tracker on one movie and decomposing its annotated-edge fates + mislink signal."""

    @staticmethod
    def diagnose(
        tracker: CellTracker, path: Path, truth: TrackGraph, spacing: Spacing, device: str
    ) -> tuple[DenseFateDiagnosis, MislinkSignal]:
        """Run a tracker on one movie and decompose its annotated-edge fates and mislink signal — the reusable core.

        Takes the tracker so a caller can score a *modified* one (e.g. a finetuned edge head) against the same
        movie, not only the shipped mount `mounted` builds.
        """
        prediction = tracker.run(path.name, path)
        matching = DistanceMatcher(spacing=spacing).match(prediction, truth)
        affinity = tracker.edge_scorer.affinities(path, prediction, device)
        return (
            DenseFateDiagnosis.of(prediction, truth, matching),
            MislinkSignal.of(prediction, truth, matching, AffinityIndex.of(prediction, affinity)),
        )

    @staticmethod
    def _mounted(root: DataRoot, device: str, movie: str) -> tuple[DenseFateDiagnosis, MislinkSignal]:
        """Mount the shipped tracker, run it on `movie`, and decompose its annotated-edge fates and mislink signal."""
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            device,
        )
        return DenseDiagnosis.diagnose(tracker, proxy.paths[0], proxy.truths[0].graph, proxy.spacing, device)


def main() -> None:
    """Decompose the dense movie's annotated-edge fates and log each fraction — the tf5/zni gating number."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Decompose a movie's annotated-edge fates under the shipped tracker.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to decompose")
    args = parser.parse_args()

    diagnosis, signal = DenseDiagnosis._mounted(DataRoot.from_config(args.config), args.device, args.movie)  # noqa: SLF001
    counts = diagnosis.counts()
    total = sum(counts.values())
    logger.info("movie=%s  annotated edges=%d", args.movie, total)
    for fate, count in counts.items():
        logger.info("  %-18s %5d  %5.1f%%", fate.name, count, 100.0 * count / max(total, 1))
    mean_true, mean_chosen = signal.means()
    logger.info(
        "mislink signal: %d edges, affinity-inverted (retrain target) = %.1f%%  [mean P_true=%.3f vs P_chosen=%.3f]",
        len(signal.p_true),
        100.0 * signal.inverted_fraction(),
        mean_true,
        mean_chosen,
    )


if __name__ == "__main__":
    main()

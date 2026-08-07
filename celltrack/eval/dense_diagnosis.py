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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from jaxtyping import Int

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
    def _label(both, reproduced, source_linked, target_claimed):
        """Fold the per-edge boolean masks into one exclusive `Fate` code, most-specific outcome first."""
        fates = np.full(len(both), Fate.ENDPOINT_MISSING, dtype=np.int64)
        mislink = both & ~reproduced & source_linked
        fates[both & ~reproduced & ~source_linked] = Fate.SKIP
        fates[mislink & ~target_claimed] = Fate.MISLINK_FREE
        fates[mislink & target_claimed] = Fate.MISLINK_CONFLICT
        fates[reproduced] = Fate.CORRECT
        return fates


@dataclass(frozen=True)
class MislinkSignal:
    """For each mislinked edge, the affinity's probability of the true successor vs the wrong one it chose.

    A mislink is either a *signal* error — the edge transformer itself scores the wrong neighbour above the
    true successor (`p_chosen > p_true`), which only a better-trained affinity (bead zni) can fix — or a
    *cost* error — the affinity ranks the true successor higher, but the nearer wrong neighbour won on the
    distance term of `distance - bonus*P` anyway (`p_true >= p_chosen`), which is the bonus/gate cost lever
    already swept and refuted (e9b). The inverted fraction is the share a retrain could even target.
    """

    p_true: Float[np.ndarray, "m"]
    p_chosen: Float[np.ndarray, "m"]

    @classmethod
    def of(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, affinity: EdgeAffinity
    ) -> "MislinkSignal":
        """Read the affinity's true- and chosen-target probability for every mislinked annotated edge."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        predicted_of_truth = DenseFateDiagnosis._invert(matching, len(truth.node_ids))
        successors = Adjacency.of(prediction).successors
        timepoints = prediction.timepoints()
        mislinked = np.flatnonzero((fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE))
        edges = truth.edge_rows()[mislinked]
        pairs = [
            cls._probabilities(predicted_of_truth[source], predicted_of_truth[target], successors, timepoints, affinity)
            for source, target in edges
        ]
        return cls(
            p_true=np.array([true for true, _ in pairs], dtype=np.float64),
            p_chosen=np.array([chosen for _, chosen in pairs], dtype=np.float64),
        )

    def inverted_fraction(self) -> float:
        """The share of mislinks the affinity itself gets wrong (chosen scored above true) — the retrain target."""
        return float(np.mean(self.p_chosen > self.p_true)) if len(self.p_true) else float("nan")

    def means(self) -> tuple[float, float]:
        """Mean true-successor and chosen-neighbour probability over the mislinks — NaN when there are none."""
        if not len(self.p_true):
            return float("nan"), float("nan")
        return float(np.mean(self.p_true)), float(np.mean(self.p_chosen))

    @staticmethod
    def _probabilities(source_row, true_row, successors, timepoints, affinity) -> tuple[float, float]:
        """The affinity probability of the true successor and of the (first) successor the linker actually chose."""
        chosen_row = successors[source_row][0]
        timepoint = int(timepoints[source_row])
        probability = affinity.probabilities(timepoint)
        sources = np.flatnonzero(timepoints == timepoint)
        targets = np.flatnonzero(timepoints == timepoint + 1)
        source_local = int(np.searchsorted(sources, source_row))
        true_local = int(np.searchsorted(targets, true_row))
        chosen_local = int(np.searchsorted(targets, chosen_row))
        return float(probability[source_local, true_local]), float(probability[source_local, chosen_local])


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
            MislinkSignal.of(prediction, truth, matching, affinity),
        )

    @staticmethod
    def mounted(root: DataRoot, device: str, movie: str) -> tuple[DenseFateDiagnosis, MislinkSignal]:
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

    diagnosis, signal = DenseDiagnosis.mounted(DataRoot.from_config(args.config), args.device, args.movie)
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

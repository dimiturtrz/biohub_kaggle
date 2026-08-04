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

from celltrack.proxy import TestMovieProxy
from celltrack.tracker import CellTracker
from core.data.tracks import TrackGraph
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
    def _invert(matching: NodeMatching, truth_count: int) -> Int[np.ndarray, "g"]:
        """For each ground-truth node, the predicted row matched to it, or `UNMATCHED` — the matching reversed."""
        predicted_of_truth = np.full(truth_count, UNMATCHED, dtype=np.int64)
        matched = np.flatnonzero(matching.is_matched())
        predicted_of_truth[matching.gt_rows[matched]] = matched
        return predicted_of_truth

    @staticmethod
    def _label(both, reproduced, source_linked, target_claimed):  # noqa: ANN001, ANN205
        """Fold the per-edge boolean masks into one exclusive `Fate` code, most-specific outcome first."""
        fates = np.full(len(both), Fate.ENDPOINT_MISSING, dtype=np.int64)
        mislink = both & ~reproduced & source_linked
        fates[both & ~reproduced & ~source_linked] = Fate.SKIP
        fates[mislink & ~target_claimed] = Fate.MISLINK_FREE
        fates[mislink & target_claimed] = Fate.MISLINK_CONFLICT
        fates[reproduced] = Fate.CORRECT
        return fates


def _diagnose(root: DataRoot, device: str, movie: str) -> DenseFateDiagnosis:
    """Mount the shipped tracker, run it on `movie`, and decompose its annotated-edge fates."""
    proc = root.processed("biohub_cell_tracking")
    proxy = TestMovieProxy.load(root, (movie,))
    tracker = CellTracker.from_packs(
        proc / "reference/pilkwang/split_0",
        proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
        proc / "cache/responses",
        device,
    )
    path, truth = proxy.paths[0], proxy.truths[0]
    prediction = tracker.run(path.name, path)
    matching = DistanceMatcher(spacing=proxy.spacing).match(prediction, truth.graph)
    return DenseFateDiagnosis.of(prediction, truth.graph, matching)


def main() -> None:
    """Decompose the dense movie's annotated-edge fates and log each fraction — the tf5/zni gating number."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Decompose a movie's annotated-edge fates under the shipped tracker.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to decompose")
    args = parser.parse_args()

    counts = _diagnose(DataRoot.from_config(args.config), args.device, args.movie).counts()
    total = sum(counts.values())
    logger.info("movie=%s  annotated edges=%d", args.movie, total)
    for fate, count in counts.items():
        logger.info("  %-18s %5d  %5.1f%%", fate.name, count, 100.0 * count / max(total, 1))


if __name__ == "__main__":
    main()

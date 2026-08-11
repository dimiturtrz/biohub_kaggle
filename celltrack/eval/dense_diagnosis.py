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

That decomposition is GROUND-TRUTH-EDGE-CENTRIC, and Jaccard is `TP / (TP + FP + FN)`: an invented link
costs exactly what a lost one does, yet a predicted edge with no annotated edge under it has no true edge
whose fate it could be, so the `Fate` split accounts for every false negative and only for the false
positives that are some mislink's wrong half. `Charge` / `ChargeDiagnosis` are the mirror image — they
iterate the links the METRIC PRICED (`core.metrics.edges.ChargedLinks`, the metric's own repaired table)
and say why each was charged, which is the only way to see the rest of the FP term at all.
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
from celltrack.eval.proxy_eval import ConfigOverride
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.edges import ChargedLinks, EdgeCounts
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


class Charge(IntEnum):
    """Why the metric charged one EMITTED link — the mirror image of `Fate`, over predictions rather than truth.

    `Fate` iterates ANNOTATED edges, so it accounts for every false negative and only for the false positives
    that happen to be a mislink's wrong half. A predicted link with no annotated edge beneath it has no true
    edge whose fate it could be, so the rest of the FP term is invisible there — and Jaccard prices an FP
    exactly as dearly as an FN. These are the outcomes of a link the metric was in a position to judge:

    - ``TRUE`` — it lands on an annotated edge (the TP term; kept so the classes sum to the charged links).
    - ``SYNTHETIC`` — an endpoint is a node the detector never produced (a gap bridge inserted it). A
      post-processing question, not a linking one.
    - ``EXTRA_CHILD`` — the source's annotated successor WAS reproduced by another of our links, and this is a
      second one off the same node. An over-fork: the annotated edge reads CORRECT, so `Fate` never sees this.
    - ``MISLINK`` — the source has an annotated successor and we linked somewhere else instead. This is the
      one class `Fate` also sees (as ``MISLINK_CONFLICT`` + ``MISLINK_FREE``), so the two must agree.
    - ``SOURCE_ENDS`` — the annotated track ENDS at the source and we linked on past it. A boundary-price
      question — and, under a ~1-2% annotation, the class most at risk of being an artefact of where the
      annotation stopped rather than where the cell did.
    - ``TARGET_ORPHAN`` — the source matched no annotated node at all; the link is charged only because its
      target continues an annotated track, i.e. we invented a parent for a cell that has one.
    """

    TRUE = 0
    SYNTHETIC = 1
    EXTRA_CHILD = 2
    MISLINK = 3
    SOURCE_ENDS = 4
    TARGET_ORPHAN = 5


@dataclass(frozen=True)
class ChargeDiagnosis:
    """Each link the metric priced tagged with the `Charge` it was priced under — the FP term decomposed.

    Four cross-cutting masks travel with the labels because they answer questions the label alone cannot.
    `cross_lineage` says whether a wrong link joined two DIFFERENT annotated lineages (an affinity failure)
    or two nodes of the same one (a within-track ordering failure). `annotated_after` says, for a link off a
    track the annotation ends at, whether OTHER lineages are still annotated later in the movie: if none are,
    the annotation simply stopped and the link may well be true against the hidden denser labels, which makes
    that share a sparsity artefact rather than a lever. `target_matched` separates a link that ran into
    unannotated tissue from one that landed on another annotated cell.

    `steals_a_lost_target` is the one that decides how much the FP term is even WORTH: it marks a wrong link
    whose target is an annotated cell whose own annotated incoming edge we did not reproduce. Such a link is
    not an independent error but the other half of a false negative — one crowding failure charged twice, so
    repairing it wins two points, and the FP and FN terms must not be added up as separate levers.
    """

    charges: Int[np.ndarray, "c"]
    cross_lineage: Bool[np.ndarray, "c"]
    annotated_after: Bool[np.ndarray, "c"]
    target_matched: Bool[np.ndarray, "c"]
    steals_a_lost_target: Bool[np.ndarray, "c"]

    @classmethod
    def of(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, detections: TrackGraph
    ) -> "ChargeDiagnosis":
        """Label every charged link by why the metric priced it, over the metric's own repaired link table.

        `detections` is the pre-post-processing node set, which is the only way to tell an invented bridge node
        from a detected one: node IDs survive post-processing, so a prediction node whose ID the detector never
        emitted is synthetic.
        """
        links = ChargedLinks.of(prediction, truth, matching)
        annotated = Adjacency.of(truth)
        lineage = annotated.components()
        synthetic = np.isin(prediction.node_ids, detections.node_ids, invert=True)
        source_matched = links.matched_source != UNMATCHED
        target_matched = links.matched_target != UNMATCHED
        safe_source = np.where(source_matched, links.matched_source, 0)
        safe_target = np.where(target_matched, links.matched_target, 0)
        return cls(
            charges=cls._label(
                links,
                has_successor=source_matched & (annotated.out_degrees[safe_source] > 0),
                on_synthetic=synthetic[links.source] | synthetic[links.target],
                source_matched=source_matched,
            ),
            cross_lineage=source_matched & target_matched & (lineage[safe_source] != lineage[safe_target]),
            annotated_after=source_matched & cls._others_annotated_after(truth, lineage)[safe_source],
            target_matched=target_matched,
            steals_a_lost_target=target_matched
            & (annotated.in_degrees[safe_target] > 0)
            & ~cls._recovered(links, len(truth.node_ids))[safe_target],
        )

    @staticmethod
    def _recovered(links: ChargedLinks, truth_count: int) -> Bool[np.ndarray, "g"]:
        """Per annotated node: did we reproduce the annotated edge INTO it — i.e. did it get its true parent?"""
        recovered = np.zeros(truth_count, dtype=bool)
        recovered[links.matched_target[links.on_annotated_edge]] = True
        return recovered

    def counts(self) -> dict[Charge, int]:
        """How many charged links fell into each class — the decomposition as absolute link counts."""
        tally = np.bincount(self.charges, minlength=len(Charge))
        return {charge: int(tally[charge]) for charge in Charge}

    def split(self, charge: Charge, mask: Bool[np.ndarray, "c"]) -> tuple[int, int]:
        """How many links of one class the mask holds for, and how many it does not — a sub-class as a pair."""
        selected = self.charges == charge
        return int((selected & mask).sum()), int((selected & ~mask).sum())

    @staticmethod
    def _label(
        links: ChargedLinks,
        has_successor: Bool[np.ndarray, "c"],
        on_synthetic: Bool[np.ndarray, "c"],
        source_matched: Bool[np.ndarray, "c"],
    ) -> Int[np.ndarray, "c"]:
        """Fold the per-link masks into one exclusive `Charge`, written least-specific first so priority is order."""
        charges = np.full(len(links.source), Charge.TRUE, dtype=np.int64)
        wrong = ~links.on_annotated_edge
        reproduced_source = np.zeros(len(links.source), dtype=bool)
        correct_sources = links.source[links.on_annotated_edge]
        if len(correct_sources):
            reproduced_source = np.isin(links.source, correct_sources)
        charges[wrong] = Charge.TARGET_ORPHAN
        charges[wrong & source_matched & ~has_successor] = Charge.SOURCE_ENDS
        charges[wrong & has_successor] = Charge.MISLINK
        charges[wrong & has_successor & reproduced_source] = Charge.EXTRA_CHILD
        charges[wrong & on_synthetic] = Charge.SYNTHETIC
        return charges

    @staticmethod
    def _others_annotated_after(truth: TrackGraph, lineage: Int[np.ndarray, "g"]) -> Bool[np.ndarray, "g"]:
        """Per annotated node: does any node of a DIFFERENT lineage carry an annotation past this node's frame?

        This is what separates "the annotation stopped here" from "this track ended here": a track whose last
        node sits where nothing else is annotated either tells us only where the labelling ran out, while one
        that ends while other lineages are still being followed is a real ending the tracker linked past.
        """
        timepoints = truth.timepoints()
        horizon = int(timepoints.max()) + 1 if len(timepoints) else 1
        grid = np.zeros((int(lineage.max()) + 1 if len(lineage) else 1, horizon), dtype=np.int64)
        np.add.at(grid, (lineage, timepoints), 1)
        strictly_after = np.cumsum(grid[:, ::-1], axis=1)[:, ::-1] - grid
        return (strictly_after.sum(axis=0)[timepoints] - strictly_after[lineage, timepoints]) > 0


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
    def configured(root: DataRoot, device: str, config: TrackerConfig) -> CellTracker:
        """The same mount as `shipped`, at an ARBITRARY config — how a diagnosis reads a swept operating point.

        A fate decomposition of the shipped default answers "what is the residual made of"; a decomposition of
        the config a sweep arm ran answers "did the arm's score move for the reason claimed". The second needs
        the arm's whole config, not just its threshold, so the mount takes one.
        """
        proc = root.processed("biohub_cell_tracking")
        return CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            device,
            config,
        )

    @staticmethod
    def shipped(root: DataRoot, device: str, threshold: float | None = None) -> CellTracker:
        """The shipped dual-seed tracker mounted from the reference packs, cache-backed — what a diagnosis runs.

        The pack layout gets one home in `eval` rather than a copy per driver: a sibling diagnosis
        (`shortcut_diagnosis`) mounts through this instead of repeating the paths. `threshold` overrides the
        shipped detection threshold — the one operating point a diagnosis of the DETECTIONS has to be able to move.
        """
        config = TrackerConfig() if threshold is None else TrackerConfig(threshold=threshold)
        return DenseDiagnosis.configured(root, device, config)

    @staticmethod
    def charged(
        tracker: CellTracker, path: Path, truth: TrackGraph, spacing: Spacing
    ) -> tuple[EdgeCounts, DenseFateDiagnosis, ChargeDiagnosis]:
        """Both decompositions of one run — the annotated edges' fates and the charged links' classes, plus the
        counts they each have to reconcile against.

        One `run_scored` serves all three: the fates and the charges are pure functions of the same
        (prediction, truth, matching), and the charge classes additionally need the DETECTION node set, which
        only the scored run hands back. Returning the raw `EdgeCounts` alongside is what makes the reconciliation
        checkable rather than asserted — the classes must sum to TP + FP exactly.
        """
        tracked = tracker.run_scored(path.name, path)
        matching = DistanceMatcher(spacing=spacing).match(tracked.graph, truth)
        return (
            EdgeCounts.of(tracked.graph, truth, matching),
            DenseFateDiagnosis.of(tracked.graph, truth, matching),
            ChargeDiagnosis.of(tracked.graph, truth, matching, tracked.detections),
        )

    @staticmethod
    def _mounted(
        root: DataRoot, device: str, movie: str, config: TrackerConfig | None = None
    ) -> tuple[DenseFateDiagnosis, MislinkSignal]:
        """Mount the tracker at `config` (the shipped default when unset), run it on `movie`, and decompose it."""
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, device, config if config is not None else TrackerConfig())
        return DenseDiagnosis.diagnose(tracker, proxy.paths[0], proxy.truths[0].graph, proxy.spacing, device)

    @staticmethod
    def _charged(
        root: DataRoot, device: str, movie: str, config: TrackerConfig
    ) -> tuple[EdgeCounts, DenseFateDiagnosis, ChargeDiagnosis]:
        """Mount the tracker at `config`, run it on `movie`, and return both decompositions of that one run."""
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, device, config)
        return DenseDiagnosis.charged(tracker, proxy.paths[0], proxy.truths[0].graph, proxy.spacing)


def _report_charges(counts: EdgeCounts, fates: DenseFateDiagnosis, charges: ChargeDiagnosis, movie: str) -> None:
    """Log the false-positive decomposition beside the counts it has to reconcile with — the FP-term report."""
    tally = charges.counts()
    logger.info("movie=%s  TP=%d FP=%d FN=%d  jaccard=%.4f", movie, counts.tp, counts.fp, counts.fn, counts.jaccard())
    logger.info("charged links=%d  (TP + FP = %d)", sum(tally.values()), counts.tp + counts.fp)
    logger.info("  %-14s %5d  (the TP term)", Charge.TRUE.name, tally[Charge.TRUE])
    for charge, count in tally.items():
        if charge is not Charge.TRUE:
            logger.info("  %-14s %5d  %5.1f%% of FP", charge.name, count, 100.0 * count / max(counts.fp, 1))
    by_fate = fates.counts()
    fate_mislinks = by_fate[Fate.MISLINK_CONFLICT] + by_fate[Fate.MISLINK_FREE]
    logger.info(
        "reconcile: Charge.MISLINK=%d vs Fate.MISLINK_CONFLICT+FREE=%d  %s",
        tally[Charge.MISLINK],
        fate_mislinks,
        "AGREE" if tally[Charge.MISLINK] == fate_mislinks else "DISAGREE",
    )
    for charge in (Charge.SYNTHETIC, Charge.EXTRA_CHILD, Charge.MISLINK, Charge.SOURCE_ENDS, Charge.TARGET_ORPHAN):
        annotated_target, unannotated_target = charges.split(charge, charges.target_matched)
        cross, _ = charges.split(charge, charges.cross_lineage)
        stolen, _ = charges.split(charge, charges.steals_a_lost_target)
        later, stopped = charges.split(charge, charges.annotated_after)
        logger.info(
            "  %-14s target annotated %d (cross-lineage %d, stole a lost target %d) / unannotated %d"
            "  | annotation runs on past source %d / stops %d",
            charge.name,
            annotated_target,
            cross,
            stolen,
            unannotated_target,
            later,
            stopped,
        )


def main() -> None:
    """Decompose the dense movie's annotated-edge fates and log each fraction — the tf5/zni gating number."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Decompose a movie's annotated-edge fates under the shipped tracker.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to decompose")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any tracker knob, exactly as `proxy_eval --set` does, so a sweep arm can be decomposed",
    )
    parser.add_argument(
        "--charges",
        action="store_true",
        help="decompose the FALSE POSITIVES too — every charged link by why the metric priced it",
    )
    args = parser.parse_args()

    config = TrackerConfig()
    for assignment in args.overrides:
        config = ConfigOverride.apply(config, assignment)
    logger.info("set=%s", list(args.overrides))
    root = DataRoot.from_config(args.config)
    if args.charges:
        _report_charges(*DenseDiagnosis._charged(root, args.device, args.movie, config), movie=args.movie)  # noqa: SLF001
        return
    diagnosis, signal = DenseDiagnosis._mounted(root, args.device, args.movie, config)  # noqa: SLF001
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

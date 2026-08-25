"""Prove our reimplemented postproc makes the SAME decisions as the published Kaggle kernel — code, not proxy.

The local proxy metric is saturated/blind above ~0.89, so it cannot arbitrate whether our reimplementation of
a frontier post-processing stage is faithful. This oracle answers the question the proxy cannot: given ONE
tracker state (nodes + coordinates + edges), does OUR stage emit the same edges the kernel's own function
emits? Agreement is measured as a frozenset diff of `(source_id, target_id)` node-id pairs, so a single flipped
decision shows up as a named divergent edge rather than a fractional score move.

The kernel reference is a flattened Kaggle notebook whose top level installs and imports a heavy dependency
stack (tracksdata, zarr, ilpy, pyscipopt). Importing it locally is neither possible nor desirable, and copying
its code into this package would be vendoring. Instead `KernelReference` parses the reference file with `ast`
and execs ONLY the named function's own source segment in a namespace seeded with the kernel's constants and
its small pure-geometry helpers — so the diff runs the kernel's ACTUAL bytes, against our stage, with nothing
transcribed but the constants (which ARE the fidelity subject and are cited to the reference line that sets
them). `KernelReference.kernel_namespace` performs that extract-and-exec.

Two components are covered, each a report method of `FidelityOracle` that maps one `TrackGraph` to the set of
edges its two sides emit (`FidelityReport`):

* `divsub_report` — the safe-division recovery (`add_safe_divisions_postlink`). Self-contained and fully
  quantified, so it is the primary check: our `AffinityDivisionRecovery` is configured to the kernel's ported
  divsub arm (geometric candidacy, geometry ranking at sister-weight 0.15, gates 8.0 / 11.0 / 10.0,
  mutual-nearest and C3 divergence both on) and the ADDED fork edges are diffed against the kernel function's
  added edges. The caps default to the kernel's shipped fractions; passing them at 1.0 lifts both sides to
  isolate the gate/score axis from cap derivation.
* `motion_relink_report` — the two-pass Hungarian re-association (`motion_relink_edges`). Both sides build
  every edge from scratch off the node set, so the full emitted edge set is diffed.

Run `python -m celltrack.eval.fidelity_oracle` for the synthetic-state report (no GPU, deterministic). The
synthetic scenarios are constructed to place genuine divisions and genuine motion tracks that clear every gate,
so a divergence in the diff is a divergence in OUR logic, not an artefact of an input neither side accepts.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Float

from celltrack.eval.kernel_reference import (
    DIVSUB_REFERENCE,
    MOTION_REFERENCE,
    SPACING,
    VOXEL_SCALE_UM,
    KernelReference,
)
from celltrack.linkers.motion_linking import MotionHungarianLinker
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing

logger = logging.getLogger(__name__)

# The kernel's ACTIVE divsub constants after the "ported_tightrad" arm override (divsub reference cell lines
# 65-70) — these, not the cell-5 preset, are what the 0.926 kernel actually runs. Named constants, no magic
# numbers: each is the fidelity subject, cited to the reference line that sets it.
_DIVSUB_PARENT_MAX_UM = 8.0  # BIOHUB_SAFE_DIV_MAX_UM, reference line 66
_DIVSUB_SISTER_MAX_UM = 11.0  # BIOHUB_SAFE_DIV_SISTER_MAX_UM, reference line 67
_DIVSUB_EXISTING_CHILD_MAX_UM = 10.0  # BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM, reference line 68
_DIVSUB_DIVERGE_UM = 2.25  # BIOHUB_SAFE_DIV_DIVERGE_UM, reference line 69
_DIVSUB_FRAME_FRAC_CAP = 0.0076  # BIOHUB_SAFE_DIV_FRAME_FRAC_CAP, reference line 42
_DIVSUB_GLOBAL_FRAC_CAP = 0.00375  # BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP, reference line 43
_DIVSUB_SISTER_WEIGHT = 0.15  # the score's sister term, reference line 2319: parent_dist + 0.15 * sister_dist

# The kernel's ACTIVE motion-relink constants (motion reference lines 173-176, with the rockerritesh config
# raising the learned bonus to 1.0 at cell line 26).
_MOTION_TIGHT_UM = 6.0  # BIOHUB_MOTION_RELINK_TIGHT_UM, reference line 173
_MOTION_RELAXED_UM = 10.0  # BIOHUB_MOTION_RELINK_RELAXED_UM, reference line 174
_MOTION_VELOCITY_WEIGHT = 0.5  # BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT, reference line 175
_MOTION_LEARNED_BONUS = 1.0  # BIOHUB_MOTION_RELINK_LEARNED_BONUS, rockerritesh cell line 26
_MOTION_MAX_FRAME_NODES = 100_000  # large so the synthetic frames are never skipped

# A fork budget far above any synthetic scenario's fork count, so the oracle measures the GATES' agreement
# rather than a cap that binds differently on the two sides (that cap divergence is reported by the audit).
_UNBOUNDED_FORKS = 10_000

# A fractional cap of 1.0 admits every proposal, so the caps never bind on either side.
_CAPS_LIFTED = 1.0

_Edge = tuple[int, int]


@dataclass(frozen=True)
class _ZeroAffinity:
    """A learned-association test double returning a zero matrix per gap: the kernel divsub arm and our
    geometric candidacy both decide on geometry alone, so the head's probabilities never enter either side's
    decision. It exists only to satisfy the `EdgeAffinity` protocol the recovery stage consumes."""

    counts: dict[int, int]

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:  # devtools-ignore: test-mirror
        sources = self.counts.get(timepoint)
        targets = self.counts.get(timepoint + 1)
        if not sources or not targets:
            return None
        return np.zeros((sources, targets), dtype=float)


@dataclass(frozen=True)
class FidelityReport:
    """One component's edge-level agreement between our reimplementation and the kernel function.

    A pure value object: `name` and the two emitted edge sets, with `agreement` and `divergent` derived from
    them. `FidelityOracle` builds it, scores it, and logs the verdict with its divergent edges.
    """

    name: str
    ours: set[_Edge]
    kernel: set[_Edge]

    def agreement(self) -> float:
        """Jaccard agreement of the two emitted edge sets; 1.0 when both are empty (a vacuous but honest tie)."""
        union = self.ours | self.kernel
        if not union:
            return 1.0
        return len(self.ours & self.kernel) / len(union)

    def divergent(self) -> tuple[set[_Edge], set[_Edge]]:
        """`(ours_only, kernel_only)` — our over-emissions and the kernel decisions our reimplementation misses."""
        return self.ours - self.kernel, self.kernel - self.ours


class FidelityOracle:
    """Diffs each of our post-processing stages against the published kernel's own function on one tracker
    state, holding the shared voxel `Spacing` the geometry and the kernel namespace both speak.

    A stage is named on the command line and looked up in a dispatch map (never an `if`/`elif` on the name):
    each entry is a report builder producing a `FidelityReport` of ours-vs-kernel emitted edges.
    """

    def __init__(self, spacing: Spacing = SPACING) -> None:
        self.spacing = spacing

    @staticmethod
    def synthetic_state() -> TrackGraph:
        """A deterministic tracker state exercising both stages: a clean division, motion tracks, and gate decoys.

        Displacements are placed along x (0.40625 um/voxel) so gate arithmetic is transparent. The graph holds a
        genuine mid-track division whose daughters are mutual-nearest orphans and diverge at t+2 (both sides must
        add it), plus a track-start "parent" whose fork the kernel's C1 gate rejects, so the diff surfaces whether
        our reimplementation shares that gate.
        """
        nodes: list[tuple[int, int, int, int, int]] = []  # (id, t, z, y, x)
        edges: list[_Edge] = []

        def node(node_id: int, t: int, x: int, *, z: int = 10, y: int = 10) -> None:
            nodes.append((node_id, t, z, y, x))

        # A mid-track parent (1 has predecessor 0) that divides at t=2 into kept child 3 and orphan daughter 4;
        # both continue at t=3 (nodes 5, 6) and separate, clearing the C3 divergence gate.
        node(0, 0, 100)
        node(1, 1, 100)
        node(3, 2, 96)
        node(4, 2, 108)
        node(5, 3, 90)
        node(6, 3, 116)
        edges.extend([(0, 1), (1, 3), (3, 5), (4, 6)])

        # A straightforward two-frame motion track, well separated from the division, that both linkers must join.
        node(20, 0, 300)
        node(21, 1, 305)
        node(22, 2, 309)
        edges.extend([(20, 21), (21, 22)])

        # A C1 DECOY: a track-START parent (30 has no predecessor) whose single child 31 and diverging mutual-nearest
        # orphan 32 clear every gate. The kernel's C1 gate rejects it because 30 is not mid-track (reference line
        # 2268); our reimplementation has no C1 gate, so it should ADD 30->32 — the divergence the oracle surfaces.
        node(30, 0, 1200)
        node(31, 1, 1196)
        node(32, 1, 1208)
        node(33, 2, 1188)
        node(34, 2, 1216)
        edges.extend([(30, 31), (31, 33), (32, 34)])

        coordinates = np.array([[t, z, y, x] for _, t, z, y, x in nodes], dtype=np.int64)
        node_ids = np.array([node_id for node_id, *_ in nodes], dtype=np.int64)
        return TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=np.array(edges, dtype=np.int64))

    def divsub_report(
        self,
        graph: TrackGraph,
        global_frac: float = _DIVSUB_GLOBAL_FRAC_CAP,
        frame_frac: float = _DIVSUB_FRAME_FRAC_CAP,
    ) -> FidelityReport:
        """Fidelity of the safe-division recovery: the ADDED fork edges, ours vs `add_safe_divisions_postlink`.

        The default runs the kernel's SHIPPED fractional caps (0.00375 global / 0.0076 per frame), so the diff
        includes the cap-derivation axis. Passing `global_frac=frame_frac=1.0` lifts both caps on BOTH sides
        identically, so the caps never bind and the diff measures ONLY the gates, the score, the mutual-nearest
        rule and the C3 divergence rule — separating "do the gates agree" from "do the caps agree".
        """
        return FidelityReport(
            name="divsub" if global_frac == _DIVSUB_GLOBAL_FRAC_CAP else "divsub_gates",
            ours=self._divsub_ours(graph, global_frac),
            kernel=self._divsub_kernel(graph, global_frac, frame_frac),
        )

    def divsub_faithful_report(self, graph: TrackGraph) -> FidelityReport:
        """Fidelity of the SHIPPED divsub arm (`kernel_faithful=True`) vs `add_safe_divisions_postlink`.

        This exercises the byte-faithful replica the frontier config actually ships, not the deprecated per-flag
        bracket the `divsub`/`divsub_gates` components characterise (those document its missing C1 gate and its
        cap-floor divergence). The ship path runs the kernel's own gate bracket, C1/C2/C3 gates and internal
        two-cap admission, so it is diffed against the kernel at the shipped fractional caps.
        """
        return FidelityReport(
            name="divsub_faithful",
            ours=self._divsub_ours_faithful(graph),
            kernel=self._divsub_kernel(graph, _DIVSUB_GLOBAL_FRAC_CAP, _DIVSUB_FRAME_FRAC_CAP),
        )

    def motion_relink_report(self, graph: TrackGraph) -> FidelityReport:
        """Fidelity of the two-pass motion re-association: every emitted edge, ours vs `motion_relink_edges`."""
        return FidelityReport(name="motion_relink", ours=self._motion_ours(graph), kernel=self._motion_kernel(graph))

    @staticmethod
    def _divsub_config(global_frac: float) -> AffinityDivisionConfig:
        """Our `AffinityDivisionRecovery` configured to the kernel's ported divsub arm."""
        return AffinityDivisionConfig(
            ranking="geometry",
            candidacy="geometric",
            sister_weight=_DIVSUB_SISTER_WEIGHT,
            parent_gate_um=_DIVSUB_PARENT_MAX_UM,
            sister_gate_um=_DIVSUB_SISTER_MAX_UM,
            existing_child_gate_um=_DIVSUB_EXISTING_CHILD_MAX_UM,
            require_mutual_nearest=True,
            require_c3_divergence=True,
            c3_divergence_um=_DIVSUB_DIVERGE_UM,
            max_added_forks=_UNBOUNDED_FORKS,
            max_fork_edge_fraction=global_frac,
        )

    @staticmethod
    def _timepoint_counts(graph: TrackGraph) -> dict[int, int]:
        """Node count per timepoint — the per-frame shape the zero-affinity stand-in reports."""
        timepoints = graph.timepoints()
        return {int(t): int(np.count_nonzero(timepoints == t)) for t in np.unique(timepoints).tolist()}

    def _divsub_ours(self, graph: TrackGraph, global_frac: float) -> set[_Edge]:
        """The fork edges OUR `AffinityDivisionRecovery` adds under the kernel's ported divsub arm."""
        affinity = _ZeroAffinity(self._timepoint_counts(graph))
        recovery = self._divsub_config(global_frac).build(
            self.spacing, affinity, min_track_length=1, gate_um=_DIVSUB_PARENT_MAX_UM
        )
        before = {(int(s), int(t)) for s, t in graph.edges.tolist()}
        return {(int(s), int(t)) for s, t in recovery.transform(graph).edges.tolist()} - before

    def _divsub_ours_faithful(self, graph: TrackGraph) -> set[_Edge]:
        """The fork edges the SHIP path (`kernel_faithful=True`) adds — its own gate bracket, C1, C3 and two-cap."""
        affinity = _ZeroAffinity(self._timepoint_counts(graph))
        recovery = AffinityDivisionConfig(kernel_faithful=True).build(
            self.spacing, affinity, min_track_length=1, gate_um=_DIVSUB_PARENT_MAX_UM
        )
        before = {(int(s), int(t)) for s, t in graph.edges.tolist()}
        return {(int(s), int(t)) for s, t in recovery.transform(graph).edges.tolist()} - before

    def _divsub_kernel(self, graph: TrackGraph, global_frac: float, frame_frac: float) -> set[_Edge]:
        """The fork edges the kernel's own `add_safe_divisions_postlink` adds for this state."""
        namespace = KernelReference.kernel_namespace(
            DIVSUB_REFERENCE,
            ["add_safe_divisions_postlink"],
            {
                "OUTPUT_SAFE_DIVISIONS": True,
                "SAFE_DIV_MAX_UM": _DIVSUB_PARENT_MAX_UM,
                "SAFE_DIV_SISTER_MAX_UM": _DIVSUB_SISTER_MAX_UM,
                "SAFE_DIV_EXISTING_CHILD_MAX_UM": _DIVSUB_EXISTING_CHILD_MAX_UM,
                "SAFE_DIV_DIVERGE_UM": _DIVSUB_DIVERGE_UM,
                "SAFE_DIV_FRAME_FRAC_CAP": frame_frac,
                "SAFE_DIV_GLOBAL_FRAC_CAP": global_frac,
                "DEEPCENTER_SAFE_DIV_VETO": False,
                "VOXEL_SCALE_UM": VOXEL_SCALE_UM,
                **KernelReference.divsub_helpers(),
            },
        )
        add_safe_divisions_postlink = cast(
            "Callable[..., list[dict[str, object]]]", namespace["add_safe_divisions_postlink"]
        )
        returned = add_safe_divisions_postlink(
            KernelReference.nodes_by_id(graph), KernelReference.edge_dicts(graph), defaultdict(int)
        )
        return {self._edge_ids(edge) for edge in returned if edge.get("safe_division")}

    def _motion_ours(self, graph: TrackGraph) -> set[_Edge]:
        """Every edge OUR two-pass `MotionHungarianLinker` builds off the node set."""
        linker = MotionHungarianLinker(
            spacing=self.spacing,
            tight_gate_um=_MOTION_TIGHT_UM,
            loose_gate_um=_MOTION_RELAXED_UM,
            affinity=None,
            affinity_bonus=_MOTION_LEARNED_BONUS,
        )
        return {(int(s), int(t)) for s, t in linker.link(graph.without_edges()).edges.tolist()}

    @staticmethod
    def _motion_kernel(graph: TrackGraph) -> set[_Edge]:
        """Every edge the kernel's own `motion_relink_edges` builds off the node set."""
        namespace = KernelReference.kernel_namespace(
            MOTION_REFERENCE,
            ["motion_relink_edges", "_position_um"],
            {
                "OUTPUT_MOTION_RELINK": True,
                "MOTION_RELINK_MAX_FRAME_NODES": _MOTION_MAX_FRAME_NODES,
                "MOTION_RELINK_VELOCITY_WEIGHT": _MOTION_VELOCITY_WEIGHT,
                "MOTION_RELINK_LEARNED_BONUS": _MOTION_LEARNED_BONUS,
                "MOTION_RELINK_TIGHT_UM": _MOTION_TIGHT_UM,
                "MOTION_RELINK_RELAXED_UM": _MOTION_RELAXED_UM,
                "VOXEL_SCALE_UM": VOXEL_SCALE_UM,
            },
        )
        motion_relink_edges = cast("Callable[..., list[dict[str, object]]]", namespace["motion_relink_edges"])
        emitted = motion_relink_edges(KernelReference.nodes_by_id(graph), defaultdict(int))
        return {FidelityOracle._edge_ids(edge) for edge in emitted}

    @staticmethod
    def _edge_ids(edge: dict[str, object]) -> _Edge:
        """The kernel edge dict's `(source_id, target_id)` as ints, coercing the exec'd namespace's `object` values."""
        return (int(cast("int", edge["source_id"])), int(cast("int", edge["target_id"])))

    @staticmethod
    def _coordinate_lookup(graph: TrackGraph) -> dict[int, tuple[float, float, float, float]]:
        """Each node id -> its `(t, z, y, x)`, for annotating divergent edges in the report."""
        return {
            int(node_id): (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for node_id, row in zip(graph.node_ids.tolist(), graph.coordinates.tolist(), strict=True)
        }

    @staticmethod
    def _log_report(report: FidelityReport, coordinates: dict[int, tuple[float, float, float, float]]) -> None:
        """Emit the verdict and, on divergence, the divergent edges with their endpoints' `(t, z, y, x)`."""
        ours_only, kernel_only = report.divergent()
        logger.info(
            "%-14s agreement=%.4f  ours=%d kernel=%d  ours_only=%d kernel_only=%d",
            report.name,
            report.agreement(),
            len(report.ours),
            len(report.kernel),
            len(ours_only),
            len(kernel_only),
        )
        for label, edges in (("ours_only", ours_only), ("kernel_only", kernel_only)):
            for source, target in sorted(edges):
                logger.info("    %s  %d->%d  %s -> %s", label, source, target, coordinates[source], coordinates[target])

    def dispatch(self) -> dict[str, Callable[[TrackGraph], FidelityReport]]:
        """Component name -> its report builder; the single source of both the CLI choices and the run map."""
        return {
            "divsub": self.divsub_report,
            "divsub_gates": lambda graph: self.divsub_report(graph, _CAPS_LIFTED, _CAPS_LIFTED),
            "divsub_faithful": self.divsub_faithful_report,
            "motion_relink": self.motion_relink_report,
        }

    def run(self, graph: TrackGraph, component_names: list[str]) -> list[FidelityReport]:
        """Score every named component on a state and log each verdict with its divergent edges."""
        dispatch = self.dispatch()
        coordinates = self._coordinate_lookup(graph)
        reports = [dispatch[name](graph) for name in component_names]
        for report in reports:
            self._log_report(report, coordinates)
        return reports


def main() -> None:
    """Run the fidelity oracle on the synthetic tracker state and log per-component edge agreement."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    oracle = FidelityOracle()
    dispatch = oracle.dispatch()
    parser = argparse.ArgumentParser(description="Diff our postproc against the published kernel's own functions.")
    parser.add_argument(
        "--component",
        choices=[*dispatch, "all"],
        default="all",
        help="which stage to check (default: all)",
    )
    parsed = parser.parse_args()
    chosen = list(dispatch) if parsed.component == "all" else [parsed.component]
    oracle.run(oracle.synthetic_state(), chosen)


if __name__ == "__main__":
    main()

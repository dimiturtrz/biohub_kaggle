"""Run the oracle fork arms and write the geffs the official ruler scores.

`celltrack.postproc.division_reparent` holds the planners; this is the instrument that feeds them a matched
tracksdata pair and prices what they do. It lives in `eval` because that is what it is: an ORACLE, reading the
annotation to size a move, never a step in the shipped pipeline.

Three arms, all scored through `celltrack.eval.official_ruler` against the champion's 0.8463:

- `--reparent` alone: +0.0031. Edge-free by construction and confirmed so (eTP +8, eFP -4, eFN -8).
- `--cull-forks --blind`: +0.0034. SHIPPABLE -- no annotation is read -- and sub-floor.
- `--cull-forks` with the annotation: +0.1263, the largest number measured on this ruler and entirely the
  value of the annotation. `celltrack.eval.fork_rank_report` shows no GT-free cue reaches past +0.008 of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from celltrack.postproc.division_reparent import MIN_DAUGHTERS, ORACLE_EDGE_PROB, ForkCull, ReparentPlan

ORACLE_MATCHED_RANK = 1000.0


@dataclass(frozen=True)
class OracleArms:
    """A matched tracksdata pair, read down into the arguments each planner asks for.

    The graphs stay `object`: tracksdata's own signatures satisfy no structural protocol, and this class is the
    one place in the module that touches them.
    """

    predicted: object
    truth: object
    keys: object

    def divisions(self) -> list[tuple[int, list[int]]]:
        """Every annotated division as `(divider, daughters)` — what `plan_reparents` walks."""
        node_ids = self.truth.node_ids()  # type: ignore[attr-defined]
        out_degree = self.truth.out_degree(node_ids)  # type: ignore[attr-defined]
        return [
            (int(node), [int(child) for child in self.truth.successors(int(node))])  # type: ignore[attr-defined]
            for node, degree in zip(node_ids, out_degree, strict=True)
            if degree >= MIN_DAUGHTERS
        ]

    def predecessors_of(self, row: int) -> list[int]:
        """Whoever already links into a predicted row — the claimant a re-parent has to take her from."""
        return [int(source) for source in self.predicted.predecessors(row)]  # type: ignore[attr-defined]

    def cull(self, *, blind: bool) -> tuple[ForkCull, ...]:
        """Plan the fork cull, with the annotation or without it.

        `blind` withholds the annotation entirely: no fork is protected and the surviving branch is chosen on
        `edge_prob` alone. That arm is SHIPPABLE, and the gap between it and the oracle is what a learned fork
        discriminator would have to earn.
        """
        ranks = self._edge_ranks()
        if blind:
            unannotated: dict[int, int] = {}
            return ForkCull.plan(
                self._forks(),
                unannotated,
                self._never,
                lambda source, target: ranks.get((source, target), 0.0) % ORACLE_MATCHED_RANK,
            )
        dividers = {divider for divider, _ in self.divisions()}
        return ForkCull.plan(
            self._forks(),
            self._annotated_of(),
            dividers.__contains__,
            lambda source, target: ranks.get((source, target), 0.0),
        )

    @staticmethod
    def _never(_: int) -> bool:
        """No annotation is consulted: the blind arm cuts every fork and keeps its likeliest branch."""
        return False

    def _forks(self) -> list[tuple[int, list[int]]]:
        rows = self.predicted.node_ids()  # type: ignore[attr-defined]
        out_degree = self.predicted.out_degree(rows)  # type: ignore[attr-defined]
        return [
            (int(row), [int(child) for child in self.predicted.successors(int(row))])  # type: ignore[attr-defined]
            for row, degree in zip(rows, out_degree, strict=True)
            if degree >= MIN_DAUGHTERS
        ]

    def _annotated_of(self) -> dict[int, int]:
        node_id = self.keys.NODE_ID  # type: ignore[attr-defined]
        matched_id = self.keys.MATCHED_NODE_ID  # type: ignore[attr-defined]
        matched = self.predicted.node_attrs(attr_keys=[node_id, matched_id])  # type: ignore[attr-defined]
        return {
            int(row): int(gt) for row, gt in zip(matched[node_id], matched[matched_id], strict=True) if int(gt) >= 0
        }

    def matched_of(self) -> dict[int, int]:
        """The annotation's own index: which predicted row each annotated node landed on."""
        return {gt: row for row, gt in self._annotated_of().items()}

    def _edge_ranks(self) -> dict[tuple[int, int], float]:
        """`ORACLE_MATCHED_RANK` for an edge the annotation agrees with, plus its affinity as a tiebreak."""
        edges = self.predicted.edge_attrs()  # type: ignore[attr-defined]
        return {
            (int(source), int(target)): ORACLE_MATCHED_RANK * bool(mask) + float(prob)
            for source, target, mask, prob in zip(
                edges[self.keys.EDGE_SOURCE],  # type: ignore[attr-defined]
                edges[self.keys.EDGE_TARGET],  # type: ignore[attr-defined]
                edges[self.keys.MATCHED_EDGE_MASK],  # type: ignore[attr-defined]
                edges["edge_prob"],
                strict=True,
            )
        }


def main() -> None:
    import argparse  # noqa: PLC0415
    import logging  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import tracksdata as td  # noqa: PLC0415
    from biohub_tracking.io import open_dataset  # type: ignore[missing-import]  # noqa: PLC0415

    from celltrack.eval.official_ruler import COMPETITION, MAX_MATCH_DISTANCE_UM, TEST_MOVIES  # noqa: PLC0415
    from core.paths import DataRoot  # noqa: PLC0415

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = DataRoot.from_config(Path("paths.yaml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="dir of <stem>.geff to re-parent")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the re-parented geffs are written")
    parser.add_argument("--reparent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cull-forks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blind", action=argparse.BooleanOptionalAction, default=False)
    arguments = parser.parse_args()
    ground_truth = data.raw(COMPETITION) / "train"
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in TEST_MOVIES:
        geff = arguments.pred_dir / f"{stem}.geff"
        if not geff.exists():
            log.info("%s: MISSING -- skipped", stem)
            continue
        loaded = td.graph.IndexedRXGraph.from_geff(geff)
        predicted = loaded[0] if isinstance(loaded, tuple) else loaded
        dataset = open_dataset(ground_truth / stem, require_tracks=True)
        truth = dataset.tracks
        predicted.match(
            truth, matching=td.metrics.DistanceMatching(max_distance=MAX_MATCH_DISTANCE_UM, scale=dataset.scale)
        )
        arms = OracleArms(predicted=predicted, truth=truth, keys=td.DEFAULT_ATTR_KEYS)
        plan = ReparentPlan.of(arms.divisions(), arms.matched_of(), arms.predecessors_of)
        if arguments.reparent:
            for move in plan.moves:
                if move.claimed_by is not None:
                    predicted.remove_edge(move.claimed_by, move.daughter)
                predicted.add_edge(
                    move.parent, move.daughter, {"edge_prob": ORACLE_EDGE_PROB, "edge_dist": 0.0}, validate_keys=False
                )
        culls: tuple[ForkCull, ...] = ()
        if arguments.cull_forks:
            culls = arms.cull(blind=arguments.blind)
            for cull in culls:
                for child in cull.dropped:
                    predicted.remove_edge(cull.parent, child)
        out = arguments.out_dir / f"{stem}.geff"
        shutil.rmtree(out, ignore_errors=True)
        predicted.to_geff(out)
        log.info(
            "%s: divisions completed=%d unreachable=%d | moves=%d of which thefts=%d | forks culled=%d edges=%d",
            stem,
            plan.completed,
            plan.unreachable,
            len(plan.moves),
            plan.thefts(),
            len(culls),
            sum(len(cull.dropped) for cull in culls),
        )


if __name__ == "__main__":
    main()

"""Union the several 0.900-tying tracker configs into one consensus graph and score its edge Jaccard.

The decorrelation instrument (:mod:`celltrack.analysis.tracker_decorrelation`) measures WHETHER the
0.900-tying configs disagree on links; this module cashes that disagreement in. If they disagree on
edges while tying on the leaderboard, each config recovers real links the others miss — so a vote-union
over their graphs is a recall lever the champion structurally lacks, and the vote threshold `k` trades
that recall against the precision of agreement.

Per movie the densest config (most detected nodes) is the reference node backbone; every config's edges
are mapped onto it by the competition's own centroid matching and voted; edges with at least `k` votes
form the consensus graph, scored against the sparse GT exactly as the leaderboard scores. `k=1` is the
full union (max recall), `k=N` the intersection (max precision). The keep/kill: does any `k` beat the
best single config's pooled edge Jaccard by more than the ~0.01 noise floor? The detector responses are
cached, so this is CPU-only and leaves the card free.

CLI: `python -m celltrack.analysis.consensus_linker [--device cpu]`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from celltrack.analysis.tracker_decorrelation import DecorrelationRun
from core.data.tracks import TrackGraph
from core.metrics.edges import EdgeCounts
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CHAMPION = "shipped_thr097"  # the operating point currently on the leaderboard at 0.900


@dataclass(frozen=True)
class ConsensusRun(DecorrelationRun):
    """A DecorrelationRun that votes the 0.900-tying configs' edges into one consensus graph (shares the mount)."""

    @staticmethod
    def _reference_name(graphs: dict[str, TrackGraph]) -> str:
        """The densest config for this movie — the most-detected node set is the best recall backbone to vote onto."""
        return max(graphs, key=lambda name: len(graphs[name].node_ids))

    @staticmethod
    def _vote(
        graphs: dict[str, TrackGraph], reference: TrackGraph, matcher: DistanceMatcher
    ) -> dict[tuple[int, int], set[str]]:
        """Per reference-node-pair, the set of configs that draw that edge once their endpoints map onto the reference.

        Each config's nodes are matched to the reference by centroid distance; an edge whose two endpoints both
        land on reference nodes casts one vote for that reference edge. Endpoints the reference never detected
        (no match within 7 um) drop — that config simply cannot vote on a link the backbone has no node for.
        Returning the voter SET (not just a count) lets the caller ask both how many agreed and whether a
        particular config — the shipped champion — was among them.
        """
        voters: dict[tuple[int, int], set[str]] = {}
        for name, graph in graphs.items():
            matching: NodeMatching = matcher.match(graph, reference)
            onto = matching.gt_rows
            for source_row, target_row in graph.edge_rows():
                mapped_source, mapped_target = onto[source_row], onto[target_row]
                if UNMATCHED not in (mapped_source, mapped_target):
                    voters.setdefault((int(mapped_source), int(mapped_target)), set()).add(name)
        return voters

    @staticmethod
    def _graph_from_pairs(reference: TrackGraph, kept: list[tuple[int, int]]) -> TrackGraph:
        """The reference nodes carrying exactly the row-pair edges in `kept`, back in node-id space."""
        edges = (
            np.array([(reference.node_ids[s], reference.node_ids[t]) for s, t in kept], dtype=reference.edges.dtype)
            if kept
            else reference.edges[:0]
        )
        return TrackGraph(node_ids=reference.node_ids, coordinates=reference.coordinates, edges=edges)

    @staticmethod
    def _consensus(reference: TrackGraph, voters: dict[tuple[int, int], set[str]], k: int) -> TrackGraph:
        """The reference nodes with the edges that at least `k` configs agreed on, back in node-id space."""
        kept = [pair for pair, names in voters.items() if len(names) >= k]
        return ConsensusRun._graph_from_pairs(reference, kept)

    @staticmethod
    def _subset_union_pairs(voters: dict[tuple[int, int], set[str]], subset: frozenset[str]) -> list[tuple[int, int]]:
        """Row-pairs any config in `subset` drew (vote>=1 within it) — the recall union of a shippable subset."""
        return [pair for pair, names in voters.items() if names & subset]

    @staticmethod
    def _net_new_over_champion(voters: dict[tuple[int, int], set[str]], k: int, champion: str) -> tuple[int, int]:
        """`(added, dropped)` high-confidence edges vote>=k carries that the shipped champion alone does not / does.

        GT-free recall-axis magnitude: `added` is edges at least `k` configs agree on that the champion never drew
        (the recall the single-threshold champion structurally misses); `dropped` is champion edges that fail to
        reach `k` votes (links no other config corroborates). A large `added` at k>=2 is the union lever the sparse
        proxy cannot see; a large `dropped` warns the vote is discarding the champion's own accepted links.
        """
        at_k = {pair for pair, names in voters.items() if len(names) >= k}
        champion_edges = {pair for pair, names in voters.items() if champion in names}
        added = len(at_k - champion_edges)
        dropped = len(champion_edges - at_k)
        return added, dropped

    def score(
        self, root: DataRoot
    ) -> tuple[dict[int, float], dict[str, float], dict[int, tuple[int, int]], list[SubsetScore]]:
        """Per vote threshold `k`: pooled consensus edge Jaccard, per-config baseline Jaccard, and the GT-free
        `(added, dropped)` edge counts vs the shipped champion pooled over movies. The 4th element ranks the
        champion-inclusive 2- and 3-config subsets (the only kernel-feasible ships) by their vote>=1 union
        recall over the champion — so one run answers both go/no-go AND which minimal subset to ship."""
        proxy, graphs = self._graphs(root)
        matcher = DistanceMatcher(spacing=proxy.spacing)
        names = list(graphs)
        stems = [path.stem for path in proxy.paths]
        truth_by_stem = {path.stem: truth.graph for path, truth in zip(proxy.paths, proxy.truths, strict=True)}

        def single_counts(name: str) -> list[EdgeCounts]:
            return [
                EdgeCounts.of(
                    graphs[name][stem], truth_by_stem[stem], matcher.match(graphs[name][stem], truth_by_stem[stem])
                )
                for stem in stems
            ]

        singles = {name: EdgeCounts.pooled(single_counts(name)).jaccard() for name in names}

        # The expensive per-config node matching is the same for every k, so vote once per movie and reuse.
        voted_by_stem: dict[str, tuple[TrackGraph, dict[tuple[int, int], set[str]]]] = {}
        for stem in stems:
            movie = {name: graphs[name][stem] for name in names}
            reference = movie[self._reference_name(movie)]
            voted_by_stem[stem] = (reference, self._vote(movie, reference, matcher))

        consensus: dict[int, float] = {}
        net_new: dict[int, tuple[int, int]] = {}
        for k in range(1, len(names) + 1):
            counts = []
            added = dropped = 0
            for stem in stems:
                reference, voters = voted_by_stem[stem]
                graph = self._consensus(reference, voters, k)
                truth = truth_by_stem[stem]
                counts.append(EdgeCounts.of(graph, truth, matcher.match(graph, truth)))
                movie_added, movie_dropped = self._net_new_over_champion(voters, k, _CHAMPION)
                added += movie_added
                dropped += movie_dropped
            consensus[k] = EdgeCounts.pooled(counts).jaccard()
            net_new[k] = (added, dropped)

        others = [name for name in names if name != _CHAMPION]
        subsets = [frozenset({_CHAMPION, *combo}) for size in (1, 2) for combo in combinations(others, size)]
        ranked = sorted(
            (self._score_subset(subset, voted_by_stem, stems, truth_by_stem, matcher) for subset in subsets),
            key=lambda scored: -scored.added,
        )
        return consensus, singles, net_new, ranked

    @staticmethod
    def _score_subset(
        subset: frozenset[str],
        voted_by_stem: dict[str, tuple[TrackGraph, dict[tuple[int, int], set[str]]]],
        stems: list[str],
        truth_by_stem: dict[str, TrackGraph],
        matcher: DistanceMatcher,
    ) -> SubsetScore:
        """A kernel-feasible ship: this subset's vote>=1 union edge Jaccard and the links it adds over the champion.

        `added` counts union edges no champion-only run drew (the recall a 2-3 config kernel buys over shipping the
        champion alone) — the GT-free lever the sparse proxy is blind to; `union_jaccard` is that union scored
        against the sparse GT (the LB-faithful axis, expected to hold near the champion, not to jump)."""
        counts: list[EdgeCounts] = []
        added = 0
        for stem in stems:
            reference, voters = voted_by_stem[stem]
            union = ConsensusRun._subset_union_pairs(voters, subset)
            graph = ConsensusRun._graph_from_pairs(reference, union)
            truth = truth_by_stem[stem]
            counts.append(EdgeCounts.of(graph, truth, matcher.match(graph, truth)))
            champion_edges = {pair for pair, names in voters.items() if _CHAMPION in names}
            added += len({pair for pair in union} - champion_edges)
        return SubsetScore(frozenset(subset), EdgeCounts.pooled(counts).jaccard(), added)


@dataclass(frozen=True)
class SubsetScore:
    """A shippable champion-inclusive config subset scored: union recall over the champion and its GT edge Jaccard."""

    subset: frozenset[str]
    union_jaccard: float
    added: int


def main() -> None:
    """Run the consensus vote and log its edge Jaccard per `k` against the best single config."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Vote-union the 0.900-tying tracker configs and score edge Jaccard.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cpu", help="cpu keeps the card free; detector responses are cached")
    parsed = parser.parse_args()
    root = DataRoot.from_config(parsed.config)
    consensus, singles, net_new, ranked = ConsensusRun(parsed.device).score(root)

    best_single = max(singles.values())
    best_name = max(singles, key=lambda name: singles[name])
    for name, jaccard in sorted(singles.items(), key=lambda item: -item[1]):
        logger.info("single  edgeJ=%.4f  %s", jaccard, name)
    for k, jaccard in sorted(consensus.items()):
        added, dropped = net_new[k]
        logger.info(
            "vote>=%d edgeJ=%.4f  delta_vs_best_single=%+.4f  new_vs_champion=%d  dropped_from_champion=%d",
            k,
            jaccard,
            jaccard - best_single,
            added,
            dropped,
        )
    best_k = max(consensus, key=lambda k: consensus[k])
    logger.info(
        "best single %s = %.4f | best consensus vote>=%d = %.4f | delta = %+.4f",
        best_name,
        best_single,
        best_k,
        consensus[best_k],
        consensus[best_k] - best_single,
    )
    logger.info(
        "GT-FREE RECALL AXIS: vote>=2 adds %d high-confidence links the champion lacks, drops %d of its own "
        "(the sparse proxy cannot see these — this is the union lever, not the edgeJ delta above)",
        *net_new[2],
    )
    logger.info("SHIPPABLE SUBSETS (champion + 1-2 configs, vote>=1 union — the kernel-feasible ships):")
    for scored in ranked:
        others = sorted(scored.subset - {_CHAMPION})
        logger.info(
            "  +%-4d union links over champion  unionJ=%.4f  = %s + [%s]",
            scored.added,
            scored.union_jaccard,
            _CHAMPION,
            ", ".join(others),
        )


if __name__ == "__main__":
    main()

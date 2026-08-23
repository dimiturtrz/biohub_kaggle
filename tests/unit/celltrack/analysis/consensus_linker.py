"""The consensus vote must count each config's edges once, in the reference's node space, and cut on `k`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from celltrack.analysis.consensus_linker import ConsensusRun
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher
from core.paths import DataRoot

# Four nodes: two cells (x=0 and x=10) across two timepoints. Columns are t, z, y, x.
_COORDS = np.array([[0, 0, 0, 0], [0, 0, 0, 10], [1, 0, 0, 0], [1, 0, 0, 10]])
_IDS = np.array([10, 11, 20, 21])  # arbitrary sparse ids: exercises the id<->row translation
_LINK_CELL0 = np.array([[10, 20]])  # x=0 cell linked across the two frames
_LINK_CELL1 = np.array([[11, 21]])  # x=10 cell linked across the two frames


def _graph(edges: np.ndarray) -> TrackGraph:
    return TrackGraph(node_ids=_IDS, coordinates=_COORDS, edges=edges)


def _both_cells() -> TrackGraph:
    return _graph(np.vstack([_LINK_CELL0, _LINK_CELL1]))


def test_vote_counts_each_config_once_in_reference_rows() -> None:
    matcher = DistanceMatcher(spacing=Spacing(1, 1, 1))
    reference = _both_cells()
    graphs = {
        "a": _both_cells(),  # links both cells
        "b": _graph(_LINK_CELL0),  # links only cell 0
        "c": _both_cells(),  # links both cells
    }
    voters = ConsensusRun._vote(graphs, reference, matcher)
    # Cell 0's link (row 0 -> row 2) is drawn by all three; cell 1's (row 1 -> row 3) by only a and c.
    assert voters[(0, 2)] == {"a", "b", "c"}
    assert voters[(1, 3)] == {"a", "c"}


def test_consensus_threshold_selects_by_vote_count() -> None:
    matcher = DistanceMatcher(spacing=Spacing(1, 1, 1))
    reference = _both_cells()
    graphs = {"a": _both_cells(), "b": _graph(_LINK_CELL0), "c": _both_cells()}
    voters = ConsensusRun._vote(graphs, reference, matcher)

    union = ConsensusRun._consensus(reference, voters, k=1)
    intersection = ConsensusRun._consensus(reference, voters, k=3)

    assert {tuple(edge) for edge in union.edges.tolist()} == {(10, 20), (11, 21)}  # k=1 keeps both
    assert {tuple(edge) for edge in intersection.edges.tolist()} == {(10, 20)}  # k=3 keeps only the unanimous link
    assert union.node_ids.tolist() == _IDS.tolist()  # nodes are the reference backbone, untouched


def test_consensus_empty_when_threshold_unreachable() -> None:
    matcher = DistanceMatcher(spacing=Spacing(1, 1, 1))
    reference = _both_cells()
    voters = ConsensusRun._vote({"a": _both_cells()}, reference, matcher)
    empty = ConsensusRun._consensus(reference, voters, k=2)  # one voter cannot reach two votes
    assert empty.edges.shape == (0, 2)


def test_net_new_over_champion_counts_added_and_dropped() -> None:
    matcher = DistanceMatcher(spacing=Spacing(1, 1, 1))
    reference = _both_cells()
    # Champion draws only cell 0's link; two challengers corroborate cell 1's link, none corroborate cell 0's.
    graphs = {
        "shipped_thr097": _graph(_LINK_CELL0),
        "challenger_a": _graph(_LINK_CELL1),
        "challenger_b": _graph(_LINK_CELL1),
    }
    voters = ConsensusRun._vote(graphs, reference, matcher)
    added, dropped = ConsensusRun._net_new_over_champion(voters, k=2, champion="shipped_thr097")
    assert added == 1  # cell 1's link (2 votes) is new vs the champion
    assert dropped == 1  # champion's cell 0 link has only its own vote, fails k=2


def test_reference_is_the_densest_config() -> None:
    sparse = TrackGraph(node_ids=_IDS[:2], coordinates=_COORDS[:2], edges=_IDS[:0].reshape(0, 2))
    dense = _both_cells()
    assert ConsensusRun._reference_name({"sparse": sparse, "dense": dense}) == "dense"


def test_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`score` orchestrates the per-`k` consensus, the champion-relative counts, and the subset ranking.

    The heavy per-config tracker run is stubbed out — only the vote/score orchestration is under test — with two
    configs over one movie whose GT links only cell 0, so every returned axis is a hand-checkable count.
    """
    truth = _graph(_LINK_CELL0)  # the sparse GT: only cell 0 genuinely links across the two frames
    proxy = SimpleNamespace(
        spacing=Spacing(1, 1, 1),
        paths=[tmp_path / "m.zarr"],
        truths=[SimpleNamespace(graph=truth)],
    )
    graphs = {"shipped_thr097": {"m": _graph(_LINK_CELL0)}, "flow_thr080": {"m": _both_cells()}}
    monkeypatch.setattr(ConsensusRun, "_graphs", lambda self, root: (proxy, graphs))

    consensus, singles, net_new, ranked = ConsensusRun("cpu").score(cast(DataRoot, tmp_path / "paths.yaml"))

    assert set(singles) == {"shipped_thr097", "flow_thr080"}  # a pooled baseline jaccard per config
    assert set(consensus) == {1, 2}  # vote>=1 (union) through vote>=2 (both configs)
    assert all(0.0 <= value <= 1.0 for value in consensus.values())
    assert net_new[2] == (0, 0)  # k=2 = intersection; it neither adds nor drops vs the champion here
    assert ranked[0].added == 1  # champion+flow union recovers cell 1's link the champion alone lacks

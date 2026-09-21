"""Equivalence classes of the oracle arms: a fork the annotation divides at, one it does not, and a blind cut."""

from collections.abc import Mapping, Sequence

from celltrack.eval.oracle_fork_arms import OracleArms


class _Keys:
    """The handful of `tracksdata.DEFAULT_ATTR_KEYS` names the arms read."""

    NODE_ID = "node_id"
    MATCHED_NODE_ID = "matched_node_id"
    EDGE_SOURCE = "source"
    EDGE_TARGET = "target"
    MATCHED_EDGE_MASK = "matched_mask"


class _FakeGraph:
    """A duck-typed stand-in for a matched tracksdata graph: columns in, columns out."""

    def __init__(
        self,
        nodes: Mapping[str, Sequence[float]],
        edges: Mapping[str, Sequence[float]],
        children: Mapping[int, Sequence[int]],
    ):
        self._nodes = nodes
        self._edges = edges
        self._children = children

    def node_ids(self) -> list[int]:
        return [int(node) for node in self._nodes["node_id"]]

    def node_attrs(self, attr_keys: list[str]) -> Mapping[str, Sequence[float]]:
        return {key: self._nodes[key] for key in attr_keys}

    def edge_attrs(self) -> Mapping[str, Sequence[float]]:
        return self._edges

    def successors(self, node: int) -> list[int]:
        return list(self._children.get(int(node), ()))

    def predecessors(self, node: int) -> list[int]:
        return [parent for parent, children in self._children.items() if int(node) in children]

    def out_degree(self, nodes: list[int]) -> list[int]:
        return [len(self._children.get(int(node), ())) for node in nodes]


def _arms(*, gt_divides: bool) -> OracleArms:
    predicted = _FakeGraph(
        nodes={"node_id": [10, 11, 12], "matched_node_id": [5, -1, 7]},
        edges={"source": [10, 10], "target": [11, 12], "edge_prob": [0.9, 0.2], "matched_mask": [0, 1]},
        children={10: [11, 12]},
    )
    truth = _FakeGraph(
        nodes={"node_id": [5, 6, 7]},
        edges={},
        children={5: [6, 7]} if gt_divides else {5: [7]},
    )
    return OracleArms(predicted=predicted, truth=truth, keys=_Keys())


def test_divisions():
    assert _arms(gt_divides=True).divisions() == [(5, [6, 7])]
    assert _arms(gt_divides=False).divisions() == []


def test_matched_of():
    assert _arms(gt_divides=True).matched_of() == {5: 10, 7: 12}


def test_predecessors_of():
    assert _arms(gt_divides=True).predecessors_of(12) == [10]
    assert _arms(gt_divides=True).predecessors_of(10) == []


def test_cull():
    assert _arms(gt_divides=True).cull(blind=False) == ()


def test_cull_keeps_the_branch_the_annotation_agrees_with():
    culls = _arms(gt_divides=False).cull(blind=False)

    assert len(culls) == 1
    assert culls[0].parent == 10
    assert culls[0].dropped == (11,)


def test_cull_blind_keeps_the_likeliest_branch_instead():
    culls = _arms(gt_divides=True).cull(blind=True)

    assert len(culls) == 1
    assert culls[0].dropped == (12,)

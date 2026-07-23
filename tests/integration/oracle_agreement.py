"""Our evaluator against the organizers', on frozen cases.

`tests/assets/metric_expected.json` holds counts produced by the organizers' `tracking_cellmot` — see
`tools/oracle/README.md` for the pinned commit and how to regenerate. Their package is the oracle for
this comparison, never a runtime dependency: the fixtures are committed so the check runs without it.
"""

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.divisions import DivisionScoring
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher


class GraphSpec(TypedDict):
    """One graph as stored in the case file: `(id, t, z, y, x)` rows and `(source, target)` links."""

    nodes: list[list[int]]
    edges: list[list[int]]


class Case(TypedDict):
    """A predicted graph and the annotation it is scored against."""

    name: str
    gt: GraphSpec
    pred: GraphSpec


class CaseFile(TypedDict):
    """The frozen case set, with the spacing and cutoff it was scored under."""

    spacing: list[float]
    max_distance: float
    cases: list[Case]


class ExpectedCounts(TypedDict):
    """The reference implementation's verdict on one case."""

    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int


ASSETS = Path(__file__).parents[1] / "assets"
CASES = cast(CaseFile, json.loads((ASSETS / "metric_cases.json").read_text(encoding="utf-8")))
EXPECTED = cast(dict[str, ExpectedCounts], json.loads((ASSETS / "metric_expected.json").read_text(encoding="utf-8")))


def graph_of(specification: GraphSpec) -> TrackGraph:
    nodes = np.array(specification["nodes"], dtype=np.int64).reshape(-1, 5)
    return TrackGraph(
        node_ids=nodes[:, 0],
        coordinates=nodes[:, 1:],
        edges=np.array(specification["edges"], dtype=np.int64).reshape(-1, 2),
    )


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
def test_counts_agree_with_the_reference_implementation(case: Case):
    spacing = Spacing(*CASES["spacing"])
    matcher = DistanceMatcher(spacing=spacing, max_distance_um=CASES["max_distance"])
    truth, prediction = graph_of(case["gt"]), graph_of(case["pred"])
    edges = EdgeCounts.of(prediction, truth, matcher.match(prediction, truth))
    divisions = DivisionScoring.of(prediction, truth, matcher).counts()
    expected = EXPECTED[case["name"]]
    assert (edges.tp, edges.fp, edges.fn) == (expected["edge_tp"], expected["edge_fp"], expected["edge_fn"])
    assert (divisions.tp, divisions.fp, divisions.fn) == (
        expected["division_tp"],
        expected["division_fp"],
        expected["division_fn"],
    )


def test_the_case_set_covers_divisions():
    """A pass means little if no case actually contains a division."""
    with_divisions = [
        name
        for name, counts in EXPECTED.items()
        if counts["division_tp"] + counts["division_fp"] + counts["division_fn"] > 0
    ]
    assert len(with_divisions) > len(EXPECTED) // 2

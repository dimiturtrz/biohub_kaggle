"""Unit tests for the deployed softmax denominator — the quantity is MASS, not source count."""

import numpy as np
import pytest
from jaxtyping import Float

from celltrack.eval.softmax_population import SoftmaxPopulation
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


class FakeAffinity:
    """An `EdgeAffinity` from an explicit per-gap matrix, rows the t sources, columns the t+1 targets."""

    def __init__(self, by_timepoint: dict[int, Float[np.ndarray, "s t"]]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        return self._by_timepoint.get(timepoint)


def _graph(rows: list[tuple[int, int]]) -> TrackGraph:
    """Nodes given as (t, x) at z=y=0 — one spatial axis is enough to place them for or outside a gate."""
    coordinates = np.array([[t, 0, 0, x] for t, x in rows], dtype=np.int64)
    return TrackGraph(
        node_ids=np.arange(len(rows), dtype=np.int64),
        coordinates=coordinates,
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_of():
    """A target's in-gate mass is the share of its column held by sources the assignment could have picked.

    Two sources at t=0, one 1 um from the target and one 50 um away; the far source holds 0.3 of the column
    and is outside a 10 um gate, so 0.7 of the deployed denominator sits on a parent the linker can choose.
    """
    detections = _graph([(0, 0), (0, 50), (1, 1)])
    affinity = FakeAffinity({0: np.array([[0.7], [0.3]])})

    population = SoftmaxPopulation.of(detections, affinity, _ISOTROPIC, gate_um=10.0)

    assert population.in_gate_mass.tolist() == [0.7]
    assert population.competitors.tolist() == [1.0]  # only the near source competes within the gate
    assert population.sources_per_gap == 2.0  # but BOTH are in the softmax the head is deployed under


def test_of_counts_every_source_when_all_are_reachable():
    """With no source out of gate the whole column is in-gate — the training and deployed denominators agree."""
    detections = _graph([(0, 0), (0, 2), (1, 1)])
    affinity = FakeAffinity({0: np.array([[0.6], [0.4]])})

    population = SoftmaxPopulation.of(detections, affinity, _ISOTROPIC, gate_um=10.0)

    assert population.in_gate_mass.tolist() == [1.0]
    assert population.competitors.tolist() == [2.0]


def test_of_skips_a_gap_with_no_affinity():
    """A gap the scorer did not score contributes nothing rather than a zero that would drag the mean down."""
    detections = _graph([(0, 0), (1, 1), (2, 2)])
    affinity = FakeAffinity({0: np.array([[1.0]])})

    population = SoftmaxPopulation.of(detections, affinity, _ISOTROPIC, gate_um=10.0)

    assert len(population.in_gate_mass) == 1  # the t=1 gap is absent, not counted as empty


def test_report(caplog: pytest.LogCaptureFixture):
    """The report states the deployed denominator beside the mass, since the gap between them is the finding."""
    detections = _graph([(0, 0), (0, 50), (1, 1)])
    affinity = FakeAffinity({0: np.array([[0.7], [0.3]])})

    with caplog.at_level("INFO"):
        SoftmaxPopulation.of(detections, affinity, _ISOTROPIC, gate_um=10.0).report("dense")

    assert "sources/gap 2" in caplog.text
    assert "0.7000" in caplog.text

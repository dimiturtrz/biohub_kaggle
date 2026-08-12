"""Unit tests for the motion-structure measurement — pinned against motion whose answer is known.

The value of this module is a single number (the MSD exponent) that a build decision rests on, so the tests
construct tracks whose regime is chosen in advance — a straight line must read 2, a rigid translation must read
as pure drift — rather than asserting whatever the code currently returns.
"""

import logging

import numpy as np
import pytest

from celltrack.analysis.motion_statistics import (
    GATE_UM,
    STATIONARY_UM,
    Chains,
    CoMovingFrame,
    FrameDrift,
    MeanSquaredDisplacement,
    MovieMotion,
    NeighbourCoupling,
    PersistenceByStep,
    PersistentRandomWalk,
    Steps,
)
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
_BALLISTIC, _DIFFUSIVE = 2.0, 1.0


def _graph(tracks: list[list[tuple[int, int, int, int]]]) -> TrackGraph:
    """A track graph from explicit `(t, z, y, x)` chains, one edge between consecutive entries of each."""
    coordinates: list[tuple[int, int, int, int]] = []
    edges: list[tuple[int, int]] = []
    for track in tracks:
        first = len(coordinates)
        coordinates.extend(track)
        edges.extend([(first + step, first + step + 1) for step in range(len(track) - 1)])
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def _straight(length: int, y_start: int = 0, step_voxels: int = 4) -> list[tuple[int, int, int, int]]:
    """A cell travelling in a dead-straight line along x at a constant rate."""
    return [(t, 0, y_start, t * step_voxels) for t in range(length)]


def _positions(graph: TrackGraph):
    """The graph's node positions in micrometres."""
    return _SPACING.to_micrometres(graph.positions())


def _steps_of(tracks: list[list[tuple[int, int, int, int]]]) -> Steps:
    """The consecutive displacements of these tracks, in micrometres."""
    graph = _graph(tracks)
    return Chains.of(graph).steps(_positions(graph))


def _rigid_body() -> tuple[TrackGraph, dict[int, int]]:
    """Three cells translating identically — the rigid-translation fixture the drift controls need."""
    graph = _graph([_straight(6, y_start=offset) for offset in (0, 40, 80)])
    positions = _positions(graph)
    chains = Chains.of(graph)
    by_row = {
        int(chain[index]): positions[chain[index + 1]] - positions[chain[index]]
        for chain in chains.rows
        for index in range(len(chain) - 1)
    }
    return graph, by_row


def test_chains_of():
    """A chain is a run of unambiguous single-parent single-child steps in CONSECUTIVE frames.

    A division would inject cytokinesis into the turn statistic, and a temporal gap would report a multi-frame
    jump as one step — both would corrupt the persistence measurement, so both must end a chain.
    """
    divided = TrackGraph(
        node_ids=np.arange(4, dtype=np.int64),
        coordinates=np.asarray([(0, 0, 0, 0), (1, 0, 0, 4), (2, 0, 4, 8), (2, 0, -4, 8)], dtype=np.int64),
        edges=np.asarray([(0, 1), (1, 2), (1, 3)], dtype=np.int64),
    )

    assert [len(chain) for chain in Chains.of(_graph([_straight(4)])).rows] == [4]
    assert [len(chain) for chain in Chains.of(divided).rows] == [2]  # stops AT the fork, follows neither daughter
    assert [len(chain) for chain in Chains.of(_graph([[(0, 0, 0, 0), (1, 0, 0, 4), (5, 0, 0, 8)]])).rows] == [2]


def test_steps():
    """Each chain of length n yields n-1 displacements, tagged with the chain they came from."""
    steps = _steps_of([_straight(4), _straight(3, y_start=40)])

    assert len(steps.vectors) == 3 + 2
    assert steps.chain_index.tolist() == [0, 0, 0, 1, 1]


def test_lengths():
    """A step of four x-voxels is 4 * 0.40625 = 1.625 um — the conversion, pinned end to end."""
    assert _steps_of([_straight(3)]).lengths() == pytest.approx([1.625, 1.625])


def test_stationary_fraction():
    """The floor is the localisation noise, and the corpus's MEDIAN step sits below it — deliberately pinned.

    A 4-voxel x step is 1.625 um, almost exactly the measured median annotated step (1.675 um), and it counts
    as standing still because a displacement that size is indistinguishable from two localisation errors. That
    is not a quirk of the threshold: it is why a single-frame velocity carries no usable direction, and why the
    reported stationary fractions run as high as 88%. A step at twice that size clears it.
    """
    assert 1.625 < STATIONARY_UM < 2 * 1.625  # the floor sits between one typical step and two
    assert _steps_of([_straight(6, step_voxels=4)]).stationary_fraction() == 1.0
    assert _steps_of([_straight(6, step_voxels=8)]).stationary_fraction() == 0.0
    assert _steps_of([[(t, 0, 0, 0) for t in range(6)]]).stationary_fraction() == 1.0


def test_turn_cosines():
    """Consecutive steps of a constant-velocity track are parallel; a back-and-forth cell reverses every step.

    The weaker-step column travels with the cosine because it is the pair's signal-to-noise handle — without it
    a noise-induced right angle and a real one are indistinguishable.
    """
    cosines, weaker = _steps_of([_straight(4)]).turn_cosines()
    reversing, _ = _steps_of([[(t, 0, 0, 4 * (t % 2)) for t in range(4)]]).turn_cosines()

    assert cosines == pytest.approx(1.0)
    assert weaker == pytest.approx(1.625)
    assert reversing == pytest.approx(-1.0)


def test_persistence_by_step_of():
    """Turn cosines are binned by the pair's WEAKER step, and an empty band reports NaN rather than zero.

    A band with no samples must not read as "no persistence" — that would be a measurement the data never made.
    """
    persistence = PersistenceByStep.of(np.asarray([1.0, -1.0, 0.5]), np.asarray([0.5, 0.5, 5.0]), edges=(1.0, 10.0))

    assert persistence.counts == (2, 1, 0)
    assert persistence.mean_cosine[0] == pytest.approx(0.0)  # +1 and -1 average out
    assert persistence.mean_cosine[1] == pytest.approx(0.5)
    assert np.isnan(persistence.mean_cosine[2])


def test_mean_squared_displacement_of():
    """A constant-velocity track has MSD proportional to lag squared, so the exponent must be exactly 2.

    This is the number the temporal-window decision rests on, so it is pinned against an analytic case rather
    than a golden value: any change to the fit that stops recovering 2 here has broken the instrument. The
    oscillating complement is what stops a fit that merely returns 2 from passing.
    """
    graph, oscillating = _graph([_straight(20)]), _graph([[(t, 0, 0, 4 * (t % 2)) for t in range(20)]])

    straight = MeanSquaredDisplacement.of(Chains.of(graph), _positions(graph), max_lag=10)
    reversing = MeanSquaredDisplacement.of(Chains.of(oscillating), _positions(oscillating), max_lag=10)

    assert straight.exponent == pytest.approx(_BALLISTIC, abs=1e-6)
    assert straight.msd_um2[0] == pytest.approx(1.625**2)
    assert reversing.exponent < _DIFFUSIVE


def test_frame_drift_of():
    """Cells moving as one body are all drift and no residual — the control that reads the coupling result.

    The measured movies sit at 61-92% drift, and that number is only interpretable if a rigid translation reads
    as 100%. The residual must vanish too, since removing the shared vector leaves nothing behind.
    """
    drift = FrameDrift.of(*_rigid_body())

    assert drift.share == pytest.approx(1.0)
    assert drift.drift_um == pytest.approx(1.625)
    assert drift.residual_um == pytest.approx(0.0, abs=1e-9)


def test_neighbour_coupling_of():
    """Permuting which cell owns which displacement leaves a pure drift untouched — the null the control encodes.

    This is why "observed equals shuffled" was read as global drift rather than local structure: under a rigid
    translation the two arms are identical BY CONSTRUCTION, so any gap between them is genuinely local.
    """
    graph, by_row = _rigid_body()

    coupling = NeighbourCoupling.of(graph, _positions(graph), by_row, seed=0)

    observed = [value for value in coupling.mean_cosine if not np.isnan(value)]
    shuffled = [value for value in coupling.shuffled_cosine if not np.isnan(value)]
    assert observed == pytest.approx(shuffled)
    assert all(value == pytest.approx(1.0) for value in observed)
    assert coupling.band_um[0] == (0.0, GATE_UM)


def test_movie_motion_of():
    """The whole report assembles off one graph, and a rigid translation reads ballistic with total drift."""
    graph, _ = _rigid_body()

    motion = MovieMotion.of("fixture", graph, _SPACING, max_lag=4, seed=0)

    assert motion.chains == 3
    assert motion.steps == 3 * 5
    assert motion.displacement.exponent == pytest.approx(_BALLISTIC, abs=1e-6)
    assert motion.drift.share == pytest.approx(1.0)


def test_report(caplog: pytest.LogCaptureFixture) -> None:
    """The report names the movie and its regime — the gating number must reach the log, not just the object."""
    graph, _ = _rigid_body()

    with caplog.at_level(logging.INFO):
        MovieMotion.of("fixture", graph, _SPACING, max_lag=4, seed=0).report()

    assert "fixture" in caplog.text
    assert "ballistic" in caplog.text
    assert "MSD exponent 2.00" in caplog.text


def test_persistent_random_walk_of():
    """Furth's parameters are recovered from a ballistic curve, and refused when there are too few points.

    A constant-velocity track has MSD = (v*t)^2, so the fit must return that speed with a correlation time far
    longer than the observation window — the ballistic limit of the model, where velocity never decorrelates.
    """
    graph = _graph([_straight(20)])
    displacement = MeanSquaredDisplacement.of(Chains.of(graph), _positions(graph), max_lag=12)

    walk = PersistentRandomWalk.of(displacement)

    assert walk.speed_um == pytest.approx(1.625, rel=0.2)
    assert walk.tau_frames > 12  # no decorrelation is visible inside the window
    assert np.isnan(PersistentRandomWalk.of(MeanSquaredDisplacement((1,), (1.0,), 2.0)).tau_frames)


def test_predict():
    """Furth's law spans both limits: ballistic at short lag, diffusive at long, plus a constant noise floor."""
    speed, tau = 2.0, 5.0

    short = PersistentRandomWalk.predict(np.array([0.01]), speed, tau, noise=0.0)
    long_lag = PersistentRandomWalk.predict(np.array([500.0]), speed, tau, noise=0.0)

    assert short[0] == pytest.approx((speed * 0.01) ** 2, rel=1e-3)  # v^2 t^2
    assert long_lag[0] == pytest.approx(2 * speed**2 * tau * 500.0, rel=1e-2)  # linear in t
    assert PersistentRandomWalk.predict(np.array([0.0]), speed, tau, noise=3.0)[0] == pytest.approx(3.0)


def test_co_moving_frame_of():
    """Removing a rigid translation leaves the cells where they started — the drift is integrated, not subtracted.

    The drift is a per-gap VELOCITY, so the correction has to accumulate: frame t moves by the sum of every
    earlier gap's drift. Subtracting the instantaneous vector instead would itself be a displacement and would
    inject the motion it is meant to remove.
    """
    graph, by_row = _rigid_body()
    positions = _positions(graph)

    corrected = CoMovingFrame.of(graph, positions, by_row).positions_um

    spread = [float(np.ptp(corrected[graph.timepoints() == t][:, 2])) for t in np.unique(graph.timepoints())]
    assert max(spread) == pytest.approx(0.0, abs=1e-9)  # every cell of a frame lands at one x, the drift gone

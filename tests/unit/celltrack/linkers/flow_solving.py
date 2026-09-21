"""The two flow engines must be interchangeable: same network in, same selected edge set out."""

import numpy as np
import pytest

from celltrack.linkers.flow_solving import MinCostFlowSolve, NetworkSimplexSolve, Transition

_COST_SCALE = 1000
_BOUNDARY = 3 * _COST_SCALE


def _random_shape(seed: int, per_frame: int, frames: int) -> tuple[int, list[Transition]]:
    """A layered candidate set with the tracker's measured density and a cost range that spans zero.

    Costs must go negative: a priced affinity makes `distance - bonus*P` negative exactly where linking is
    worth it, and a non-negative-only engine has to rewrite the network to accept them. An all-positive
    fixture would leave that rewrite untested, because zero flow would be optimal.
    """
    rng = np.random.default_rng(seed)
    count = per_frame * frames
    seen: set[tuple[int, int]] = set()
    transitions: list[Transition] = []
    for frame in range(frames - 1):
        for local in range(per_frame):
            source = frame * per_frame + local
            for target_local in rng.integers(0, per_frame, rng.integers(0, 4)):
                target = (frame + 1) * per_frame + int(target_local)
                if (source, target) in seen:
                    continue
                seen.add((source, target))
                transitions.append((source, target, round(rng.uniform(-15, 5) * _COST_SCALE)))
    return count, transitions


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_network_simplex_solve_selected(seed: int) -> None:
    """The reference engine selects each transition at most once, and selects what OR-tools selects."""
    count, transitions = _random_shape(seed, per_frame=12, frames=10)
    boundary = [_BOUNDARY] * count
    simplex = NetworkSimplexSolve().selected(count, boundary, boundary, transitions)
    assert len(simplex) == len(set(simplex))
    assert set(simplex) == set(MinCostFlowSolve().selected(count, boundary, boundary, transitions))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_min_cost_flow_solve_selected(seed: int) -> None:
    """The accelerated engine reproduces the reference edge set — the equality the swap rests on."""
    count, transitions = _random_shape(seed, per_frame=12, frames=10)
    boundary = [_BOUNDARY] * count
    ortools = MinCostFlowSolve().selected(count, boundary, boundary, transitions)
    assert len(ortools) == len(set(ortools))
    assert set(ortools) == set(NetworkSimplexSolve().selected(count, boundary, boundary, transitions))


def test_min_cost_flow_solve_selected_costs_the_same_objective() -> None:
    """Agreement is not an accident of a sparse fixture: the chosen sets price identically at a denser gate."""
    count, transitions = _random_shape(seed=7, per_frame=25, frames=12)
    price = {(source, target): cost for source, target, cost in transitions}
    boundary = [_BOUNDARY] * count
    simplex = NetworkSimplexSolve().selected(count, boundary, boundary, transitions)
    ortools = MinCostFlowSolve().selected(count, boundary, boundary, transitions)
    assert sum(price[link] for link in simplex) == sum(price[link] for link in ortools)


def test_min_cost_flow_solve_selected_on_an_empty_network() -> None:
    """A gap that admits no candidate links nothing rather than raising — the boundary case `_arrays` guards."""
    assert MinCostFlowSolve().selected(4, [_BOUNDARY] * 4, [_BOUNDARY] * 4, []) == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_division_surcharge_beyond_reach_reduces_to_the_one_to_one_network(seed: int) -> None:
    """A surcharge no transition can out-earn must leave the shipped edge set untouched — the wiring proof.

    The same reduction the linker's `boundary_cost = 0` has: an arm is only readable once its disabled limit is
    the thing it claims to extend. At a surcharge past the widest negative transition no second link can pay for
    itself, so the optimum is the 1-to-1 one, and any difference here is the arc being mis-priced, not a
    division being found.
    """
    count, transitions = _random_shape(seed, per_frame=12, frames=10)
    boundary = [_BOUNDARY] * count
    for engine in (NetworkSimplexSolve(), MinCostFlowSolve()):
        assert set(engine.selected(count, boundary, boundary, transitions, 99 * _COST_SCALE)) == set(
            engine.selected(count, boundary, boundary, transitions)
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_division_engines_agree_with_the_second_arc_open(seed: int) -> None:
    """The two engines stay interchangeable once the division arc exists — the shift argument holds."""
    count, transitions = _random_shape(seed, per_frame=12, frames=10)
    boundary = [_BOUNDARY] * count
    simplex = NetworkSimplexSolve().selected(count, boundary, boundary, transitions, _COST_SCALE)
    ortools = MinCostFlowSolve().selected(count, boundary, boundary, transitions, _COST_SCALE)
    price = {(source, target): cost for source, target, cost in transitions}
    assert sum(price[link] for link in simplex) == sum(price[link] for link in ortools)


def test_division_selects_two_children_from_one_parent() -> None:
    """A Y whose two branches each out-earn the surcharge is linked as a fork, which the 1-to-1 network cannot."""
    transitions: list[Transition] = [(0, 1, -10 * _COST_SCALE), (0, 2, -10 * _COST_SCALE)]
    boundary = [_BOUNDARY] * 3
    assert set(NetworkSimplexSolve().selected(3, boundary, boundary, transitions, _COST_SCALE)) == {(0, 1), (0, 2)}
    assert len(NetworkSimplexSolve().selected(3, boundary, boundary, transitions)) == 1


def test_division_does_not_undercut_a_track_start() -> None:
    """A parentless detection cannot buy its own appearance through the division arc — the surcharge closes it."""
    transitions: list[Transition] = [(0, 1, -10 * _COST_SCALE)]
    boundary = [_BOUNDARY] * 2
    assert set(MinCostFlowSolve().selected(2, boundary, boundary, transitions, 0)) == {(0, 1)}


def test_min_cost_flow_solve_or_network_simplex() -> None:
    """The chooser returns an engine that satisfies the same contract."""
    count, transitions = _random_shape(seed=11, per_frame=8, frames=6)
    boundary = [_BOUNDARY] * count
    chosen = MinCostFlowSolve.or_network_simplex().selected(count, boundary, boundary, transitions)
    assert set(chosen) == set(NetworkSimplexSolve().selected(count, boundary, boundary, transitions))

"""Equivalence classes of an annotated division's predicted shape: claimed, free, already right, missing."""

from celltrack.postproc.division_reparent import ForkCull, Reparent, ReparentPlan


def _plan(
    divisions: list[tuple[int, list[int]]],
    predicted_of: dict[int, int],
    edges: list[tuple[int, int]],
) -> ReparentPlan:
    def predecessors(row: int) -> list[int]:
        return [source for source, target in edges if target == row]

    return ReparentPlan.of(divisions, predicted_of, predecessors)


def test_of():
    plan = _plan([(1, [2, 3])], {1: 10, 2: 20, 3: 30}, [(11, 20), (10, 30)])

    assert plan.completed == 1
    assert plan.moves == (Reparent(parent=10, daughter=20, claimed_by=11),)
    assert plan.thefts() == 1


def test_thefts():
    plan = _plan([(1, [2, 3])], {1: 10, 2: 20, 3: 30}, [(10, 30)])

    assert plan.moves == (Reparent(parent=10, daughter=20, claimed_by=None),)
    assert plan.thefts() == 0


def test_of_asks_for_no_move_where_the_division_is_already_predicted():
    plan = _plan([(1, [2, 3])], {1: 10, 2: 20, 3: 30}, [(10, 20), (10, 30)])

    assert plan.completed == 1
    assert plan.moves == ()


def test_of_calls_a_division_unreachable_when_a_daughter_has_no_predicted_row():
    plan = _plan([(1, [2, 3])], {1: 10, 2: 20}, [(11, 20)])

    assert plan == type(plan)(moves=(), completed=0, unreachable=1)


def test_of_calls_a_division_unreachable_when_the_parent_is_unmatched():
    plan = _plan([(1, [2, 3])], {2: 20, 3: 30}, [(11, 20), (12, 30)])

    assert plan.unreachable == 1
    assert plan.moves == ()


def test_plan():
    culls = ForkCull.plan(
        [(10, [20, 30])],
        {10: 1},
        {2}.__contains__,
        lambda parent, child: float(child),
    )

    assert culls == (ForkCull(parent=10, dropped=(20,)),)


def test_plan_spares_a_fork_whose_annotation_divides_there_too():
    assert ForkCull.plan([(10, [20, 30])], {10: 1}, {1}.__contains__, lambda parent, child: float(child)) == ()


def test_plan_cuts_an_unannotated_fork_because_nothing_protects_it():
    culls = ForkCull.plan([(10, [20, 30])], {}, {1}.__contains__, lambda parent, child: float(child))

    assert culls[0].dropped == (20,)


def test_is_theft():
    assert Reparent(parent=10, daughter=20, claimed_by=11).is_theft()
    assert not Reparent(parent=10, daughter=20, claimed_by=None).is_theft()

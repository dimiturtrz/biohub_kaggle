import ast
from pathlib import Path

import pytest

from celltrack.operating_point import TrackerConfig

# The layers whose whole output is a NUMBER — a score, a checkpoint, a diagnosis. A bare `TrackerConfig()`
# here silently describes the 0.892 pipeline while the run reports on the 0.895 one we submit.
_MEASURING_LAYERS = (Path("celltrack/eval"), Path("celltrack/training"))


def test_post_init():
    """An operating point that cannot mean what it says is refused at construction, not obeyed quietly."""
    with pytest.raises(ValueError, match="edge_blend must sum to 1"):
        TrackerConfig(edge_blend=(0.8, 0.8))  # rescales the logits -> shifts the softmax temperature
    with pytest.raises(ValueError, match="threshold must be a probability"):
        TrackerConfig(threshold=5.0)
    with pytest.raises(ValueError, match="min_track_length"):
        TrackerConfig(min_track_length=0)

    assert TrackerConfig().threshold == 0.97  # the LB-measured best, not the round number


def test_shipped():
    """The submitted 0.895 recipe, pinned field by field — this is the contract four modules now mount.

    Pinned rather than smoke-tested because the whole point of the method is that the selector, the kernel and
    the diagnostics agree on ONE operating point. If a field here drifts, the thing that breaks is not this
    test but the comparability of every checkpoint we select and every number we quote, silently.
    """
    shipped = TrackerConfig.shipped()

    assert shipped.threshold == 0.97  # LB ladder 0.99/0.98/0.97 = 0.887/0.891/0.892
    assert (shipped.linker.name, shipped.linker.disappearance_cost) == ("flow", 3.0)  # global solver, its peak
    assert shipped.bidirectional_edges is True  # +0.004 on the LB under a global consumer, -0.0027 under a local one
    assert (shipped.linker.gate_um, shipped.linker.affinity_bonus) == (10.0, 20.0)
    assert (shipped.min_track_length, shipped.smooth_strength) == (6, 0.8)  # both LB-arbitrated, not proxy
    # Divisions are IN the submitted recipe (0.899 vs the 0.895 base) and the proxy cannot arbitrate the axis,
    # so the pin is the leaderboard's: symmetry ranking, no probability floors, and both the gates and the fork
    # budget DERIVED rather than set. Every one of these is a field default, so the recipe is the type itself.
    division = shipped.division
    assert division is not None
    assert division.ranking == "symmetry"  # beat probability and the frontier's geometry at identical gates
    assert (division.min_second_prob, division.min_kept_prob) == (0.0, 0.0)  # the floor excluded the true case
    assert (division.parent_gate_um, division.sister_gate_um) == (None, None)  # derived from the linker's gate
    assert division.max_added_forks is None  # ~1072, ~400 and the derived ~157 forks all scored 0.899


def test_no_measuring_module_constructs_a_bare_operating_point():
    """Nothing that reports a number may build `TrackerConfig()` — it must ask for the recipe it means.

    The class carries TWO operating points: the field defaults (per-frame assignment linker, no fusion — the
    0.892 tier) and `shipped()` (flow + fusion — the 0.895 recipe we submit). They are NOT interchangeable:
    re-ranking two saved checkpoints under both INVERTS which one wins, and the threshold ladder reverses
    order. So "which one a caller got" decides which pipeline an unlabelled measurement describes.

    Merging the two was considered and rejected — `LinkerConfig()` alone still defaults to `assignment`, so
    merging protects PARTIALLY while reading as total, and a default meaning "our current best" silently
    re-points every note written against it. Keeping both is right; reaching the neutral one by ACCIDENT is
    what must be impossible, and only these layers can corrupt a number by doing so.
    """
    offenders = [
        f"{source}:{node.lineno}"
        for layer in _MEASURING_LAYERS
        for source in layer.rglob("*.py")
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrackerConfig"
        and not (node.args or node.keywords)
    ]

    assert not offenders, f"measuring modules must mount TrackerConfig.shipped(): {offenders}"

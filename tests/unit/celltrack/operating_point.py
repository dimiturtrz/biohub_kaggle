import ast
from pathlib import Path

import pytest

from celltrack.edges.calibrated_fusion import CalibratedFusionOptions
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


def test_frontier_replica():
    """The genuine 0.926 stack as one operating point — every frontier arm ON, and `shipped` left untouched.

    Pinned like `shipped` because the kernel and any local eval mount THIS one definition: a drift here silently
    ships a different pipeline under the replica's name. The DeepCenter veto is the mounted `center_pack` and the
    secondary seed the mounted `secondary_pack`, both orthogonal to the config, so they are not asserted here.
    """
    replica = TrackerConfig.frontier_replica()

    assert replica.threshold == 0.96875  # between the shipped 0.97 and the high-precision 0.99
    # Calibrated dual-seed fusion: both heads flipped, exactly as proxy_eval._with_calibrated_fusion does.
    assert isinstance(replica.edge_options.calibrated_fusion, CalibratedFusionOptions)
    assert replica.detector_blend == pytest.approx(0.525)  # seed-1 share = 1 - the frontier's 0.475 secondary
    assert replica.detector_align_moments is True
    # divsub in its BYTE-FAITHFUL form: kernel_faithful turns on the kernel's own C1/C2/C3/two-cap stage
    # (the fidelity oracle proved the per-flag require_* bracket diverges — it forks off track-starts).
    division = replica.division
    assert division is not None
    assert division.kernel_faithful is True
    # Motion relink linker, tight-then-loose at the annotated one-frame displacement bounds.
    assert (replica.linker.name, replica.linker.tight_um, replica.linker.gate_um) == ("motion", 6.0, 10.0)

    assert TrackerConfig.shipped().linker.name == "flow"  # shipped is not perturbed by building the replica
    assert TrackerConfig.shipped().edge_options.calibrated_fusion is None


def test_kernel_faithful_replica():
    """The full byte-faithful replica: `frontier_replica` plus all six kernel-fidelity switches ON, in one home.

    The single incantation the deliverable names. `frontier_replica` already carries the faithful divsub; this
    also flips the fusion (E1-E4), the tracker's own admission+guard toggle (E5+E6) and the motion cost replica,
    so the whole stack reproduces the published kernel rather than our shipped approximations.
    """
    replica = TrackerConfig.kernel_faithful_replica()

    assert replica.kernel_faithful is True  # E5 (edge admission) + E6 (detection guard)
    assert replica.linker.kernel_faithful is True  # the motion relink cost replica
    fusion = replica.edge_options.calibrated_fusion
    assert fusion is not None
    assert fusion.kernel_faithful is True  # the calibrated fusion divergences (E1-E4)
    division = replica.division
    assert division is not None
    assert division.kernel_faithful is True  # the divsub recovery replica, inherited from frontier_replica

    # frontier_replica itself stays the genuine (non-strict) stack — building the faithful one does not perturb it.
    assert TrackerConfig.frontier_replica().kernel_faithful is False


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

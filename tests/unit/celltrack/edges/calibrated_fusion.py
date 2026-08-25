"""Unit tests for the margin-gated calibrated dual-seed edge fusion — the frontier's #1 edge lever.

`CalibratedFusion` is a pure, model-free function of two seeds' per-gap logit matrices, so every case here
runs on tiny hand-built tensors: the gate (low-margin AND consensus), the retention guard, and the two
calibration/ramp variants selected by `kernel_faithful`. No UNet, no head, no GPU.
"""

import pytest
import torch

from celltrack.edges.calibrated_fusion import CalibratedFusion, CalibratedFusionOptions
from celltrack.edges.seed_moment_alignment import SeedMomentAlignment


def test_calibrated_fusion_fuse():
    """Where the primary is already decided (wide top-2 margin), the fusion returns exactly the primary softmax."""
    primary = torch.tensor([[5.0, 0.0], [0.0, 5.0]])  # each target has a clear winner
    secondary = torch.tensor([[0.0, 3.0], [3.0, 0.0]])  # the other seed disagrees, but the gate never opens
    fused = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    assert torch.allclose(fused, torch.softmax(primary, dim=0))


def test_calibrated_fusion_blends_only_the_low_margin_consensus_column():
    """A target the primary is unsure about, where BOTH seeds pick the same parent, moves toward the blend."""
    primary = torch.tensor([[5.0, 0.1], [0.0, 0.0]])  # target 0 decided, target 1 a near-tie for source 0
    secondary = torch.tensor([[9.0, 9.0], [0.0, 0.0]])  # both seeds rank source 0 first on target 1
    fused = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    baseline = torch.softmax(primary, dim=0)
    assert torch.allclose(fused[:, 0], baseline[:, 0])  # decided target untouched
    assert not torch.allclose(fused[:, 1], baseline[:, 1])  # unsure + agreed target blended
    assert fused[0, 1] > baseline[0, 1]  # the agreed parent gets more mass


def test_calibrated_fusion_skips_when_seeds_disagree_on_the_parent():
    """Low margin is not enough — if the secondary picks a DIFFERENT parent, the primary is left untouched."""
    primary = torch.tensor([[0.1, 0.0], [0.0, 0.0]])  # target 0 a near-tie, primary favours source 0
    secondary = torch.tensor([[0.0, 0.0], [9.0, 0.0]])  # secondary favours source 1 on target 0 — no consensus
    fused = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    assert torch.allclose(fused, torch.softmax(primary, dim=0))


def test_calibrated_fusion_single_source_has_no_margin_to_gate_on():
    """One candidate source means no ambiguity, so the gate stays shut and the column is the primary's."""
    primary = torch.tensor([[0.1, 0.2]])  # one source, two targets
    secondary = torch.tensor([[9.0, 9.0]])
    fused = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    assert torch.allclose(fused, torch.softmax(primary, dim=0))  # softmax over one source is all ones


def test_calibrated_fusion_retention_guard_counts_candidates():
    """The guard keeps the blend only if it retains enough candidates (edges above the threshold) versus the primary."""
    guard = CalibratedFusion(CalibratedFusionOptions(cand_thresh=0.5, retention_frac=0.9))
    primary = torch.tensor([[0.9, 0.9], [0.1, 0.1]])  # two candidates above 0.5
    kept = torch.tensor([[0.9, 0.6], [0.1, 0.4]])  # two candidates — comfortably retained
    collapsed = torch.tensor([[0.9, 0.4], [0.1, 0.3]])  # col 1 none above 0.5, below 0.9 of primary's two
    assert guard._retained(primary, kept)
    assert not guard._retained(primary, collapsed)


def test_calibrated_fusion_reverts_the_whole_frame_when_the_guard_trips():
    """An unreachable retention bar makes the fusion fall back to the primary for the entire gap."""
    primary = torch.tensor([[5.0, 0.1], [0.0, 0.0]])  # target 1 is a gated near-tie the blend would move
    secondary = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
    reverting = CalibratedFusionOptions(retention_frac=100.0)  # no blend can ever clear this
    fused = CalibratedFusion(reverting).fuse(primary, secondary)
    assert torch.allclose(fused, torch.softmax(primary, dim=0))


def test_kernel_faithful_off_by_default():
    """The published-kernel edge math is opt-in: the fusion options default to our variant."""
    assert CalibratedFusionOptions().kernel_faithful is False


def test_kernel_faithful_off_is_byte_identical_to_the_variant():
    """With the flag off the fused matrix is exactly the variant's — the calibration and gate are unchanged."""
    primary = torch.tensor([[5.0, 0.1], [0.0, 0.0]])
    secondary = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
    explicit_off = CalibratedFusion(CalibratedFusionOptions(kernel_faithful=False)).fuse(primary, secondary)
    default = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    assert torch.equal(explicit_off, default)


def test_kernel_faithful_ramp_is_continuous():
    """E4: the blend weight ramps ~0 as the margin approaches the threshold and reaches blend_w at margin 0."""
    # Column 0's two source logits give a softmax top-2 margin of ~0.349 (just under the 0.35 threshold);
    # column 1 is a dead tie, margin 0. The secondary ranks the same parent first in both, so the gate is open.
    primary = torch.tensor([[0.728, 0.0], [0.0, 0.0]])
    secondary = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
    fusion = CalibratedFusion(CalibratedFusionOptions(kernel_faithful=True))
    aligned = SeedMomentAlignment.align_per_column(secondary, primary)
    weight = fusion._faithful_blend_weight(primary, torch.softmax(primary, dim=0), aligned)
    assert weight.shape == (1, 2)
    assert weight[0, 0] < 0.01  # margin ~0.349 -> ramp is ~zero, not the full 0.15
    assert weight[0, 1] == pytest.approx(0.15, abs=1e-6)  # margin 0 -> the full blend_w


def test_kernel_faithful_ramp_zeroed_when_seeds_disagree_on_the_parent():
    """Even at margin 0 the continuous ramp contributes nothing where the two heads pick different parents."""
    primary = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # every column a dead tie -> max uncertainty
    secondary = torch.tensor([[0.0, 0.0], [9.0, 9.0]])  # secondary ranks source 1 first everywhere
    fusion = CalibratedFusion(CalibratedFusionOptions(kernel_faithful=True))
    aligned = SeedMomentAlignment.align_per_column(secondary, primary)
    weight = fusion._faithful_blend_weight(primary, torch.softmax(primary, dim=0), aligned)
    # primary argmax is source 0, secondary argmax is source 1 -> disagreement -> weight zeroed despite margin 0
    assert torch.count_nonzero(weight) == 0


def test_kernel_faithful_calibration_is_per_column():
    """E3: per-target-column calibration differs from the scalar-global one on a non-uniform logit matrix."""
    primary = torch.tensor([[3.0, 0.0], [1.0, 0.0]])  # columns with different means and spreads
    secondary = torch.tensor([[2.0, 5.0], [0.0, 1.0]])
    per_column = SeedMomentAlignment.align_per_column(secondary, primary)
    scalar_global = SeedMomentAlignment.align(secondary, primary)
    assert not torch.allclose(per_column, scalar_global)


def test_kernel_faithful_fuse_differs_from_the_variant():
    """Flipping the flag changes the fused matrix — the continuous ramp and per-column calibration bite."""
    primary = torch.tensor([[5.0, 0.1], [0.0, 0.0]])  # target 1 a low-margin consensus column
    secondary = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
    variant = CalibratedFusion(CalibratedFusionOptions()).fuse(primary, secondary)
    faithful = CalibratedFusion(CalibratedFusionOptions(kernel_faithful=True)).fuse(primary, secondary)
    assert faithful.shape == variant.shape
    assert torch.isfinite(faithful).all()
    assert not torch.allclose(faithful, variant)
    assert torch.allclose(faithful.sum(dim=0), torch.ones(2))  # still a distribution over sources

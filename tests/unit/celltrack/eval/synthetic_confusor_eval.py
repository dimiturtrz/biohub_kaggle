"""Unit tests for the synthetic inverted-confusor ranking harness — the pure scoring logic.

`SyntheticConfusorEval.score` is a GPU integration concern (needs a checkpoint and a CUDA device) and carries
a test-mirror escape at its definition; these cover the population statistics, the all-vs-hard split, the
geometry rule that isolates the confusor axis, and that `build` generates a non-empty scene set.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float
from torch import Tensor

from celltrack.data.joint_dataset import PairTarget
from celltrack.eval.synthetic_confusor_eval import ConfusorScore, SyntheticConfusorEval, _RankPool


class _NoFrames:
    """A FrameSource stand-in for the pure `_pool_columns` tests, which never read a frame."""

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        raise AssertionError("_pool_columns must not read frames")


def test_of() -> None:
    empty = ConfusorScore.of(np.array([]), np.array([]), np.array([]))
    assert empty.n_cases == 0
    assert np.isnan(empty.top1)
    assert np.isnan(empty.inverted_fraction)
    # rank 0 twice, rank 1 once -> top1 2/3; only the middle case has P_chosen > P_true -> inverted 1/3.
    score = ConfusorScore.of(
        np.array([0, 1, 0]),
        np.array([0.9, 0.2, 0.8]),
        np.array([0.9, 0.5, 0.8]),
    )
    assert score.n_cases == 3
    assert score.top1 == pytest.approx(2 / 3)
    assert score.inverted_fraction == pytest.approx(1 / 3)
    assert score.mean_p_true == pytest.approx((0.9 + 0.2 + 0.8) / 3)


def test_add() -> None:
    pool = _RankPool()
    pool.add(np.array([0.7, 0.3]), true_source=0, is_hard=False)  # true IS argmax -> rank 0, not inverted
    pool.add(np.array([0.2, 0.8]), true_source=0, is_hard=True)  # rival outranks parent -> inverted, hard
    result = pool.result()
    assert result.every.n_cases == 2
    assert result.every.top1 == pytest.approx(0.5)
    assert result.every.inverted_fraction == pytest.approx(0.5)
    assert result.hard.n_cases == 1  # only the flagged case reaches the hard pool
    assert result.hard.top1 == pytest.approx(0.0)


def test_result() -> None:
    pool = _RankPool()
    pool.add(np.array([0.9, 0.1]), true_source=0, is_hard=False)  # easy, correct
    pool.add(np.array([0.1, 0.9]), true_source=0, is_hard=True)  # confusor, wrong
    result = pool.result()
    assert result.every.n_cases == 2
    assert result.every.top1 == pytest.approx(0.5)
    assert result.hard.n_cases == 1
    assert result.hard.top1 == pytest.approx(0.0)  # the hard subset diverges from the pooled read


def _target(source_centres: np.ndarray, target_centres: np.ndarray, edge_matrix: np.ndarray) -> PairTarget:
    return PairTarget(
        frames=_NoFrames(),  # not read by _pool_columns
        timepoint=0,
        source_centres=source_centres,
        target_centres=target_centres,
        edge_matrix=edge_matrix,
    )


def test_pool_columns_hard_flag_from_geometry() -> None:
    # Two sources at z=0 and z=10. Target 0 sits at z=9 (nearest source is 1, but its true parent is 0 ->
    # HARD). Target 1 sits at z=1 (nearest source is 0, true parent is 0 -> easy).
    sources = np.array([[0, 0, 0], [0, 0, 10]])
    targets = np.array([[0, 0, 9], [0, 0, 1]])
    edge_matrix = np.array([[1.0, 1.0], [0.0, 0.0]])  # both targets' true parent is source 0
    pool = _RankPool()
    posteriors = np.array([[0.5, 0.5], [0.5, 0.5]])
    SyntheticConfusorEval._pool_columns(pool, posteriors, _target(sources, targets, edge_matrix))
    result = pool.result()
    assert result.every.n_cases == 2
    assert result.hard.n_cases == 1  # only the target whose nearest source is not its parent


def test_pool_columns_without_gt_parent_skipped() -> None:
    sources = np.array([[0, 0, 0]])
    targets = np.array([[0, 0, 1], [0, 0, 2]])
    edge_matrix = np.array([[1.0, 0.0]])  # second target has no GT parent
    pool = _RankPool()
    SyntheticConfusorEval._pool_columns(pool, np.array([[0.9, 0.9]]), _target(sources, targets, edge_matrix))
    assert pool.result().every.n_cases == 1


def test_build() -> None:
    # A tiny generation: the faithful-inverted scene set is non-empty and the velocity feed matches the flag.
    harness = SyntheticConfusorEval.build(2, 0.45, seed=1234, faithful=False, prior_velocity=False)
    assert len(harness.targets) > 0
    assert all(target.edge_matrix.shape[1] > 0 for target in harness.targets)
    assert harness.prior_velocity is False
    assert harness.velocity.enabled is False

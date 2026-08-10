from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer
from celltrack.training import edge_finetune
from celltrack.training.edge_finetune import (
    EdgeFinetuneConfig,
    EdgeHardNegativeFinetuner,
    GapSupervision,
)
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph whose node ids are its row indices, so edge ids and rows coincide for the fixture."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


class _StubScorer:
    """A scorer whose UNet is frozen and whose transformer is one weight, so a step provably moves that weight."""

    def __init__(self) -> None:
        self.detector = torch.nn.Linear(1, 1)
        self.transformer = torch.nn.Linear(1, 1)

    def _gap_logits(self, gap: EdgeGap) -> torch.Tensor:
        pairs = torch.ones(len(gap.src_positions), len(gap.tgt_positions))
        return self.transformer.weight.reshape(()) * pairs  # (s, t) logits carrying the transformer's grad


def test_finetune(monkeypatch: pytest.MonkeyPatch):
    """One optimisation over a supervised gap moves the transformer weight (and leaves the frozen UNet untouched)."""
    monkeypatch.setattr(edge_finetune.TemporalUNetDetector, "_open_source", staticmethod(lambda path: "src"))
    gap = GapSupervision(
        timepoint=0,
        source_positions=np.zeros((2, 3), dtype=np.float32),
        target_positions=np.ones((2, 3), dtype=np.float32),
        gt_matrix=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),  # source 0 -> target 0 is the true edge
        hard_negatives=np.array([[0, 1]], dtype=np.int64),  # source 0 -> target 1 is a mined hard negative
    )
    scorer = _StubScorer()
    before = scorer.transformer.weight.detach().clone()
    detector_frozen_before = scorer.detector.weight.requires_grad

    typed_scorer = cast(EdgeTransformerScorer, scorer)
    EdgeHardNegativeFinetuner(typed_scorer, EdgeFinetuneConfig(device="cpu", steps=3)).finetune([(Path("video"), gap)])

    assert not torch.equal(scorer.transformer.weight, before)  # the head learned
    assert detector_frozen_before and not scorer.detector.weight.requires_grad  # the UNet was frozen


def test_of():
    """`of` builds one supervised gap per frame: the GT edge matrix by matching, plus its mined hard negatives."""
    # Two cells per frame, far apart so the identity matching is unambiguous; only node 0 -> node 2 is annotated.
    graph = _graph(
        [[0, 0, 0, 0], [0, 0, 0, 10], [1, 0, 0, 0], [1, 0, 0, 10]],
        [[0, 2]],
    )
    matcher = DistanceMatcher(spacing=Spacing(1.0, 1.0, 1.0))

    gaps = GapSupervision.of(graph, graph, matcher, hard_negatives=1)

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.timepoint == 0
    assert gap.gt_matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]  # only source 0 -> target 0 is an annotated edge
    assert gap.hard_negatives.tolist() == [[0, 1]]  # source 0's nearest wrong target is the far decoy at local col 1


def test_gt_matrix():
    """A cell of the matrix is 1 exactly where both detections match GT nodes the annotation links."""
    source_gt = np.array([10, UNMATCHED], dtype=np.int64)  # source 0 matches GT node 10; source 1 is unmatched
    target_gt = np.array([20, 21], dtype=np.int64)
    gt_edge_codes = {10 * 100 + 20}  # GT links node 10 -> node 20

    matrix = GapSupervision._gt_matrix(source_gt, target_gt, gt_edge_codes, count=100)

    assert matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]  # only (source 0 -> target 0) is an annotated edge


def test_mine_hard_negatives():
    """Each source with a true successor contributes its nearest *wrong* targets, closest first, up to the budget."""
    gt_matrix = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # source 0's true successor is target 1
    source_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # target 0 is nearest, target 2 next, target 1 (the true one) is far and must be excluded as a negative.
    target_positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0], [0.0, 0.0, 2.0]], dtype=np.float32)

    mined = GapSupervision._mine_hard_negatives(gt_matrix, source_positions, target_positions, keep=2)

    assert mined.tolist() == [[0, 0], [0, 2]]  # the two nearest wrong targets, nearest first; never the true one


def test_mine_hard_negatives_ignores_a_source_without_a_true_successor():
    """A source the annotation does not continue supplies no hard negatives — there is no positive to contrast."""
    gt_matrix = np.zeros((1, 3), dtype=np.float32)
    positions = np.zeros((1, 3), dtype=np.float32)
    targets = np.ones((3, 3), dtype=np.float32)

    assert GapSupervision._mine_hard_negatives(gt_matrix, positions, targets, keep=2).shape == (0, 2)

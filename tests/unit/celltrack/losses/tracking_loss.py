"""Unit tests for the joint loss assembly — the per-pair terms, their padded-batch vectorisation, `_PairOutcome`.

The numeric correctness of each term lives beside it in this package; here the loss is exercised through its own
DECOUPLED surface — the `PairPrediction`/`PairGroundTruth`/`BatchedPrediction` input structs it defines, filled
from a real stubbed forward (autouse backbone/transformer in `conftest`) so a real forward feeds a real loss
without the GPU. The loss imports nothing of the model or the dataset; the tiny adapters below stand in for the
trainer's own repacking at the call site.
"""

import torch

from celltrack.losses.tracking_loss import (
    BatchedPrediction,
    PairGroundTruth,
    PairPrediction,
    TrackingLoss,
    _PairOutcome,
)
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import BatchedForward, JointForward, JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.training.contrastive_term import ContrastiveTerm, ContrastiveTermConfig
from celltrack.training.joint_config import DataCfg, JointTrainConfig, ModelCfg, RuntimeCfg


def _config() -> JointTrainConfig:
    """A tiny CPU config with no downsample — the grid the loss reads the centres on is the raw voxel grid."""
    return JointTrainConfig(
        model=ModelCfg(out_channels=2, layers=(2, 4)),
        data=DataCfg(downsample=(1, 1, 1)),
        runtime=RuntimeCfg(device="cpu"),
    )


def _loss() -> TrackingLoss:
    config = _config()
    contrastive = ContrastiveTerm(ContrastiveTermConfig.from_loss(config.loss, config.model.out_channels))
    return TrackingLoss(config.loss.to_tracking_loss_config(config.data.downsample), contrastive)


def _model() -> JointModel:
    out_channels = 2
    detector = TemporalUNetDetector(out_channels, (2, 4))  # stubbed backbone via conftest
    transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=out_channels + 4 * _POS_EMBED_DIM, hidden_dim=8, n_heads=1, n_blocks=1
    )
    return JointModel(detector, transformer, downsample=(1, 1, 1))


def _prediction(out: JointForward) -> PairPrediction:
    return PairPrediction(out.edge_logits, out.detection_t, out.detection_t1, out.source_features, out.target_features)


def _batched_prediction(batched: BatchedForward) -> BatchedPrediction:
    return BatchedPrediction(
        batched.logits,
        batched.detect_t,
        batched.detect_t1,
        batched.source_mask,
        batched.target_mask,
        [_prediction(forward) for forward in batched.unpad()],
    )


def _ground_truth(sources: torch.Tensor, targets: torch.Tensor, edge_matrix: torch.Tensor) -> PairGroundTruth:
    return PairGroundTruth(sources, targets, edge_matrix, frame_t=torch.zeros(4, 8, 8))


def _outcome(edge_logits: torch.Tensor, edge_matrix: torch.Tensor) -> _PairOutcome:
    """A pair outcome whose loss terms are placeholders — these tests are about the edge tensors beside them."""
    return _PairOutcome(
        total=torch.tensor(3.0, requires_grad=True),
        edge=torch.tensor(1.0),
        detection=torch.tensor(2.0),
        contrastive=torch.tensor(0.5),
        hard_negative=torch.tensor(0.25),
        edge_logits=edge_logits,
        edge_matrix=edge_matrix,
    )


def test_logged():
    """Every term reaches the run's metric table under its own name, detached from the graph."""
    outcome = _outcome(torch.zeros(1, 1), torch.zeros(1, 1))
    assert outcome.logged() == {
        "train_loss": 3.0,
        "edge_loss": 1.0,
        "det_loss": 2.0,
        "contrastive_loss": 0.5,
        "hard_negative_loss": 0.25,
    }


def test_difficulty():
    """Hand-computed: of the two ANNOTATED sources one is ranked top-1 and one is not, so the defect is 1 - 1/2."""
    edge_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    edge_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 5.0], [9.0, 0.0, 0.0]])  # row 2 is unannotated, ignored
    assert _outcome(edge_logits, edge_matrix).difficulty() == 0.5


def test_difficulty_is_none_without_annotated_sources():
    """A pair whose annotation links nothing has no defect to report, so the sampler is told to keep what it stored."""
    torch.manual_seed(0)
    assert _outcome(torch.rand(2, 2), torch.zeros(2, 2)).difficulty() is None


def test_per_pair():
    """One pair's forward yields a finite `_PairOutcome` whose total is differentiable — every term is real."""
    model = _model().eval()
    sources, targets = torch.tensor([[2, 4, 4]]), torch.tensor([[2, 4, 4], [1, 2, 2]])
    out = model.forward(torch.zeros(4, 8, 8), torch.zeros(4, 8, 8), sources, targets)

    outcome = _loss().per_pair(_prediction(out), _ground_truth(sources, targets, torch.tensor([[1.0, 0.0]])))

    assert isinstance(outcome, _PairOutcome)
    assert torch.isfinite(outcome.total) and outcome.total.requires_grad
    assert outcome.reliability is None  # off by default


def test_batched():
    """The padded batch yields a (B,) total for backward and one `_PairOutcome` per pair for the diagnostics."""
    sources = [torch.tensor([[2, 4, 4]]), torch.tensor([[1, 2, 2]])]
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]]), torch.tensor([[3, 5, 5]])]
    edges = [torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]])]
    model = _model().eval()
    windows = torch.stack([torch.stack([torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)], dim=0) for _ in sources], dim=0)
    batched = model.batched_forward(windows, sources, targets)
    grounds = [_ground_truth(s, t, e) for s, t, e in zip(sources, targets, edges, strict=True)]

    total, outcomes = _loss().batched(_batched_prediction(batched), grounds)

    assert total.shape == (2,) and torch.isfinite(total).all()
    assert len(outcomes) == 2 and all(isinstance(o, _PairOutcome) for o in outcomes)


def test_reliability_weights():
    """Reliabilities normalise to batch-mean 1; a single missing one falls the whole batch back to ones."""
    zero = torch.zeros(())

    def _with(reliability: torch.Tensor | None) -> _PairOutcome:
        return _PairOutcome(
            torch.tensor(1.0), zero, zero, zero, zero, torch.zeros(1, 1), torch.zeros(1, 1), reliability=reliability
        )

    weighted = _loss().reliability_weights([_with(torch.tensor(2.0)), _with(torch.tensor(4.0))])
    assert torch.allclose(torch.stack(weighted), torch.tensor([2 / 3, 4 / 3]), atol=1e-5)  # mean-1 normalised

    off = _loss().reliability_weights([_with(torch.tensor(2.0)), _with(None)])
    assert all(weight.item() == 1.0 for weight in off)  # any missing -> unweighted

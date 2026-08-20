"""The joint run's loss assembly — every training term for a pair, and the padded-batch vectorisation of them.

The trainer owns the STEP (forward, backward, optimiser); `TrackingLoss` owns the OBJECTIVE. It computes the four
decoupled terms — a detection `BalancedBCE` per frame, a `SoftmaxFocalBCE` over the links, an optional InfoNCE
over the node features, an optional hard-negative margin — for one pair (`per_pair`) or, for the pure-default
config, vectorised over the whole padded batch with no per-pair Python loop (`batched`).

It reaches for nothing above the loss layer: the model's forward and the dataset's sample arrive REPACKAGED into
the loss's own input structs (`PairPrediction`/`PairGroundTruth`/`BatchedPrediction`), the objective's knobs arrive as
its own `TrackingLossConfig`, and the contrastive term arrives as a bare `ContrastiveObjective` callable — so this
module imports only its sibling loss primitives and lives beside them, not up in `training/`. `_PairOutcome` is
the result type both entry points return; it lives here because it is what the loss produces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.losses.association_objective import AssociationObjective
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.sample_weight import SampleWeight

# The (1,4,4) downsample makes the heads' grid isotropic at the z-spacing, so one downsampled voxel is this many
# micrometres on every axis — the unit the reliability weight's crowd gate is measured in (see downsample memory).
_ISOTROPIC_UM = 1.625


@dataclass(frozen=True)
class TrackingLossConfig:
    """The objective's own knobs — every value the terms read, and nothing about the model or the run.

    Filled from the trainer's `LossCfg` (`LossCfg.to_tracking_loss_config`), so the loss never sees the whole
    training config: it carries the term weights, the two masking thresholds, the head grid, and the linker's
    admission gate the reliability weight measures crowd in. The link normalisation lives in the injected
    `AssociationObjective`, not here — that strategy owns the axes, the balancing, and any slack row.
    """

    neg_weight: float
    ignore_ambiguous_above: float | None
    det_weight: float
    contrastive_weight: float
    hard_negative_weight: float
    hard_negatives: int
    reliability_weighting: bool
    downsample: tuple[int, int, int]
    gate_um: float


@dataclass(frozen=True)
class PairPrediction:
    """One pair's model outputs as the loss consumes them — the loss's own view, not the model's forward type."""

    edge_logits: Float[Tensor, "s u"]
    detection_t: Tensor
    detection_t1: Tensor
    source_features: Float[Tensor, "s d"]
    target_features: Float[Tensor, "u d"]


@dataclass(frozen=True)
class PairGroundTruth:
    """One pair's ground truth as the loss consumes it — the loss's own view, not the dataset's sample type."""

    source_centres: Int[Tensor, "s 3"]
    target_centres: Int[Tensor, "u 3"]
    edge_matrix: Float[Tensor, "s u"]
    frame_t: Tensor


@dataclass(frozen=True)
class BatchedPrediction:
    """The padded-batch forward as the loss consumes it — the `(B, ...)` blocks, plus the per-pair views behind them.

    `per_pair` carries each pair's unpadded prediction for the detached diagnostics (the difficulty read and the
    ignored-fraction), which the vectorised terms cannot recover from the padded block alone.
    """

    logits: Float[Tensor, "b max max"]
    detect_t: Tensor
    detect_t1: Tensor
    source_mask: Tensor
    target_mask: Tensor
    per_pair: list[PairPrediction]


# The contrastive term as the loss needs it — features in, one scalar out, nothing about where it acts. The
# trainer supplies the implementation (it owns the projection head, a model weight the loss layer may not hold);
# the loss only ever calls it. Injecting this callable is what keeps the objective in the loss layer.
ContrastiveObjective = Callable[[Float[Tensor, "s d"], Float[Tensor, "u d"], Float[Tensor, "s u"]], Float[Tensor, ""]]


@dataclass(frozen=True)
class _PairOutcome:
    """One pair's forward — the loss total that is stepped, each term on its own scale, and the edge tensors.

    The edge logits and the annotated matrix ride along because the difficulty estimate the sampler feeds on is
    read off exactly them, and they are free here: the training step already computed both.
    """

    total: Tensor
    edge: Tensor
    detection: Tensor
    contrastive: Tensor
    hard_negative: Tensor
    edge_logits: Tensor
    edge_matrix: Tensor
    ignored_fraction: float | None = None
    reliability: Tensor | None = None  # the pair's raw SNR/(1+crowd) weight, batch-normalised by the step

    def logged(self) -> dict[str, float]:
        """The detached per-term floats under the names the run tracks — a summed number hides which half moved.

        The masked fraction rides along only when the mask is on, so an unasked run's rows are the rows of
        yesterday. It is worth a column: it falling toward zero as the detector sharpens IS the self-balancing
        property — a model whose confident responses are all annotated has nothing left to leave unsupervised.
        """
        ambiguous = {} if self.ignored_fraction is None else {"ignored_fraction": self.ignored_fraction}
        return {
            "train_loss": float(self.total.detach()),
            "edge_loss": float(self.edge.detach()),
            "det_loss": float(self.detection.detach()),
            "contrastive_loss": float(self.contrastive.detach()),
            "hard_negative_loss": float(self.hard_negative.detach()),
            **ambiguous,
        }

    def difficulty(self) -> float | None:
        """Fraction of ANNOTATED sources whose true successor the logits do not rank top-1; `None` if there are none.

        Not the loss: with ~1-2% dense annotation a pair's loss tracks how many edges it happens to carry, so
        weighting by it would chase label count. This is bounded [0, 1], independent of edge count, and is the
        defect itself — a mislink is precisely a source whose argmax target is not its annotated successor.
        """
        annotated = self.edge_matrix.sum(dim=1) > 0
        if not bool(annotated.any()):
            return None
        rows = self.edge_matrix[annotated]
        predicted = self.edge_logits[annotated].argmax(dim=1, keepdim=True)
        return 1.0 - float(rows.gather(1, predicted).mean())


class TrackingLoss:
    """Every training term for the joint model — per pair, or vectorised over the padded batch. No optimiser here."""

    def __init__(
        self, config: TrackingLossConfig, contrastive: ContrastiveObjective, objective: AssociationObjective
    ) -> None:
        self._config = config
        self._contrastive = contrastive
        self._objective = objective

    def per_pair(self, prediction: PairPrediction, target: PairGroundTruth, *, detection: bool = True) -> _PairOutcome:
        """Every loss term for one pair from its already-computed forward (GT centres divided to the heads' grid).

        `detection=False` (generated hard scenes) zeroes the detection term: their blob appearance would corrupt
        a detector saturated on real data, while the association terms they carry teach the appearance-agnostic
        motion rule. The backbone still sees them through the edge gradient, which is ~30x smaller than det here.
        """
        config = self._config
        edge_logits = prediction.edge_logits
        edge_matrix = target.edge_matrix
        centres_t_grid = self._grid(target.source_centres)
        centres_t1_grid = self._grid(target.target_centres)
        edge = self._objective.per_pair(edge_logits, edge_matrix)
        ignore = config.ignore_ambiguous_above
        det = (
            BalancedBCE.of(prediction.detection_t.unsqueeze(0), [centres_t_grid], config.neg_weight, ignore)
            + BalancedBCE.of(prediction.detection_t1.unsqueeze(0), [centres_t1_grid], config.neg_weight, ignore)
            if detection
            else torch.zeros((), device=str(edge_logits.device))
        )
        contrastive = self._contrastive(prediction.source_features, prediction.target_features, edge_matrix)
        decoys = HardNegativeMargin.mine(
            target.source_centres, target.target_centres, edge_matrix, config.hard_negatives
        )
        hard_negative = HardNegativeMargin.of(edge_logits, edge_matrix, decoys)
        loss = (
            edge
            + config.det_weight * det
            + config.contrastive_weight * contrastive
            + config.hard_negative_weight * hard_negative
        )
        ignored = self._ignored((prediction.detection_t, centres_t_grid), (prediction.detection_t1, centres_t1_grid))
        reliability = (
            SampleWeight.of(
                target.frame_t,
                centres_t_grid,
                centres_t1_grid.to(torch.float32) * _ISOTROPIC_UM,  # isotropic µm on the heads' own grid
                centres_t_grid.to(torch.float32) * _ISOTROPIC_UM,
                config.gate_um,
            )
            if config.reliability_weighting
            else None
        )
        return _PairOutcome(loss, edge, det, contrastive, hard_negative, edge_logits, edge_matrix, ignored, reliability)

    def batched(
        self, prediction: BatchedPrediction, targets: list[PairGroundTruth]
    ) -> tuple[Float[Tensor, " b"], list[_PairOutcome]]:
        """Edge + detection VECTORISED over the padded batch — the per-pair total for backward, plus the outcomes.

        The pure-default hot path: `SoftmaxFocalBCE.batched` reads the padded `(B, max, max)` logit block and
        `BalancedBCE.per_frame` the `(B, ...)` detection maps, so the whole GPU loss is batched tensor ops with no
        Python loop over ragged logits. Returns the `(B,)` total for the step to weight and back-propagate, and
        the per-pair `_PairOutcome` list — detached diagnostics for the sampler and the logs — packaged from the
        padded result. Optional terms (contrastive, hard-negative) are zero here: the caller routes any run that
        asks for them to the per-pair loop.
        """
        config = self._config
        edges = self._padded_edges(targets, prediction.logits.shape[1], prediction.logits.device)
        edge = self._objective.batched(prediction.logits, edges, (prediction.source_mask, prediction.target_mask))
        grids_t = [self._grid(target.source_centres) for target in targets]
        grids_t1 = [self._grid(target.target_centres) for target in targets]
        detection = BalancedBCE.per_frame(
            prediction.detect_t, grids_t, config.neg_weight, config.ignore_ambiguous_above
        ) + BalancedBCE.per_frame(prediction.detect_t1, grids_t1, config.neg_weight, config.ignore_ambiguous_above)
        total = edge + config.det_weight * detection  # (B,)
        zero = edge.new_zeros(())
        outcomes = [
            _PairOutcome(
                total[index],
                edge[index],
                detection[index],
                zero,
                zero,
                forward.edge_logits,
                target.edge_matrix,
                self._ignored((forward.detection_t, grids_t[index]), (forward.detection_t1, grids_t1[index])),
            )
            for index, (forward, target) in enumerate(zip(prediction.per_pair, targets, strict=True))
        ]
        return total, outcomes

    def reliability_weights(self, outcomes: list[_PairOutcome]) -> list[Tensor]:
        """Per-pair reliability multipliers normalised to batch-mean 1 — redistributes emphasis, no rescale.

        All ones when the weighting is off (any pair carries no reliability), so an unasked run steps exactly
        yesterday's gradient. Mean-1 normalisation keeps the total loss scale — and therefore the effective
        learning rate — unchanged, so only the RELATIVE weight of clean vs ambiguous pairs moves.
        """
        raw = [out.reliability for out in outcomes]
        device = outcomes[0].total.device
        if any(weight is None for weight in raw):
            return [torch.ones((), device=device) for _ in outcomes]
        stacked = torch.stack([weight for weight in raw if weight is not None])
        return list(stacked / stacked.mean().clamp(min=1e-6))

    def _grid(self, centres: Int[Tensor, "n 3"]) -> Int[Tensor, "n 3"]:
        """Annotated centres divided to the heads' downsampled grid — the coordinate the detection loss reads."""
        downsample = torch.tensor(self._config.downsample, dtype=torch.float32, device=centres.device)
        return (centres.to(torch.float32) / downsample).round().long()

    @staticmethod
    def _padded_edges(targets: list[PairGroundTruth], size: int, device: torch.device) -> Float[Tensor, "b max max"]:
        """Each pair's `(s, u)` one-hot annotation zero-padded into the `(B, size, size)` block the loss reads."""
        edges = torch.zeros(len(targets), size, size, device=device)
        for index, target in enumerate(targets):
            matrix = target.edge_matrix
            edges[index, : matrix.shape[0], : matrix.shape[1]] = matrix.to(device)
        return edges

    def _ignored(self, *frames: tuple[Tensor, Tensor]) -> float | None:
        """The pair's mean unsupervised fraction, or `None` when nothing is masked — a sigmoid, no backward."""
        above = self._config.ignore_ambiguous_above
        if above is None:
            return None
        shares = [BalancedBCE.ignored_fraction(logits.unsqueeze(0), [centres], above) for logits, centres in frames]
        return sum(shares) / len(shares)

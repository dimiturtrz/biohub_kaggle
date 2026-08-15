"""The joint run's loss assembly — every training term for a pair, and the padded-batch vectorisation of them.

The trainer owns the STEP (forward, backward, optimiser); `JointLoss` owns the OBJECTIVE. It computes the four
decoupled terms — a detection `BalancedBCE` per frame, a `SoftmaxFocalBCE` over the links, an optional InfoNCE
over the node features, an optional hard-negative margin — for one pair (`per_pair`) or, for the pure-default
config, vectorised over the whole padded batch with no per-pair Python loop (`batched`). `_PairOutcome` is the
result type both return; it lives here because it is what the loss produces, not what the loop drives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.data.joint_dataset import PairSample
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.sample_weight import SampleWeight
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.models.joint_model import BatchedForward, JointForward
from celltrack.operating_point import TrackerConfig
from celltrack.training.contrastive_term import ContrastiveTerm
from celltrack.training.joint_config import JointTrainConfig

# The (1,4,4) downsample makes the heads' grid isotropic at the z-spacing, so one downsampled voxel is this many
# micrometres on every axis — the unit the reliability weight's crowd gate is measured in (see downsample memory).
_ISOTROPIC_UM = 1.625
_GATE_UM = TrackerConfig.shipped().linker.gate_um  # the linker's admission gate — a source's competing rivals


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


class JointLoss:
    """Every training term for the joint model — per pair, or vectorised over the padded batch. No optimiser here."""

    def __init__(self, config: JointTrainConfig, contrastive: ContrastiveTerm) -> None:
        self._config = config
        self._contrastive = contrastive

    def per_pair(self, out: JointForward, sample: PairSample, *, detection: bool = True) -> _PairOutcome:
        """Every loss term for one pair from its already-computed forward (GT centres divided to the heads' grid).

        `detection=False` (generated hard scenes) zeroes the detection term: their blob appearance would corrupt
        a detector saturated on real data, while the association terms they carry teach the appearance-agnostic
        motion rule. The backbone still sees them through the edge gradient, which is ~30x smaller than det here.
        """
        loss_config = self._config.loss
        source_centres, target_centres, edge_matrix = sample.source_centres, sample.target_centres, sample.edge_matrix
        downsample = torch.tensor(self._config.data.downsample, dtype=torch.float32, device=str(out.edge_logits.device))
        centres_t_grid = (source_centres.to(torch.float32) / downsample).round().long()
        centres_t1_grid = (target_centres.to(torch.float32) / downsample).round().long()
        edge = SoftmaxFocalBCE.of(
            out.edge_logits, edge_matrix, loss_config.link_axes, balanced=loss_config.balanced_links
        )
        ignore = loss_config.ignore_ambiguous_above
        det = (
            BalancedBCE.of(out.detection_t.unsqueeze(0), [centres_t_grid], loss_config.neg_weight, ignore)
            + BalancedBCE.of(out.detection_t1.unsqueeze(0), [centres_t1_grid], loss_config.neg_weight, ignore)
            if detection
            else torch.zeros((), device=str(out.edge_logits.device))
        )
        contrastive = self._contrastive.forward(out.source_features, out.target_features, edge_matrix)
        decoys = HardNegativeMargin.mine(source_centres, target_centres, edge_matrix, loss_config.hard_negatives)
        hard_negative = HardNegativeMargin.of(out.edge_logits, edge_matrix, decoys)
        loss = (
            edge
            + loss_config.det_weight * det
            + loss_config.contrastive_weight * contrastive
            + loss_config.hard_negative_weight * hard_negative
        )
        ignored = self._ignored((out.detection_t, centres_t_grid), (out.detection_t1, centres_t1_grid))
        reliability = (
            SampleWeight.of(
                sample.frame_t,
                centres_t_grid,
                centres_t1_grid.to(torch.float32) * _ISOTROPIC_UM,  # isotropic µm on the heads' own grid
                centres_t_grid.to(torch.float32) * _ISOTROPIC_UM,
                _GATE_UM,
            )
            if loss_config.reliability_weighting
            else None
        )
        return _PairOutcome(
            loss, edge, det, contrastive, hard_negative, out.edge_logits, edge_matrix, ignored, reliability
        )

    def batched(
        self, batched: BatchedForward, samples: list[PairSample]
    ) -> tuple[Float[Tensor, " b"], list[_PairOutcome]]:
        """Edge + detection VECTORISED over the padded batch — the per-pair total for backward, plus the outcomes.

        The pure-default hot path: `SoftmaxFocalBCE.batched` reads the padded `(B, max, max)` logit block and
        `BalancedBCE.per_frame` the `(B, ...)` detection maps, so the whole GPU loss is batched tensor ops with no
        Python loop over ragged logits. Returns the `(B,)` total for the step to weight and back-propagate, and
        the per-pair `_PairOutcome` list — detached diagnostics for the sampler and the logs — packaged from the
        padded result. Optional terms (contrastive, hard-negative) are zero here: the caller routes any run that
        asks for them to the per-pair loop.
        """
        loss = self._config.loss
        edges = self._padded_edges(samples, batched.logits.shape[1], batched.logits.device)
        edge = SoftmaxFocalBCE.batched(
            batched.logits,
            edges,
            (batched.source_mask, batched.target_mask),
            loss.link_axes,
            balanced=loss.balanced_links,
        )
        grids_t = [self._grid(sample.source_centres) for sample in samples]
        grids_t1 = [self._grid(sample.target_centres) for sample in samples]
        detection = BalancedBCE.per_frame(
            batched.detect_t, grids_t, loss.neg_weight, loss.ignore_ambiguous_above
        ) + BalancedBCE.per_frame(batched.detect_t1, grids_t1, loss.neg_weight, loss.ignore_ambiguous_above)
        total = edge + loss.det_weight * detection  # (B,)
        zero = edge.new_zeros(())
        outcomes = [
            _PairOutcome(
                total[index],
                edge[index],
                detection[index],
                zero,
                zero,
                forward.edge_logits,
                sample.edge_matrix,
                self._ignored((forward.detection_t, grids_t[index]), (forward.detection_t1, grids_t1[index])),
            )
            for index, (forward, sample) in enumerate(zip(batched.unpad(), samples, strict=True))
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
        downsample = torch.tensor(self._config.data.downsample, dtype=torch.float32, device=centres.device)
        return (centres.to(torch.float32) / downsample).round().long()

    @staticmethod
    def _padded_edges(samples: list[PairSample], size: int, device: torch.device) -> Float[Tensor, "b max max"]:
        """Each pair's `(s, u)` one-hot annotation zero-padded into the `(B, size, size)` block the loss reads."""
        edges = torch.zeros(len(samples), size, size, device=device)
        for index, sample in enumerate(samples):
            matrix = sample.edge_matrix
            edges[index, : matrix.shape[0], : matrix.shape[1]] = matrix.to(device)
        return edges

    def _ignored(self, *frames: tuple[Tensor, Tensor]) -> float | None:
        """The pair's mean unsupervised fraction, or `None` when nothing is masked — a sigmoid, no backward."""
        above = self._config.loss.ignore_ambiguous_above
        if above is None:
            return None
        shares = [BalancedBCE.ignored_fraction(logits.unsqueeze(0), [centres], above) for logits, centres in frames]
        return sum(shares) / len(shares)

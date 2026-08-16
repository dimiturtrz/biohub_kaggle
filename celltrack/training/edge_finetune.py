"""Hard-negative finetune of the edge transformer on real detections — the zni cost-tiered probe.

The dense-movie mislinks are 100 % affinity-inverted (`dense_diagnosis.MislinkSignal`): the transformer scores
a wrong *near* neighbour above the true *far* successor on the hard crowding cases. That is a signal error of a
specific shape — a confident wrong near-neighbour — and the fix for that shape is hard-negative mining, which
neither of `lna`'s refuted substrates (synthetic pairs, real positives only) supplied.

This finetunes the pilkwang edge transformer on *real detections*: run the shipped detector, match each
detection to the ground truth, and for every annotated source with a true successor mine its nearest wrong
target detections as explicit hard negatives. The objective keeps the frontier's focal-BCE over the sparse
annotated edges (softmax over sources, one parent per target) and adds a term that pushes the softmax
probability of `source → mined-decoy` toward zero — directly deflating the `P_chosen ≈ 0.69` the gate measured.
The UNet is frozen (its features are the fixed substrate); only the transformer head moves.

Run as a probe: a few hundred steps on the dense videos, then re-score the dense proxy and the inversion gate.
If the inversion falls and the dense Jaccard rises the lever escalates to a full train; if it degrades like the
earlier substrates the lever is refuted with the hard-negative shape now also ruled out.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int

from celltrack.eval.dense_diagnosis import DenseDiagnosis, DenseFateDiagnosis, Fate, MislinkSignal
from celltrack.eval.proxy import TestMovieProxy
from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer
from celltrack.models.temporal_unet_detector import TemporalUNetDetector, _VideoSource
from celltrack.tracker import CellTracker
from core.data.tracks import TrackGraph
from core.metrics.matching import UNMATCHED, DistanceMatcher
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# Train on the three denser-annotated 6bba movies and hold out the dense test movie for evaluation, so a drop in
# its mislink inversion is generalisation, not memorisation of the movie the finetune saw.
_TRAIN_STEMS = ("6bba_05b6850b", "6bba_969618f6", "6bba_fc83837d")
_EVAL_STEM = "6bba_05db0fb1"


@dataclass(frozen=True)
class GapSupervision:
    """One frame gap's training tensors: candidate positions, the GT edge matrix, and mined hard-negative pairs.

    `gt_matrix[i, j]` is 1 where detection `i` at `t` and detection `j` at `t+1` both match ground-truth nodes
    the annotation links; `hard_negatives` are `(source_row, decoy_row)` pairs where the source has a true
    successor elsewhere, so its link to the near decoy is a known negative — the inflated probability to deflate.
    """

    timepoint: int
    source_positions: Float[np.ndarray, "s 3"]
    target_positions: Float[np.ndarray, "t 3"]
    gt_matrix: Float[np.ndarray, "s t"]
    hard_negatives: Int[np.ndarray, "h 2"]

    @staticmethod
    def of(
        detections: TrackGraph, truth: TrackGraph, matcher: DistanceMatcher, hard_negatives: int
    ) -> list[GapSupervision]:
        """Build per-gap supervision from real detections: GT edge matrix by matching, hard negatives by proximity."""
        det_to_gt = matcher.match(detections, truth).gt_rows
        gt_edge_codes = GapSupervision._edge_codes(truth.edge_rows(), len(truth.node_ids))
        timepoints = detections.timepoints()
        positions = detections.positions().astype(np.float32)
        samples: list[GapSupervision] = []
        for timepoint in np.unique(timepoints)[:-1].tolist():
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            if len(sources) == 0 or len(targets) == 0:
                continue
            gt_matrix = GapSupervision._gt_matrix(
                det_to_gt[sources], det_to_gt[targets], gt_edge_codes, len(truth.node_ids)
            )
            if gt_matrix.sum() == 0:  # no annotated edge spans this gap — nothing to supervise
                continue
            mined = GapSupervision._mine_hard_negatives(
                gt_matrix, positions[sources], positions[targets], hard_negatives
            )
            samples.append(GapSupervision(int(timepoint), positions[sources], positions[targets], gt_matrix, mined))
        return samples

    @staticmethod
    def _edge_codes(edge_rows: Int[np.ndarray, "e 2"], count: int) -> set[int]:
        """The GT edges as flat `source*count + target` codes — an O(1) membership test for a detected pair."""
        return set((edge_rows[:, 0] * count + edge_rows[:, 1]).tolist())

    @staticmethod
    def _gt_matrix(
        source_gt: Int[np.ndarray, "s"], target_gt: Int[np.ndarray, "t"], gt_edge_codes: set[int], count: int
    ) -> Float[np.ndarray, "s t"]:
        """1 where a source and target detection match GT nodes the annotation links, else 0."""
        matrix = np.zeros((len(source_gt), len(target_gt)), dtype=np.float32)
        for i, gt_source in enumerate(source_gt):
            if gt_source == UNMATCHED:
                continue
            for j, gt_target in enumerate(target_gt):
                if gt_target != UNMATCHED and gt_source * count + gt_target in gt_edge_codes:
                    matrix[i, j] = 1.0
        return matrix

    @staticmethod
    def _mine_hard_negatives(
        gt_matrix: Float[np.ndarray, "s t"],
        source_positions: Float[np.ndarray, "s 3"],
        target_positions: Float[np.ndarray, "t 3"],
        keep: int,
    ) -> Int[np.ndarray, "h 2"]:
        """For each source with a true successor, its `keep` nearest *wrong* target detections — the hard negatives.

        The mining itself lives with the term that consumes it (`HardNegativeMargin`), so this probe and the
        joint trainer mine the SAME decoy set and differ only in the objective placed over it — which is the
        whole comparison between the refuted absolute form and the margin form.
        """
        mined = HardNegativeMargin.mine(
            torch.from_numpy(source_positions), torch.from_numpy(target_positions), torch.from_numpy(gt_matrix), keep
        )
        return mined.numpy().astype(np.int64)


@dataclass(frozen=True)
class EdgeFinetuneConfig:
    """The probe's knobs — the seed to finetune, the training videos, and the hard-negative objective weight."""

    device: str = "cuda"
    threshold: float = 0.99
    steps: int = 400
    learning_rate: float = 1.0e-4
    hard_negatives: int = 4
    hard_negative_weight: float = 1.0
    objective: str = "absolute"
    train_stems: tuple[str, ...] = _TRAIN_STEMS
    save_to: Path = field(default=Path("edge_predictor_zni.pth"))


class EdgeHardNegativeFinetuner:
    """Finetunes one seed's edge transformer on real-detection hard negatives, UNet frozen."""

    def __init__(self, scorer: EdgeTransformerScorer, config: EdgeFinetuneConfig) -> None:
        self._scorer = scorer
        self._config = config
        scorer.detector.requires_grad_(requires_grad=False)  # the UNet features are the fixed substrate

    def finetune(self, gaps: list[tuple[Path, GapSupervision]]) -> None:
        """Run the probe's optimisation: sample gaps, deflate hard-negative probabilities, keep the annotated edges."""
        transformer = self._scorer.transformer
        transformer.train()
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=self._config.learning_rate)
        sources = {path: TemporalUNetDetector._open_source(path) for path in {path for path, _ in gaps}}  # noqa: SLF001
        for step in range(self._config.steps):
            path, gap = gaps[step % len(gaps)]
            loss = self._loss(sources[path], gap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                logger.info("step %4d  loss=%.4f", step, loss.item())
        transformer.eval()

    def _loss(self, source: _VideoSource, gap: GapSupervision) -> Float[torch.Tensor, ""]:
        """Frontier focal-BCE on the annotated edges plus the hard-negative probability the mislinks inflate."""
        scored = EdgeGap(source, gap.timepoint, gap.source_positions, gap.target_positions, self._config.device)
        logits = self._scorer._gap_logits(scored)  # noqa: SLF001  (finetuning the scorer's own head; UNet frozen)
        target = torch.as_tensor(gap.gt_matrix, device=self._config.device)
        loss = SoftmaxFocalBCE.of(logits, target)
        if len(gap.hard_negatives):
            loss = loss + self._config.hard_negative_weight * self._hard_negative_penalty(logits, target, gap)
        return loss

    def _hard_negative_penalty(
        self, logits: Float[torch.Tensor, "s t"], target: Float[torch.Tensor, "s t"], gap: GapSupervision
    ) -> Float[torch.Tensor, ""]:
        """The mined-decoy penalty. `absolute` deflates the decoy probability (refuted: flattens `P_true` too);
        `margin` reads only `logit_decoy - logit_true`, so a row-wide depression cancels and only re-ranking pays."""
        decoys = torch.as_tensor(gap.hard_negatives, device=self._config.device)
        if self._config.objective == "margin":
            return HardNegativeMargin.of(logits, target, decoys)
        probability = torch.softmax(logits, dim=0)
        return probability[decoys[:, 0], decoys[:, 1]].mean()

    # CLI orchestration of the zni probe: mount the training proxy + tracker, mine gaps, report the read-out.
    @staticmethod
    def _mount(root: DataRoot, device: str) -> tuple[TestMovieProxy, CellTracker]:
        """The training proxy and the shipped tracker mounted once — the tracker is finetuned in place, then scored."""
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root, _TRAIN_STEMS)
        tracker = CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            device,
        )
        return proxy, tracker

    @staticmethod
    def _report(label: str, fate: DenseFateDiagnosis, signal: MislinkSignal) -> None:
        """Log the held-out movie's correct/mislink counts and the affinity inversion — the probe's read-out."""
        counts = fate.counts()
        mean_true, mean_chosen = signal.means()
        logger.info(
            "%-6s correct=%d mislink=%d  inversion=%.1f%%  P_true=%.3f P_chosen=%.3f",
            label,
            counts[Fate.CORRECT],
            counts[Fate.MISLINK_CONFLICT] + counts[Fate.MISLINK_FREE],
            100.0 * signal.inverted_fraction(),
            mean_true,
            mean_chosen,
        )

    @staticmethod
    def _training_gaps(
        proxy: TestMovieProxy, tracker: CellTracker, config: EdgeFinetuneConfig
    ) -> list[tuple[Path, GapSupervision]]:
        """Real-detection supervision for every training video — detections cached, matched, hard-negatives mined."""
        matcher = DistanceMatcher(spacing=proxy.spacing)
        gaps: list[tuple[Path, GapSupervision]] = []
        for path, truth in zip(proxy.paths, proxy.truths, strict=True):
            detections = tracker.detector.nodes(path.name, path, config.threshold)
            gaps.extend(
                (path, gap) for gap in GapSupervision.of(detections, truth.graph, matcher, config.hard_negatives)
            )
        return gaps


def main() -> None:
    """Finetune the edge transformer on real-detection hard negatives and save the head — the zni probe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Hard-negative finetune of the edge transformer on real detections.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--objective", choices=("absolute", "margin"), default="absolute")
    parser.add_argument("--save-to", type=Path, default=Path("edge_predictor_zni.pth"))
    parsed = parser.parse_args()

    config = EdgeFinetuneConfig(
        device=parsed.device,
        steps=parsed.steps,
        hard_negative_weight=parsed.hard_negative_weight,
        objective=parsed.objective,
        save_to=parsed.save_to,
    )
    root = DataRoot.from_config(parsed.config)
    proxy, tracker = EdgeHardNegativeFinetuner._mount(root, config.device)  # noqa: SLF001
    gaps = EdgeHardNegativeFinetuner._training_gaps(proxy, tracker, config)  # noqa: SLF001
    logger.info("mined %d supervised gaps over %d train videos", len(gaps), len(config.train_stems))

    evaluation = TestMovieProxy.load(root, (_EVAL_STEM,))
    eval_path, eval_truth = evaluation.paths[0], evaluation.truths[0].graph
    EdgeHardNegativeFinetuner._report(  # noqa: SLF001
        "before", *DenseDiagnosis.diagnose(tracker, eval_path, eval_truth, evaluation.spacing, config.device)
    )

    seed1 = tracker.edge_scorer.scorers[0]
    EdgeHardNegativeFinetuner(seed1, config).finetune(gaps)
    EdgeHardNegativeFinetuner._report(  # noqa: SLF001
        "after", *DenseDiagnosis.diagnose(tracker, eval_path, eval_truth, evaluation.spacing, config.device)
    )

    torch.save(
        {f"transformer.{name}": weight for name, weight in seed1.transformer.state_dict().items()}, config.save_to
    )
    logger.info("saved finetuned edge transformer to %s", config.save_to)


if __name__ == "__main__":
    main()

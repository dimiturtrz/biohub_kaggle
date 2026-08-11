"""Several seeds' edge transformers, blended in logit space — the edge analogue of the detector seed blend.

The detector already averages two seeds' per-frame logits (probabilities saturate near 0/1 and blend to mush;
logits keep the seeds' disagreement). The edge transformer earns the same treatment: each seed scores every
gap from its own UNet features and transformer head, we take a convex combination of their pre-softmax logits,
and only then soft-max over the sources of each target. On the 4-movie proxy this lifts the assignment linker
from 0.9125→0.9146 (with the density gap-bridge, 0.9134→0.9154) at 0.8·seed1 + 0.2·seed2.

Optionally the blended probabilities are then fused with a second, reversed pass — see
`BlendedEdgeTransformerScorer.fuse`.

Two further transforms ride on `EdgeBlendOptions`, both default OFF, so the shipped path is unchanged unless
asked for: `align_seed_moments` puts the seeds on one logit scale before the weighted sum
(`seed_moment_alignment`), and `view_tta` scores each seed over four flip views fused by a logarithmic opinion
pool (`flip_view_edge_scoring`). `seed_logit_moments` reports the raw per-seed scales the first of those acts
on, so the quality-versus-units question is settled by measurement rather than by a sweep, and `seed_scores`
reports the whole un-collapsed per-seed matrices the weighted sum throws away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from celltrack.edges.flip_view_edge_scoring import FlipViewEdgeScorer
from celltrack.edges.seed_moment_alignment import LogitMoments, SeedMomentAlignment
from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer, PrecomputedEdgeAffinity, video_gaps
from celltrack.models.prior_velocity import GapHistory
from core.data.tracks import TrackGraph

_DENOMINATOR_FLOOR = 1e-12


class EdgeBlendOptions(BaseModel):
    """The two disclosed frontier transforms around the blend, both default OFF — one knob object, not three flags.

    They travel together because they are asked the same way (a candidate arm of a proxy sweep) and neither is
    part of the shipped path until one of them earns it. `bidirectional` stays a separate argument: it is a
    shipped setting the tracker re-points per run, not a candidate.

    Pydantic (not a plain dataclass) for the reason `LinkerConfig` is: it is reached by a `--set
    edge_options.view_tta=true` override, which is user input, and it is the nested-config shape
    `celltrack.eval.proxy_eval.ConfigOverride` speaks — so the knobs are settable from the CLI without a
    second override mechanism, and a misspelt one is rejected at construction rather than ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    align_seed_moments: bool = False
    view_tta: bool = False


_DEFAULT_OPTIONS = EdgeBlendOptions()


@dataclass(frozen=True)
class GapSeedMoments:
    """One gap's per-seed logit moments, seed order matching the blend's — the alignment's evidence, per gap."""

    timepoint: int
    moments: tuple[LogitMoments, ...]


@dataclass(frozen=True)
class GapSeedScores:
    """One gap's per-seed RAW logits and per-seed probabilities, seed order matching the blend's.

    The blend keeps only the weighted MEAN of the seeds' logits; the DISPERSION between them — the one
    quantity an ensemble has that a single model cannot — is destroyed by the weighted sum before any
    consumer sees it. This carries both halves out intact so a diagnosis can ask whether that dispersion
    knows anything (`celltrack.eval.seed_dispersion`).

    `logits` are raw: pre-alignment and pre-weighting, the same convention `seed_logit_moments` reports in,
    because a scale difference between the seeds is part of what a dispersion read has to account for rather
    than something to normalise away first. `probabilities` are what each seed ALONE would hand the linker
    under this scorer's fusion setting, so they are directly comparable to the blended matrix beside them.
    """

    timepoint: int
    logits: Float[np.ndarray, "k s t"]
    probabilities: Float[np.ndarray, "k s t"]


class BlendedEdgeTransformerScorer:
    """A convex blend of several `EdgeTransformerScorer` seeds in logit space, then softmax-over-sources.

    The gap node order (shared detector nodes) is identical across seeds, so each seed's `s×t` logit matrix
    aligns row-for-row and the weighted sum is well defined. Drop-in for `EdgeTransformerScorer` in the linker
    pipeline: `affinities` returns the same `PrecomputedEdgeAffinity` the assignment cost consumes.

    With `bidirectional` the gap is additionally scored backwards and the two normalisations are fused by
    `fuse`. Fusion happens at the probability level, after the seed blend — the seeds still combine as logits.
    """

    def __init__(
        self,
        scorers: tuple[EdgeTransformerScorer, ...],
        weights: tuple[float, ...],
        *,
        bidirectional: bool = False,
        options: EdgeBlendOptions = _DEFAULT_OPTIONS,
    ) -> None:
        if len(scorers) != len(weights):
            raise ValueError(f"got {len(scorers)} scorers but {len(weights)} weights")
        self.scorers = scorers
        self.weights = weights
        self.bidirectional = bidirectional
        self.options = options

    @classmethod
    def from_packs(
        cls,
        packs: tuple[Path, ...],
        weights: tuple[float, ...],
        device: str = "cpu",
        *,
        bidirectional: bool = False,
        options: EdgeBlendOptions = _DEFAULT_OPTIONS,
    ) -> BlendedEdgeTransformerScorer:
        """Mount one `EdgeTransformerScorer` per pack and pair each with its blend weight."""
        scorers = tuple(EdgeTransformerScorer.from_pack(pack, device) for pack in packs)
        return cls(scorers, weights, bidirectional=bidirectional, options=options)

    def with_bidirectional(self, *, bidirectional: bool) -> BlendedEdgeTransformerScorer:
        """The same mounted seeds under the other fusion setting — so a sweep re-points the knob, not the mount."""
        return BlendedEdgeTransformerScorer(
            self.scorers, self.weights, bidirectional=bidirectional, options=self.options
        )

    def with_options(self, options: EdgeBlendOptions) -> BlendedEdgeTransformerScorer:
        """The same mounted seeds under a different transform set — the options half of re-pointing a sweep.

        Separate from `with_bidirectional` because the two are re-pointed by different owners: fusion is a
        per-pass direction question (`fused_affinities` flips it mid-run), the options are the config's
        operating point. Chaining both is how `CellTracker.with_config` re-mounts nothing at all.
        """
        return BlendedEdgeTransformerScorer(
            self.scorers, self.weights, bidirectional=self.bidirectional, options=options
        )

    @torch.no_grad()
    def affinities(self, path: Path, detections: TrackGraph, device: str) -> PrecomputedEdgeAffinity:
        """The blended source→target probability matrices, one per scored frame gap of the video.

        Gaps arrive in ascending time so each seed can carry its own `GapHistory` forward — the prior-velocity
        state a widened head reads. Per SEED, not per blend: the velocity a head consumes has to be the one its
        own weights produced (that is what its trainer fed it), and two seeds need not even agree on whether
        they were widened for the feature at all. A blend of unwidened seeds carries empty histories and is
        byte-identical to what this computed before the feature existed.
        """
        histories = tuple(GapHistory() for _ in self.scorers)
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for gap in video_gaps(path, detections, device):
            probabilities, histories = self._gap_probabilities(gap, histories)
            by_timepoint[gap.timepoint] = probabilities.cpu().numpy()
        return PrecomputedEdgeAffinity(by_timepoint)

    def _gap_probabilities(
        self, gap: EdgeGap, histories: tuple[GapHistory, ...]
    ) -> tuple[Float[Tensor, "s t"], tuple[GapHistory, ...]]:
        """One gap's probabilities and each seed's carried history — forward, optionally fused with the reverse."""
        blended, carried = self._forward_logits(gap, histories)
        forward = torch.softmax(blended, dim=0)
        if not self.bidirectional:
            return forward, carried
        reverse = torch.softmax(self._reverse_logits(gap), dim=0).T
        return self.fuse(forward, reverse), carried

    def seed_logit_moments(self, path: Path, detections: TrackGraph, device: str) -> list[GapSeedMoments]:
        """Every gap's per-seed logit mean and std, in ascending time — the numbers the blend weights act on.

        Reported BEFORE any alignment and before the weighting, because the question this answers is whether
        the seeds' raw scales differ at all: if they do, the shipped 0.8/0.2 is partly a temperature fix and
        aligning should move the best blend toward even; if they do not, 0.8/0.2 is the quality weighting we
        recorded it as. One pass over the video answers it — no sweep, no inference from a curve's shape.
        """
        histories = tuple(GapHistory() for _ in self.scorers)
        reported: list[GapSeedMoments] = []
        with torch.no_grad():
            for gap in video_gaps(path, detections, device):
                scored = self._seed_scored(gap, histories)
                histories = tuple(history for _, history in scored)
                moments = tuple(SeedMomentAlignment.moments(logits) for logits, _ in scored)
                reported.append(GapSeedMoments(gap.timepoint, moments))
        return reported

    @torch.no_grad()
    def seed_scores(self, path: Path, detections: TrackGraph, device: str) -> list[GapSeedScores]:
        """Every gap's per-seed logits AND per-seed probabilities, in ascending time — the blend, un-collapsed.

        `affinities` returns the weighted sum only, so the seeds' spread is gone by the time a linker or a
        diagnosis can look at it. This is the same streaming pass (each seed carrying its own history, gaps in
        time order, so the numbers are the ones the shipped run computes) with the collapse omitted.
        """
        histories = tuple(GapHistory() for _ in self.scorers)
        reported: list[GapSeedScores] = []
        for gap in video_gaps(path, detections, device):
            scored = self._seed_scored(gap, histories)
            histories = tuple(history for _, history in scored)
            logits = [seed_logits for seed_logits, _ in scored]
            probabilities = [
                self._seed_probabilities(scorer, gap, seed_logits)
                for scorer, seed_logits in zip(self.scorers, logits, strict=True)
            ]
            reported.append(
                GapSeedScores(
                    timepoint=gap.timepoint,
                    logits=torch.stack(logits).cpu().numpy(),
                    probabilities=torch.stack(probabilities).cpu().numpy(),
                )
            )
        return reported

    def _seed_probabilities(
        self, scorer: EdgeTransformerScorer, gap: EdgeGap, logits: Float[Tensor, "s t"]
    ) -> Float[Tensor, "s t"]:
        """What ONE seed alone would hand the linker for this gap — its own softmax, fused as the blend fuses.

        The fusion setting is applied per seed rather than skipped, because the comparison a dispersion read
        makes is against the blended matrix the linker consumed: a per-seed number taken forward-only while the
        blend was bidirectional would differ from it for a reason that has nothing to do with the seeds.
        """
        forward = torch.softmax(logits, dim=0)
        if not self.bidirectional:
            return forward
        reverse = torch.softmax(self._reverse_seed_logits(scorer, gap), dim=0).T
        return self.fuse(forward, reverse)

    def _seed_scored(
        self, gap: EdgeGap, histories: tuple[GapHistory, ...]
    ) -> list[tuple[Float[Tensor, "s t"], GapHistory]]:
        """Each seed's raw forward logits for the gap, with the history it hands the next gap — one per seed."""
        if self.options.view_tta:
            return [
                FlipViewEdgeScorer(scorer).gap_logits(gap, history)
                for scorer, history in zip(self.scorers, histories, strict=True)
            ]
        return [
            scorer._seed_logits(gap, history)  # noqa: SLF001
            for scorer, history in zip(self.scorers, histories, strict=True)
        ]

    def _forward_logits(
        self, gap: EdgeGap, histories: tuple[GapHistory, ...]
    ) -> tuple[Float[Tensor, "s t"], tuple[GapHistory, ...]]:
        """The seeds' forward logits convex-combined, plus the history each seed hands the next gap."""
        scored = self._seed_scored(gap, histories)
        aligned = self._aligned([logits for logits, _ in scored])
        weighted = torch.stack([weight * logits for weight, logits in zip(self.weights, aligned, strict=True)])
        return weighted.sum(dim=0), tuple(history for _, history in scored)

    def _reverse_logits(self, gap: EdgeGap) -> Float[Tensor, "t s"]:
        """The seeds' logits for the gap scored with the temporal pair SWAPPED, combined before normalisation.

        The reverse direction carries no prior velocity — see `EdgeTransformerScorer._gap_logits` for why its
        sources have no non-circular history — so a widened head reads zeros here and no state crosses gaps.
        """
        scored = [self._reverse_seed_logits(scorer, gap) for scorer in self.scorers]
        weighted = torch.stack(
            [weight * logits for weight, logits in zip(self.weights, self._aligned(scored), strict=True)]
        )
        return weighted.sum(dim=0)

    def _reverse_seed_logits(self, scorer: EdgeTransformerScorer, gap: EdgeGap) -> Float[Tensor, "t s"]:
        """One seed's reverse-direction logits — pooled over the flip views when view TTA is on."""
        if self.options.view_tta:
            return FlipViewEdgeScorer(scorer).reverse_gap_logits(gap)
        return scorer._gap_logits(gap, reverse=True)  # noqa: SLF001

    def _aligned(self, logits: list[Float[Tensor, "s t"]]) -> list[Float[Tensor, "s t"]]:
        """Seeds after the first rescaled onto the first seed's moments — the identity when alignment is off.

        Seed 0 is the reference rather than, say, the mean of all seeds, so the aligned blend is expressed in
        the units the shipped 0.8/0.2 was tuned in: a weight sweep run on top of this is directly comparable to
        the one already recorded, which is what makes the quality-vs-units question answerable.
        """
        if not self.options.align_seed_moments:
            return logits
        reference = logits[0]
        return [reference, *(SeedMomentAlignment.align(other, reference) for other in logits[1:])]

    @staticmethod
    def fuse(forward: Float[Tensor, "s t"], reverse: Float[Tensor, "s t"]) -> Float[Tensor, "s t"]:
        """The harmonic mean of the two directions' probabilities — a pair scores high only if both agree.

        `forward` is normalised over the sources of each target (who is this cell's parent?); `reverse` is the
        same gap scored with the temporal pair swapped and transposed back, so it is normalised over the targets
        of each source (which successor does this cell claim?). They are different questions of the same weights.

        Our dense mislinks are affinity-inverted — a confident wrong NEAR neighbour (P 0.686) beats the true far
        successor (P 0.134). Such a neighbour tends to win forwards and lose backwards, because it already has a
        better parent of its own. The harmonic mean is dominated by the smaller of its two arguments, so it
        demands mutual consistency; an arithmetic mean would let the one confident direction carry the disagreed
        pair through at half its strength, which is exactly the failure mode.

        Zero-safe by construction: the denominator is floored, and a pair both directions call impossible fuses
        to 0 rather than 0/0. Flooring rather than `torch.where`-ing keeps a single expression, and `where`
        would evaluate the NaN branch anyway.
        """
        total = forward + reverse
        return 2.0 * forward * reverse / total.clamp(min=_DENOMINATOR_FLOOR)

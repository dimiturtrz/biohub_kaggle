"""Confusor-restricted proxy referee — an offline association metric that SEES the linker/head axis the
aggregate CTC proxy is blind to above 0.90.

The wall-break (0.902 -> 0.924 LB) was a PURE linker swap: per-frame greedy `motion` -> global `ilp`, same
detections, same affinity. Aggregate CTC saturates and even INVERTS at that ceiling, so it cannot referee a
proxy-blind association win (finer + complete-candidate). The confusor is an association-STRUCTURE
phenomenon: `MislinkSignal` reads, over the GT edges the tracker MISLINKED, the affinity's true- vs
chosen-target probability and the RANK of the true successor. That set shifts with the linker, so `mislinks`
is a linker-sensitive integer that moves where the normalised CTC score has saturated.

Two axes, one ruler — selected by `--source`:

  pack  : the CHAMPION pilkwang affinity under two LINKERS (motion, ilp). Proves the mislink metric is
          linker-sensitive (the 0.902-vs-0.924 A/B). If `ilp` shows fewer mislinks / lower inverted_fraction
          than `motion` while the split score barely moves, the ruler is real and unblocks offline gating.

  joint : a jointly-trained checkpoint's OWN detector+transformer (`from_joint`), fixed detector, swapped
          HEAD. avl8 (C=288 head on the frozen pilkwang detector) and finer122 (a (1,2,2) co-adapt head) both
          ride this path. Because the detector is unchanged, the detected-candidate set — the mislink
          DENOMINATOR — matches the champion pilkwang run, so the rank-split moves iff the head got better at
          the confusor. `--gate` prints the head-gain verdict against the champion baseline row.

Champion baseline (pilkwang affinity, ilp linker, CV8):
    mislinks=77  r0=3  r1=37  r2=16  r3+/uns=21   (74/77 head-bound: r>=1)

Rank split on the residual mislinks:
    r0    = affinity ranks the TRUE successor first yet the linker still mislinked -> a COST/STRUCTURE error
            the solver could fix (card-free lever). Linker-owned; a head swap does NOT move it.
    r>=1  = affinity itself ranks a wrong neighbour above true -> a SIGNAL error only a better head fixes
            (finer/e9ci, GPU-gated).
    unscored (-1) = an endpoint the affinity never scored (bridge midpoint / multi-gap chosen).

celltrack's `ilp` linker is geometry/motion-global and IGNORES the edge affinity, so under it a head swap
cannot move the mislink COUNT — only the affinity READOUT over the fixed mislinked set (top1 / inv / ranks),
which an affinity-USING submission linker (tracksdata ILPSolver) would convert into fewer real mislinks.
`motion`/`flow` are bonus-readers, so their inv/pTrue are valid; read ilp's COUNT + ranks only.

Run (GPU-if-available; on a cold logit cache the affinity recompute is the cost — put it on CUDA):
    uv run python -m celltrack.analysis.confusor_referee --source pack
    uv run python -m celltrack.analysis.confusor_referee --source joint --checkpoint armB_warmdet.pt \
        --linkers ilp --gate
    uv run python -m celltrack.analysis.confusor_referee --source joint --checkpoint finer122_coadapt_long.pt \
        --linkers motion ilp flow
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import override

import numpy as np
import torch

from celltrack.eval.dense_diagnosis import AffinityIndex, MislinkSignal
from celltrack.eval.proxy import CV_MOVIES, TestMovieProxy
from celltrack.linkers.linkers import LinkerConfig
from celltrack.operating_point import TrackerConfig
from celltrack.reference_mount import ReferenceMount
from celltrack.tracker import CellTracker
from core.metrics.matching import DistanceMatcher
from core.paths import DataRoot

logger = logging.getLogger(__name__)

DATASET = "biohub_cell_tracking"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHAMPION_ROW = "mislinks=77  r0=3  r1=37  r2=16  r3+/uns=21  (74/77 head-bound)"
_CHAMPION_R1 = 37
# Head-gain gate (joint source): a head beats champion iff it ranks true first on more mislinks AND fewer
# inversions AND the r>=1 signal-error band collapses below the champion's 37.
_GATE_TOP1, _GATE_INV, _GATE_R1 = 0.45, 0.45, 30
_RANK_R2, _RANK_R3 = 2, 3  # rank bins beyond r0/r1; named to satisfy PLR2004


class AffinitySource(ABC):
    """How the tracker's edge affinity is mounted for a linker — the one axis the three referees differ on."""

    @abstractmethod
    def tracker_for(self, proc: Path, cfg: TrackerConfig) -> CellTracker:
        """Return the shipped tracker carrying this source's affinity, configured for `cfg`'s linker."""

    @staticmethod
    def build(args: argparse.Namespace, proc: Path) -> AffinitySource:
        """The source the `--source` flag names, its joint checkpoint stem resolved under `proc`."""
        if args.source == "pack":
            return PackAffinity()
        return JointAffinity(checkpoint=proc / args.checkpoint)


@dataclass
class PackAffinity(AffinitySource):
    """The champion pilkwang two-pack blend. Loaded once; the linker is re-pointed with `with_config` (its
    native len-2 seed blend rides along, so `cfg` keeps the shipped `edge_blend`)."""

    _pipeline: CellTracker | None = field(default=None, repr=False)

    @override
    def tracker_for(self, proc: Path, cfg: TrackerConfig) -> CellTracker:
        if self._pipeline is None:
            self._pipeline = ReferenceMount.pilkwang(proc, DEVICE, TrackerConfig.shipped())
        return self._pipeline.with_config(cfg)


@dataclass
class JointAffinity(AffinitySource):
    """A jointly-trained checkpoint's own detector+transformer. Re-mounted per linker (logits cached), so the
    config's linker is baked at mount; `from_joint` builds its own single-member (1.0,) blend and ignores the
    shipped len-2 `edge_blend` in `cfg`, so no `with_config` re-weight can mismatch it."""

    checkpoint: Path

    @override
    def tracker_for(self, proc: Path, cfg: TrackerConfig) -> CellTracker:
        return ReferenceMount.joint(self.checkpoint, proc, DEVICE, cfg)


@dataclass
class RefereeRow:
    """One linker's pooled confusor readout over CV8 — the print row and the gate input."""

    linker: str
    n: int
    inv: float
    p_true: float
    p_chosen: float
    top1: float
    r0: int
    r1: int
    r2: int
    r3plus_uns: int

    @classmethod
    def score(
        cls, source: AffinitySource, proc: Path, proxy: TestMovieProxy, matcher: DistanceMatcher, linker: str
    ) -> RefereeRow:
        base = TrackerConfig.shipped()
        cfg = replace(base, linker=LinkerConfig(name=linker, gate_um=base.linker.gate_um))
        configured = source.tracker_for(proc, cfg)
        signals = []
        for path, truth in zip(proxy.paths, proxy.truths, strict=True):
            tracked = configured.run_scored(path.name, path)
            matching = matcher.match(tracked.graph, truth.graph)
            index = AffinityIndex.of(tracked.detections, tracked.affinity)
            signals.append(MislinkSignal.of(tracked.graph, truth.graph, matching, index))
            logger.info("  %s / %s done", linker, path.stem)
        pooled = MislinkSignal.pooled(signals)
        mean_true, mean_chosen = pooled.means()
        ranks = pooled.ranks
        return cls(
            linker=linker,
            n=len(pooled.p_true),
            inv=pooled.inverted_fraction(),
            p_true=mean_true,
            p_chosen=mean_chosen,
            top1=pooled.top1_fraction(),
            r0=int(np.sum(ranks == 0)),
            r1=int(np.sum(ranks == 1)),
            r2=int(np.sum(ranks == _RANK_R2)),
            r3plus_uns=int(np.sum(ranks >= _RANK_R3)) + int(np.sum(ranks == -1)),
        )

    def log(self) -> None:
        logger.info(
            "%-8s %8d %8.3f %8.3f %8.3f %8.3f %8d %8d %8d %8d",
            self.linker,
            self.n,
            self.inv,
            self.p_true,
            self.p_chosen,
            self.top1,
            self.r0,
            self.r1,
            self.r2,
            self.r3plus_uns,
        )

    def verdict(self) -> str:
        """Head-gain gate (joint source): does this head beat the champion's confusor readout?"""
        gain = (self.r1 < _GATE_R1) and (self.inv < _GATE_INV) and (self.top1 > _GATE_TOP1)
        tag = "SUBMIT-CANDIDATE" if gain else "HEAD-BOUND-CONFIRMED (no submit)"
        return (
            f"VERDICT: {tag}  (gate: top1>{_GATE_TOP1} [{self.top1:.3f}] AND inv<{_GATE_INV} "
            f"[{self.inv:.3f}] AND r1<{_GATE_R1} [{self.r1}]; champ r1={_CHAMPION_R1})"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Confusor-restricted proxy referee (pack or joint affinity).")
    parser.add_argument("--source", choices=("pack", "joint"), default="pack")
    parser.add_argument("--checkpoint", default="armB_warmdet.pt", help="joint source: .pt stem under processed/")
    parser.add_argument("--linkers", nargs="+", default=None, help="default: motion ilp (pack) / ilp (joint)")
    parser.add_argument("--gate", action="store_true", help="joint source: print the head-gain verdict per row")
    args = parser.parse_args()

    root = DataRoot.from_config(Path("paths.yaml"))
    proc = root.processed(DATASET)
    proxy = TestMovieProxy.load(root, CV_MOVIES)
    matcher = DistanceMatcher(spacing=proxy.spacing)
    source = AffinitySource.build(args, proc)
    linkers = args.linkers or (["motion", "ilp"] if args.source == "pack" else ["ilp"])

    if args.gate:
        logger.info("champion baseline: %s", CHAMPION_ROW)
    logger.info(
        "%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s",
        "linker",
        "mislinks",
        "inv",
        "pTrue",
        "pChos",
        "top1",
        "r0",
        "r1",
        "r2",
        "r3+/uns",
    )
    for linker in linkers:
        row = RefereeRow.score(source, proc, proxy, matcher, linker)
        row.log()
        if args.gate:
            logger.info("%s", row.verdict())
            logger.info("NOTE: mislink COUNT (%d) is linker-geometry, NOT head-sensitive under ilp.", row.n)


if __name__ == "__main__":
    main()

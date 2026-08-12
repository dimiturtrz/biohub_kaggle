"""How cells actually move — the structure of the motion, not just how far it goes.

The campaign knows the DISPLACEMENT distribution (median 1.67 um, p99 7.20, max 9.96) and has used it to
derive the linker gate. It has never measured the motion's STRUCTURE, and every motion-shaped attempt so far
has been refuted: a damped-velocity linker cost, a motion-predicted geometry inside the flow solver, a prior
velocity fed to the edge head. All three estimate velocity from ONE previous step, where the signal is the
same size as the noise — the median step (1.67 um) equals the median localisation offset (1.67 um), so a
single-frame velocity has a signal-to-noise ratio near one. Whether that is why they failed, or whether the
motion simply carries no usable direction, is the question this module answers.

IT GATES AN ARCHITECTURE CHANGE. The backbone's temporal attention has no positional encoding, so it is
permutation-invariant over time and cannot represent motion at all; the proposed fix is a positional encoding
plus a window wider than two frames, which would let the model integrate displacement over several steps. That
only pays if displacement ACCUMULATES — if cells hold a direction, averaging over frames beats the per-step
noise floor. If they tumble, averaging converges to zero and the wider window buys nothing. So the exponent
below decides whether to build it.

THE INSTRUMENT IS MEAN SQUARED DISPLACEMENT, not the per-step turn angle, because localisation noise corrupts
the two differently. Independent per-node error adds a CONSTANT to the MSD at every lag, so it flattens the
curve at short lags and leaves the large-lag slope alone; a log-log fit taken from a lag where the constant no
longer dominates therefore reads the true scaling. The same noise biases every turn angle towards 90 degrees
with no lag to escape along. Turn angle is still reported, cut by step length, because a long step has a
better signal-to-noise ratio than a short one — so persistence RISING with step length is evidence the
direction is real rather than an artefact, and that cut is a control the aggregate cannot provide.

THE COUPLING QUESTION IS SEPARATE AND IS ABOUT A DIFFERENT LEVER. If neighbouring cells move together — tissue
flowing, cells pushed rather than crawling — then a cell's neighbours predict its displacement even when its
own history does not, which is a signal available at inference from the current frame alone. The shuffled
control is what makes that readable: same frame, same displacements, partners permuted, so any coupling above
the control is spatial structure rather than a shared global drift.

All of it is measured on ANNOTATED tracks. Those are ground truth, so no linker error contaminates the motion
— at the price that the annotated cells are a sparse sample of the population, and a lineage the annotators
chose to follow may not move like one they skipped.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float, Int

from celltrack.eval.proxy import CV_MOVIES, TestMovieProxy
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# A step shorter than this is indistinguishable from standing still. A displacement is the difference of two
# independently localised positions, so it carries sqrt(2) times the per-node error, and the measured median
# offset on the dense movie is 1.675 um (celltrack.analysis.localisation) — hence sqrt(2) * 1.675.
STATIONARY_UM = 2.369
# The linker's admission gate, and so the scale at which a neighbour is a candidate the assignment could take
# instead. Coupling is reported in multiples of it rather than at invented radii.
GATE_UM = 10.0
_COUPLING_BANDS = (1.0, 2.0, 4.0)
# Below this lag the localisation constant still dominates the MSD, so the log-log fit starts above it.
_FIT_FROM_LAG = 2
_MIN_FIT_POINTS = 3
_BALLISTIC, _DIFFUSIVE = 2.0, 1.0
_PERCENTILES = (25.0, 50.0, 75.0, 95.0)
# A frame's mean displacement is only a drift estimate if enough cells contribute to it.
_MIN_DRIFT_SAMPLES = 3
# Coupling is a statement about a PAIR, so a frame holding one cell has nothing to say.
_MIN_COUPLED = 2


@dataclass(frozen=True)
class Chains:
    """Unambiguous lineage segments: maximal runs of one-in, one-out nodes, in row space.

    A chain BREAKS at a division rather than following either daughter, because the two daughters' motion is
    driven by cytokinesis rather than by whatever the mother was doing — including it would inject a spurious
    direction reversal into exactly the statistic this module reports. It also breaks at a temporal gap, since
    a chain is only a velocity record if consecutive rows are consecutive frames.
    """

    rows: tuple[Int[np.ndarray, "k"], ...]

    @classmethod
    def of(cls, graph: TrackGraph) -> Self:
        """Every maximal single-parent single-child run of consecutive frames."""
        adjacency = Adjacency.of(graph)
        timepoints = graph.timepoints()
        chains: list[Int[np.ndarray, "k"]] = []
        visited: set[int] = set()
        for start in cls._starts(adjacency):
            if start in visited:
                continue
            chain = cls._walk(start, adjacency, timepoints)
            visited.update(chain)
            if len(chain) > 1:
                chains.append(np.asarray(chain, dtype=np.int64))
        return cls(rows=tuple(chains))

    @staticmethod
    def _starts(adjacency: Adjacency) -> list[int]:
        """Rows that begin a chain: no single unambiguous parent to have come from."""
        return [row for row in range(len(adjacency.in_degrees)) if adjacency.in_degrees[row] != 1]

    @staticmethod
    def _walk(start: int, adjacency: Adjacency, timepoints: Int[np.ndarray, "n"]) -> list[int]:
        """Follow single successors from `start` while each step advances exactly one frame."""
        chain, row = [start], start
        while len(adjacency.successors[row]) == 1:
            successor = adjacency.successors[row][0]
            if adjacency.in_degrees[successor] != 1 or timepoints[successor] != timepoints[row] + 1:
                break
            chain.append(successor)
            row = successor
        return chain

    def steps(self, positions_um: Float[np.ndarray, "n 3"]) -> "Steps":
        """Every consecutive displacement, tagged with the chain it came from."""
        vectors = [positions_um[chain[1:]] - positions_um[chain[:-1]] for chain in self.rows]
        origins = [np.full(len(chain) - 1, index) for index, chain in enumerate(self.rows)]
        return Steps(
            vectors=np.concatenate(vectors) if vectors else np.zeros((0, 3)),
            chain_index=np.concatenate(origins) if origins else np.zeros(0, dtype=np.int64),
        )


@dataclass(frozen=True)
class Steps:
    """Consecutive per-frame displacement vectors in micrometres, and which chain each belongs to."""

    vectors: Float[np.ndarray, "s 3"]
    chain_index: Int[np.ndarray, "s"]

    def lengths(self) -> Float[np.ndarray, "s"]:
        """How far each step travels."""
        return np.linalg.norm(self.vectors, axis=1)

    def stationary_fraction(self, threshold_um: float = STATIONARY_UM) -> float:
        """Share of steps shorter than the localisation noise — cells that may not have moved at all."""
        return float(np.mean(self.lengths() < threshold_um))

    def turn_cosines(self) -> tuple[Float[np.ndarray, "p"], Float[np.ndarray, "p"]]:
        """For each consecutive step PAIR, the cosine between them and the shorter of the two lengths.

        The shorter length is the pair's signal-to-noise handle: a turn measured between two long steps is
        far better determined than one involving a step the size of the localisation error, so conditioning
        on it is what separates real persistence from noise-induced right angles.
        """
        same_chain = self.chain_index[:-1] == self.chain_index[1:]
        first, second = self.vectors[:-1][same_chain], self.vectors[1:][same_chain]
        first_length, second_length = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
        usable = (first_length > 0) & (second_length > 0)
        cosines = np.sum(first[usable] * second[usable], axis=1) / (first_length[usable] * second_length[usable])
        return cosines, np.minimum(first_length[usable], second_length[usable])


@dataclass(frozen=True)
class PersistenceByStep:
    """Mean turn cosine within bands of the pair's weaker step — the noise control, as a trend.

    Pure localisation noise produces cosines centred on zero at EVERY step length. Real directional
    persistence is diluted by noise in inverse proportion to step length, so it must RISE across these bands.
    A flat profile is the null; a rising one is the signal.
    """

    edges_um: tuple[float, ...]
    mean_cosine: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def of(cls, cosines: Float[np.ndarray, "p"], weaker_um: Float[np.ndarray, "p"], edges: tuple[float, ...]) -> Self:
        """Bin the turn cosines by the weaker step's length."""
        bands = np.digitize(weaker_um, edges)
        means = tuple(
            float(cosines[bands == band].mean()) if np.any(bands == band) else float("nan")
            for band in range(len(edges) + 1)
        )
        counts = tuple(int(np.sum(bands == band)) for band in range(len(edges) + 1))
        return cls(edges_um=edges, mean_cosine=means, counts=counts)


@dataclass(frozen=True)
class MeanSquaredDisplacement:
    """MSD against lag, and the log-log slope — the scaling that says ballistic, diffusive, or confined.

    Independent localisation error inflates every lag by the same constant, so it steepens nothing: the slope
    read above `_FIT_FROM_LAG` is the motion's, while the intercept absorbs the noise. An exponent near 2 is
    straight-line travel and means displacement ACCUMULATES over frames; near 1 is a random walk, where a
    multi-frame average converges to zero and a wider temporal window carries no more direction than one step.
    """

    lags: tuple[int, ...]
    msd_um2: tuple[float, ...]
    exponent: float

    @classmethod
    def of(cls, chains: Chains, positions_um: Float[np.ndarray, "n 3"], max_lag: int) -> Self:
        """Time-average and ensemble-average the squared displacement at each lag."""
        lags = tuple(range(1, max_lag + 1))
        msd = tuple(cls._at_lag(chains, positions_um, lag) for lag in lags)
        return cls(lags=lags, msd_um2=msd, exponent=cls._exponent(lags, msd))

    @staticmethod
    def _at_lag(chains: Chains, positions_um: Float[np.ndarray, "n 3"], lag: int) -> float:
        """Mean squared displacement over every pair of rows `lag` frames apart within a chain."""
        squares = [
            np.sum((positions_um[chain[lag:]] - positions_um[chain[:-lag]]) ** 2, axis=1)
            for chain in chains.rows
            if len(chain) > lag
        ]
        return float(np.concatenate(squares).mean()) if squares else float("nan")

    @staticmethod
    def _exponent(lags: tuple[int, ...], msd: tuple[float, ...]) -> float:
        """Slope of log MSD against log lag, fitted above the lag where the noise constant dominates."""
        usable = [(lag, value) for lag, value in zip(lags, msd, strict=True) if lag >= _FIT_FROM_LAG and value > 0]
        if len(usable) < _MIN_FIT_POINTS:
            return float("nan")
        log_lag = np.log([lag for lag, _ in usable])
        return float(np.polyfit(log_lag, np.log([value for _, value in usable]), 1)[0])


@dataclass(frozen=True)
class NeighbourCoupling:
    """Do near neighbours move together — and does that beat a same-frame shuffle?

    Cells in a tissue can be carried rather than crawling, in which case a cell's displacement is predictable
    from its neighbours' even when its own past is too noisy to help. That would be a signal available from the
    CURRENT frame at inference, needing no temporal window at all. The shuffled control holds the frame and the
    set of displacements fixed and permutes who owns which, so a whole-frame drift scores the same in both
    arms and only genuinely LOCAL agreement separates them.
    """

    band_um: tuple[tuple[float, float], ...]
    mean_cosine: tuple[float, ...]
    shuffled_cosine: tuple[float, ...]
    pairs: tuple[int, ...]

    @classmethod
    def of(
        cls,
        graph: TrackGraph,
        positions_um: Float[np.ndarray, "n 3"],
        steps_by_row: dict[int, int],
        seed: int,
    ) -> Self:
        """Compare same-frame neighbour displacement agreement against a permuted control, by separation."""
        bands = tuple(pairwise(band * GATE_UM for band in (0.0, *_COUPLING_BANDS)))
        observed: list[list[float]] = [[] for _ in bands]
        shuffled: list[list[float]] = [[] for _ in bands]
        generator = np.random.default_rng(seed)
        for rows in cls._by_timepoint(graph, steps_by_row):
            directions = cls._unit(np.asarray([steps_by_row[row] for row in rows]))
            cls._accumulate(positions_um[rows], directions, bands, observed)
            cls._accumulate(positions_um[rows], generator.permutation(directions), bands, shuffled)
        return cls(
            band_um=bands,
            mean_cosine=tuple(float(np.mean(values)) if values else float("nan") for values in observed),
            shuffled_cosine=tuple(float(np.mean(values)) if values else float("nan") for values in shuffled),
            pairs=tuple(len(values) for values in observed),
        )

    @staticmethod
    def _by_timepoint(graph: TrackGraph, steps_by_row: dict[int, int]) -> list[Int[np.ndarray, "k"]]:
        """Rows grouped by frame, keeping only those whose displacement is known."""
        timepoints = graph.timepoints()
        known = np.asarray(sorted(steps_by_row), dtype=np.int64)
        return [known[timepoints[known] == t] for t in np.unique(timepoints[known])]

    @staticmethod
    def _unit(vectors: Float[np.ndarray, "k 3"]) -> Float[np.ndarray, "k 3"]:
        """Displacement directions, with zero-length steps left at zero so they contribute no agreement."""
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)

    @staticmethod
    def _accumulate(
        positions_um: Float[np.ndarray, "k 3"],
        directions: Float[np.ndarray, "k 3"],
        bands: tuple[tuple[float, float], ...],
        into: list[list[float]],
    ) -> None:
        """Add every within-band pair's direction cosine to its band."""
        if len(positions_um) < _MIN_COUPLED:
            return
        separation = np.linalg.norm(positions_um[:, None, :] - positions_um[None, :, :], axis=2)
        agreement = directions @ directions.T
        upper = np.triu(np.ones_like(separation, dtype=bool), k=1)
        for index, (low, high) in enumerate(bands):
            selected = upper & (separation >= low) & (separation < high)
            into[index].extend(agreement[selected].tolist())


@dataclass(frozen=True)
class FrameDrift:
    """How much of a frame's motion is one vector every cell shares — and what is left once it is removed.

    This is the control the neighbour coupling demands. Coupling measured at the shuffled level says the
    agreement survives permuting who owns which displacement, which means it is a WHOLE-FRAME drift rather
    than local structure. That distinction decides what the agreement is worth: a shared drift is estimable
    from the current frame alone by averaging over every detection — hundreds of samples against the single
    noisy step a per-track velocity gets — but it also shifts every candidate equally, so it can only help
    where the residual individual motion is what separates the true successor from its decoy.

    `share` is the drift's length against the mean step length. Near 1 the frame moves as a body and
    individual motion is negligible; near 0 the cells move independently and a drift term has nothing to
    subtract. `residual_um` is the median step length after removing it — the motion a linker still has to
    resolve, and the honest measure of what any drift correction leaves behind.
    """

    share: float
    drift_um: float
    residual_um: float

    @classmethod
    def of(cls, graph: TrackGraph, steps_by_row: dict[int, int]) -> Self:
        """Per frame, the mean displacement vector against the mean displacement magnitude."""
        shares: list[float] = []
        drifts: list[float] = []
        residuals: list[float] = []
        for rows in NeighbourCoupling._by_timepoint(graph, steps_by_row):  # noqa: SLF001
            vectors = np.asarray([steps_by_row[row] for row in rows])
            if len(vectors) < _MIN_DRIFT_SAMPLES:
                continue
            drift = vectors.mean(axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            if not lengths.mean():
                continue
            shares.append(float(np.linalg.norm(drift) / lengths.mean()))
            drifts.append(float(np.linalg.norm(drift)))
            residuals.append(float(np.median(np.linalg.norm(vectors - drift, axis=1))))
        return cls(*(float(np.mean(values)) if values else float("nan") for values in (shares, drifts, residuals)))


@dataclass(frozen=True)
class MovieMotion:
    """One movie's motion structure."""

    stem: str
    chains: int
    steps: int
    step_percentiles_um: tuple[float, ...]
    stationary_fraction: float
    mean_turn_cosine: float
    persistence: PersistenceByStep
    displacement: MeanSquaredDisplacement
    coupling: NeighbourCoupling
    drift: FrameDrift

    @classmethod
    def of(cls, stem: str, graph: TrackGraph, spacing: Spacing, max_lag: int, seed: int) -> Self:
        """Measure every motion statistic on one annotated track graph."""
        positions_um = spacing.to_micrometres(graph.positions())
        chains = Chains.of(graph)
        steps = chains.steps(positions_um)
        cosines, weaker = steps.turn_cosines()
        by_row = cls._steps_by_row(chains, positions_um)
        return cls(
            stem=stem,
            chains=len(chains.rows),
            steps=len(steps.vectors),
            step_percentiles_um=tuple(float(v) for v in np.percentile(steps.lengths(), _PERCENTILES)),
            stationary_fraction=steps.stationary_fraction(),
            mean_turn_cosine=float(cosines.mean()) if len(cosines) else float("nan"),
            persistence=PersistenceByStep.of(cosines, weaker, (STATIONARY_UM, 2 * STATIONARY_UM, 4 * STATIONARY_UM)),
            displacement=MeanSquaredDisplacement.of(chains, positions_um, max_lag),
            coupling=NeighbourCoupling.of(graph, positions_um, by_row, seed),
            drift=FrameDrift.of(graph, by_row),
        )

    @staticmethod
    def _steps_by_row(chains: Chains, positions_um: Float[np.ndarray, "n 3"]) -> dict[int, int]:
        """Each row's own outgoing displacement, for the rows that have one."""
        return {
            int(chain[index]): positions_um[chain[index + 1]] - positions_um[chain[index]]  # type: ignore[misc]
            for chain in chains.rows
            for index in range(len(chain) - 1)
        }

    def report(self) -> None:
        """Log this movie's motion, ordered so the gating number reads first."""
        regime = "ballistic" if self.displacement.exponent > (_BALLISTIC + _DIFFUSIVE) / 2 else "diffusive-or-confined"
        logger.info(
            "  %-16s chains %4d steps %5d  MSD exponent %.2f (%s)",
            self.stem,
            self.chains,
            self.steps,
            self.displacement.exponent,
            regime,
        )
        logger.info(
            "    step um p25/p50/p75/p95 %s  stationary %.1f%%",
            "/".join(f"{v:.2f}" for v in self.step_percentiles_um),
            100 * self.stationary_fraction,
        )
        logger.info(
            "    turn cosine %.3f overall; by weaker step %s",
            self.mean_turn_cosine,
            self._counted(self.persistence.mean_cosine, self.persistence.counts),
        )
        logger.info(
            "    frame drift %.2f um = %.0f%% of mean step; residual after removal %.2f um",
            self.drift.drift_um,
            100 * self.drift.share,
            self.drift.residual_um,
        )
        logger.info(
            "    neighbour cosine %s vs shuffled %s",
            self._counted(self.coupling.mean_cosine, self.coupling.pairs),
            "  ".join(f"{value:+.3f}" for value in self.coupling.shuffled_cosine),
        )

    @staticmethod
    def _counted(values: tuple[float, ...], counts: tuple[int, ...]) -> str:
        """Per-band means with the sample size beside each — a band's mean is unreadable without its n."""
        return "  ".join(f"{value:+.3f}(n={count})" for value, count in zip(values, counts, strict=True))


def main() -> None:
    """Measure the structure of annotated cell motion — the gate on a wider temporal window."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Persistence, scaling and neighbour coupling of cell motion.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    proxy = TestMovieProxy.load(DataRoot.from_config(args.config), CV_MOVIES)
    logger.info("motion structure over %d annotated movies (ballistic = 2.0, diffusive = 1.0)", len(proxy.paths))
    for path, truth in zip(proxy.paths, proxy.truths, strict=True):
        MovieMotion.of(path.stem, truth.graph, proxy.spacing, args.max_lag, args.seed).report()


if __name__ == "__main__":
    main()

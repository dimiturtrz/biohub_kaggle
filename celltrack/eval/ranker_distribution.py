"""Is the ported ranker being fed the distribution it was FITTED on? — a per-feature drift table.

The CC0 local-association re-ranker carries 0.85 of the association evidence in a reproducible public 0.915
pipeline, and priced into our flow linker at that ratio it costs us 0.023 — monotone in its weight, so it is
not a tuning problem: the net's probabilities are actively misleading our cost. There are exactly two
explanations, and they have opposite consequences:

* **(A) our features are off-distribution.** One or more of the 22 is computed in the wrong units, off the
  wrong graph, or under a different convention than the artifact was trained with. The MLP then receives a
  vector unlike anything it saw, and its output is noise dressed as a probability. A bug — fixable.
* **(B) the ranker is faithful and does not transfer.** It was fitted on the candidate lists of a per-frame
  Hungarian over motion-predicted distances; we ask it about the arcs of a global min-cost flow over raw
  distances. A finding — and the artifact's own manifest already warns of it ("use only as a constrained
  local association tie-breaker, not as a global edge veto").

The two are told apart by measurement, not argument, because the artifact ships its own training-time
normalisation: `mean` and `std` tensors inside `local_association_ranker.pt` ARE the per-feature training
distribution (`local_association_features.stats.json` holds only row counts). So this module computes our 22
feature vectors on the real gated candidates of a real movie — off the tracker's OWN first linking pass, the
same context `CellTracker.fused_affinities` + `LinkerConfig.without_ranker` produce for pass 2 — and prints
each feature's own mean and std beside the artifact's, with the gap standardised by the training std. A
faithfully computed feature sits a fraction of a training std away; one in the wrong units, off the wrong
graph, or constant where it should vary lands many stds out, and says so by name.

*** WHY THE POPULATION IS PRINTED BEFORE THE TABLE. *** Following `shortcut_diagnosis`: a drift number is a
statement about a population, and this one has a specific degenerate mode. If the gate admits ~one candidate
per source, `candidate_rank_dist` is 1 and `candidate_count` is 1 for nearly every row BY CONSTRUCTION, and
their "drift" measures the gate rather than the port; the feature means over a handful of rows likewise move
with the sample, not the code. So the row count, the source count and the rows-per-source are printed first
and the table is refused outright when they cannot carry it. The artifact's own training population is
printed on the same line as the reference every one of our numbers is read against.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity
from celltrack.edges.association_ranker import AssociationContext, AssociationFeatures, AssociationRanker
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.proxy_eval import RANKER_ARTIFACT, TrackerProxyEval
from celltrack.linkers.linkers import LinkerConfig
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The artifact's own training population, from `local_association_features.stats.json`: 385648 candidate rows
# over 126705 groups, a group being one source's gated candidate list. Their ratio is the reference our
# rows-per-source is read against — it is what "the list this net was taught to choose from" was SIZED like,
# and a port whose lists are a different size is asking the net a differently-shaped question whatever the
# individual features say.
_TRAINING_ROWS = 385648
_TRAINING_GROUPS = 126705
_TRAINING_ROWS_PER_SOURCE = _TRAINING_ROWS / _TRAINING_GROUPS

# A source whose gated list holds one candidate poses no choice: `candidate_rank_dist` and `candidate_count`
# are then 1 by construction, `has_learned_edge` is whatever pass 1 did with the only option, and the drift
# those columns show is the gate's, not the port's. 1.05 admits a population where at most one source in
# twenty is in that state.
_MIN_ROWS_PER_SOURCE = 1.05

# The standard error of a mean over `k` rows is (std / sqrt k) training stds, so 500 rows puts it at 0.045 —
# an order below the one-std bar a feature is FLAGGED at, i.e. small enough that a flag is the code and not
# the sample. Below that the table would be reporting its own noise.
_MIN_ROWS = 500

# A feature whose mean sits a full training std from the training mean is, for the net, an input it saw in
# the tail rather than the body of its training data — the point at which "computed differently" stops being
# a quibble. One std rather than two because 22 features each shifted half a std compose into a vector far
# outside anything the net was fitted on, and this table exists to catch that.
_OFF_DISTRIBUTION_Z = 1.0

# Spread is the other half of the same question: a feature can share the training mean and still be wrong if
# it barely varies (a constant column) or varies wildly (a units error that also happens to centre). The band
# is half to double the training std — outside it, our column is not the same random variable as theirs.
_MIN_SPREAD_RATIO = 0.5
_MAX_SPREAD_RATIO = 2.0

# A zero-variance training column would divide by zero; the artifact's own loader floors its std here too.
_STD_FLOOR = 1e-6

# The artifact reports a 0.9777 validation score, so in-distribution it separates decisively. Its output
# distribution is therefore diagnostic on its own: the share above this, and the spread of the probabilities,
# say whether the net is deciding anything on our rows or emitting a near-constant.
_DECISIVE_PROBABILITY = 0.5

_QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)

# The processed dataset the artifact and the movies both live under.
_DATASET = "biohub_cell_tracking"


@dataclass(frozen=True)
class FeaturePopulation:
    """The gated candidate rows our pass-1 context actually presents to the ranker, as the net receives them.

    `values` is exactly what `AssociationRanker.probabilities` is handed — the manifest-ordered, RAW (not yet
    standardised) feature matrix — so a mean computed here is directly comparable with the artifact's stored
    training mean. `groups` counts the sources that contributed at least one gated candidate, which is the
    artifact's "group": the rows-per-source ratio is a fact about the candidate lists, not about the movie.
    """

    names: tuple[str, ...]
    values: Float[np.ndarray, "k f"]
    groups: int

    @classmethod
    def of(
        cls, ranker: AssociationRanker, context: AssociationContext, gate_um: float, max_gaps: int | None = None
    ) -> "FeaturePopulation":
        """Every gap's gated candidate rows, built through the ranker's own feature code — never a re-implementation.

        `AssociationContext` and `AssociationFeatures` are the SHIPPED port; measuring a private copy of them
        would answer whether this file computes the features correctly, which is not the question. The context
        is taken already built rather than as its four ingredients, because it is precisely the bundle the
        ranker itself is handed. `max_gaps` truncates the frame gaps for a cheap read on a long movie.
        """
        blocks: list[Float[np.ndarray, "k f"]] = []
        groups = 0
        for timepoint in np.unique(context.timepoints)[:-1][:max_gaps].tolist():
            sources, targets = context.rows_at(int(timepoint)), context.rows_at(int(timepoint) + 1)
            if not len(sources) or not len(targets):
                continue
            features = AssociationFeatures.of(context, sources, targets, int(timepoint), gate_um)
            gated = features.gated()
            if not gated.any():
                continue
            blocks.append(features.matrix(ranker.feature_names))
            groups += int(gated.any(axis=1).sum())
        return cls(names=ranker.feature_names, values=cls._stack(blocks, len(ranker.feature_names)), groups=groups)

    @classmethod
    def pooled(cls, populations: Sequence["FeaturePopulation"]) -> "FeaturePopulation":
        """Several movies' candidate rows as one population — what the pooled table is computed over."""
        if not populations:
            raise ValueError("a pooled feature population needs at least one movie to take its feature names from")
        names = populations[0].names
        mismatched = [population.names for population in populations if population.names != names]
        if mismatched:
            raise ValueError("pooling populations built from different feature orders would permute the table")
        return cls(
            names=names,
            values=np.concatenate([population.values for population in populations], axis=0),
            groups=sum(population.groups for population in populations),
        )

    def size(self) -> int:
        """How many gated candidate rows the population holds — the artifact's "rows"."""
        return int(self.values.shape[0])

    def rows_per_source(self) -> float:
        """Candidates per source list, the artifact's rows/groups — NaN when no source contributed at all."""
        return float(self.size() / self.groups) if self.groups else float("nan")

    def is_reportable(self) -> bool:
        """Whether the population can carry a drift table at all — the guard the whole report sits behind."""
        return self.size() >= _MIN_ROWS and self.rows_per_source() >= _MIN_ROWS_PER_SOURCE

    def moments(self) -> tuple[Float[np.ndarray, "f"], Float[np.ndarray, "f"]]:
        """Our per-feature mean and std — the two numbers compared against the artifact's stored pair."""
        if not self.size():
            empty = np.full(len(self.names), np.nan, dtype=np.float64)
            return empty, empty
        return self.values.mean(axis=0), self.values.std(axis=0)

    @staticmethod
    def _stack(blocks: list[Float[np.ndarray, "k f"]], features: int) -> Float[np.ndarray, "k f"]:
        """The gaps' rows as one matrix, keeping the feature width when no gap produced a single gated pair."""
        return np.concatenate(blocks, axis=0) if blocks else np.empty((0, features), dtype=np.float64)


@dataclass(frozen=True)
class FeatureDrift:
    """One feature's own mean and spread beside the training distribution the artifact's weights were fitted to.

    The standardised gap is divided by the TRAINING std, not by ours: the net standardises its input with the
    training std, so a mean displaced by one training std is displaced by exactly one unit of the net's own
    input scale — the quantity the first `Linear` actually sees.
    """

    name: str
    ours_mean: float
    ours_std: float
    training_mean: float
    training_std: float

    @classmethod
    def table(cls, population: FeaturePopulation, ranker: AssociationRanker) -> tuple["FeatureDrift", ...]:
        """Every feature's drift, worst first — the manifest's own column order resolves the pairing."""
        ours_mean, ours_std = population.moments()
        drifts = [
            cls(
                name=name,
                ours_mean=float(ours_mean[index]),
                ours_std=float(ours_std[index]),
                training_mean=float(ranker.mean[index]),
                training_std=float(ranker.std[index]),
            )
            for index, name in enumerate(population.names)
        ]
        return tuple(sorted(drifts, key=lambda drift: -drift.z()))

    def z(self) -> float:
        """How many TRAINING stds our mean sits from theirs — the net's own input scale, signed toward ours."""
        return abs(self.ours_mean - self.training_mean) / max(self.training_std, _STD_FLOOR)

    def spread(self) -> float:
        """Our std as a multiple of theirs — 1 is the same random variable, 0 a column that does not vary."""
        return self.ours_std / max(self.training_std, _STD_FLOOR)

    def is_off_distribution(self) -> bool:
        """Whether this column is a different quantity than the one trained on — by location OR by spread."""
        return self.z() >= _OFF_DISTRIBUTION_Z or not (_MIN_SPREAD_RATIO <= self.spread() <= _MAX_SPREAD_RATIO)

    def line(self) -> str:
        """The table row: both means, both stds, the standardised gap, the spread ratio, and a flag."""
        return (
            f"  {'!!' if self.is_off_distribution() else '  '} {self.name:<22} "
            f"ours {self.ours_mean:>10.4f} +- {self.ours_std:>9.4f}   "
            f"trained {self.training_mean:>10.4f} +- {self.training_std:>9.4f}   "
            f"z={self.z():>8.2f}  spread={self.spread():>6.2f}"
        )


@dataclass(frozen=True)
class RankerOutput:
    """The ranker's probability on our rows — decisive on data it recognises, flat or saturated on data it does not."""

    probability: Float[np.ndarray, "k"]

    @classmethod
    def of(cls, ranker: AssociationRanker, population: FeaturePopulation) -> "RankerOutput":
        """Score the population's rows exactly as the linker's cost term does."""
        return cls(probability=ranker.probabilities(population.values))

    def decisive_fraction(self) -> float:
        """The share of candidates the net calls a successor — NaN on an empty population."""
        if not self.probability.size:
            return float("nan")
        return float(np.mean(self.probability > _DECISIVE_PROBABILITY))

    def line(self) -> str:
        """The output distribution as quantiles plus the decisive share — a flat one is itself the diagnosis."""
        if not self.probability.size:
            return "  P(ranker): no rows scored"
        quantiles = np.quantile(self.probability, _QUANTILES)
        spread = "  ".join(f"p{int(q * 100):02d}={value:.4f}" for q, value in zip(_QUANTILES, quantiles, strict=True))
        return (
            f"  P(ranker): {spread}  mean={self.probability.mean():.4f}  "
            f"frac>{_DECISIVE_PROBABILITY:g}={self.decisive_fraction():.4f}"
        )


@dataclass(frozen=True)
class DriftReport:
    """One population's read-out: the counts FIRST, then the drift table and the ranker's output — or the refusal.

    The refusal is as much the deliverable as the table. A drift number computed over a hundred rows, or over
    lists holding one candidate each, reads to a later reader exactly like a verdict on the port, so nothing
    downstream of a failed guard is printed.
    """

    label: str
    population: FeaturePopulation
    ranker: AssociationRanker

    def lines(self) -> list[str]:
        """The report as log lines: population line first, then either the drift block or the stated refusal."""
        head = self._population()
        if not self.population.is_reportable():
            return [head, f"  REFUSED: {self._refusal()}"]
        drifts = FeatureDrift.table(self.population, self.ranker)
        off = [drift.name for drift in drifts if drift.is_off_distribution()]
        return [
            head,
            *[drift.line() for drift in drifts],
            f"  off-distribution ({len(off)}/{len(drifts)}): {', '.join(off) if off else 'none'}",
            RankerOutput.of(self.ranker, self.population).line(),
        ]

    def _population(self) -> str:
        """The counts every drift number has to be read against — always emitted before that number."""
        return (
            f"{self.label}: {self.population.size()} gated candidate rows over {self.population.groups} sources "
            f"({self.population.rows_per_source():.3f} per source; the artifact trained on "
            f"{_TRAINING_ROWS} rows / {_TRAINING_GROUPS} groups = {_TRAINING_ROWS_PER_SOURCE:.3f})"
        )

    def _refusal(self) -> str:
        """Why this population gets no table — stated as the defect, so the refusal is actionable, not a shrug."""
        if self.population.size() < _MIN_ROWS:
            return (
                f"{self.population.size()} rows is under {_MIN_ROWS}, so a per-feature mean carries more "
                "sampling error than the drift it would be reporting (see the module docstring)"
            )
        return (
            f"{self.population.rows_per_source():.3f} candidates per source is under {_MIN_ROWS_PER_SOURCE}: "
            "the gate poses almost no choice, so the candidate-list columns are constant by construction and "
            "their drift would measure the gate, not the port"
        )


class AffinityFuser(Protocol):
    """The one thing this diagnosis needs a mounted tracker for beyond its detections — `fused_affinities`.

    A protocol rather than `CellTracker` for the same reason `TestMovieProxy` takes a `LinkingPipeline`: the
    diagnosis depends on the rule it must share with the tracker, not on the tracker's whole recipe, and the
    fake that exercises its wiring in a unit test then needs no models.
    """

    def fused_affinities(self, path: Path, nodes: TrackGraph, affinity: EdgeAffinity | None) -> EdgeAffinity | None:
        """The bidirectionally fused probabilities the linker's agreement knobs read — `None` when unused."""
        ...


@dataclass(frozen=True)
class ScoredMovie:
    """One movie's detections and every affinity + geometry the context linking reads — a tracker pass, unpacked.

    The fused affinity is resolved through the tracker's OWN `fused_affinities` rather than re-derived here: a
    second copy of that rule would let the reconstruction link under a different cost than the pipeline does,
    which is precisely the failure a drift measurement cannot survive.
    """

    detections: TrackGraph
    affinity: EdgeAffinity | None
    mutual: EdgeAffinity | None
    spacing: Spacing
    volume_shape: tuple[int, int, int] | None

    @classmethod
    def of(
        cls, tracker: AffinityFuser, path: Path, detections: TrackGraph, affinity: EdgeAffinity | None
    ) -> "ScoredMovie":
        """Unpack one already-run tracker pass into the linking inputs, fusing the affinity by the tracker's rule."""
        video = CellVideo.from_ome_zarr(path)
        return cls(
            detections=detections,
            affinity=affinity,
            mutual=tracker.fused_affinities(path, detections, affinity),
            spacing=video.spacing,
            volume_shape=video.volume_shape,
        )


@dataclass(frozen=True)
class RankerDriftDiagnosis:
    """Reading the feature vectors the tracker's own FIRST pass hands the re-ranker, on one movie.

    The context linking here is not a stand-in for the tracker's: it is built from the same linker config
    through `LinkerConfig.without_ranker` and the same fused affinity through `CellTracker.fused_affinities`,
    which is exactly what the tracker does before it scores the ranker. A diagnosis that linked under a
    different cost would characterise a population the pipeline never scores.
    """

    linker: LinkerConfig
    max_gaps: int | None = None

    def pass_one(self) -> LinkerConfig:
        """The cost the CONTEXT graph is linked under — the arm minus the term that cannot exist yet.

        Dropping the ranker bonus is also what lets the tracker run here without the re-ranker mounted into it:
        the pass this diagnosis reads is by definition the one that ran before any ranker probability existed.
        """
        return self.linker.without_ranker()

    def of_movie(self, ranker: AssociationRanker, movie: ScoredMovie) -> FeaturePopulation:
        """One movie's gated candidate rows, off the pass-1 linking of a single detection + affinity pass."""
        config = self.pass_one()
        linker = config.build(movie.spacing, movie.affinity, movie.mutual, movie.volume_shape)
        context = AssociationContext.of(movie.spacing, movie.detections, linker.link(movie.detections), movie.affinity)
        return FeaturePopulation.of(ranker, context, config.gate_um, self.max_gaps)

    def by_movie(self, ranker: AssociationRanker, movies: Mapping[str, ScoredMovie]) -> dict[str, FeaturePopulation]:
        """Each movie's candidate population, keyed by stem — one detection pass per movie, one mounted tracker."""
        return {stem: self.of_movie(ranker, movie) for stem, movie in movies.items()}


def main() -> None:
    """Report, per movie and pooled, how far our 22 ranker features sit from the distribution they were fitted on."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Are the ported ranker's features on their training distribution?")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", action="append", dest="movies", help="movie stem (repeatable)")
    # Required rather than defaulted: the population is a fact about the DETECTIONS, so the arm's operating
    # point is named at the call rather than inherited from a default that may not be the arm's.
    parser.add_argument("--threshold", type=float, required=True, help="the arm's detection threshold")
    parser.add_argument(
        "--disappearance", type=float, default=0.0, help="linker disappearance cost (the flow linker's boundary cost)"
    )
    parser.add_argument("--max-gaps", type=int, help="read only the first N frame gaps of each movie")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="any tracker knob, as in proxy_eval (repeatable); the arm's linker is what defines the population",
    )
    args = parser.parse_args()

    sweep = TrackerProxyEval(args.device, (args.threshold,), (args.disappearance,), overrides=tuple(args.overrides))
    arm = sweep.config_at(args.threshold, args.disappearance)
    diagnosis = RankerDriftDiagnosis(linker=arm.linker, max_gaps=args.max_gaps)
    root = DataRoot.from_config(args.config)
    proxy = TestMovieProxy.load(root, tuple(args.movies or (DENSE_MOVIE,)))
    ranker = AssociationRanker.from_artifact(root.processed(_DATASET) / RANKER_ARTIFACT)
    tracker = DenseDiagnosis.shipped(root, args.device).with_config(replace(arm, linker=diagnosis.pass_one()))
    passes = {path.stem: (path, tracker.run_scored(path.name, path)) for path in proxy.paths}
    movies = {
        stem: ScoredMovie.of(tracker, path, scored.detections, scored.affinity)
        for stem, (path, scored) in passes.items()
    }
    by_movie = diagnosis.by_movie(ranker, movies)
    logger.info("threshold=%g  set=%s  linker=%s", args.threshold, list(args.overrides), diagnosis.pass_one())
    reports = [DriftReport(label=stem, population=population, ranker=ranker) for stem, population in by_movie.items()]
    reports.append(
        DriftReport(label="POOLED", population=FeaturePopulation.pooled(list(by_movie.values())), ranker=ranker)
    )
    for line in [line for report in reports for line in report.lines()]:
        logger.info("%s", line)


if __name__ == "__main__":
    main()

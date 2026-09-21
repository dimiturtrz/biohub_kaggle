"""The NON-INVERTING local ruler: the OFFICIAL competition metric on the density-spanning test movies.

The cached proxy saturates and INVERTS above ~0.90 — dw0 scored +0.0267 held-out and −0.028 on the
leaderboard — so a top-end proxy number is not evidence. This runs the official metric itself
(`biohub_tracking.metrics.evaluate`, max_distance 7µm, the dataset's own scale) and moves monotonically
with the leaderboard.

It is a RELATIVE ruler, not an absolute one: the champion's own geffs score 0.8463 here against its LB
0.924, because these six movies are the dense end of the corpus. Compare arms against that 0.8463, never
against a leaderboard number.

Read the DENSE rows, not just the micro: micro is dominated by the sparse movies that already sit near
1.0, while the whole open gap lives in the two dense ones.

The metric and the graph loader are injected. This module owns the accumulation and the score, not the
donor's evaluation schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

DIVISION_WEIGHT = 0.1
MAX_MATCH_DISTANCE_UM = 7.0
COMPETITION = "biohub_cell_tracking"
CHAMPION_PREDICTIONS = "kaggle/biohub-tracking-support-pack-50ep-v1/repo/predictions/local/unet_transformer/split_0"

# The six test movies of the pmkf split, in ascending density. The two dense ones carry the gap.
TEST_MOVIES = (
    "6bba_718b21f9",
    "6bba_784a78c9",
    "6bba_bb9f20c3",
    "6bba_09961292",
    "44b6_ddf577ad",
    "6bba_57b7cc1e",
)


@dataclass(frozen=True)
class MetricCounts:
    """True/false positives and false negatives of one axis, for one movie or summed over several."""

    tp: int
    fp: int
    fn: int

    def __add__(self, other: MetricCounts) -> MetricCounts:
        return MetricCounts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)

    def jaccard(self) -> float:
        """TP / (TP + FP + FN), NaN on an empty axis — the official metric's own convention."""
        denominator = self.tp + self.fp + self.fn
        return self.tp / denominator if denominator > 0 else float("nan")

    def is_empty(self) -> bool:
        return self.tp + self.fp + self.fn == 0


@dataclass(frozen=True)
class TrackingScore:
    """Edge and division agreement — one movie's row, or the micro-average over a set of them."""

    edges: MetricCounts
    divisions: MetricCounts

    @classmethod
    def micro(cls, scores: Iterable[TrackingScore]) -> TrackingScore:
        """Summed counts, then one ratio — NOT the mean of per-movie ratios.

        A macro average would let the sparse movies, which already score near 1.0, outvote the dense ones
        that hold the entire gap.
        """
        total = cls(edges=MetricCounts(0, 0, 0), divisions=MetricCounts(0, 0, 0))
        for score in scores:
            total = TrackingScore(edges=total.edges + score.edges, divisions=total.divisions + score.divisions)
        return total

    def value(self) -> float:
        """The competition score. Division agreement is unweighted away when no movie has a division."""
        if self.divisions.is_empty():
            return self.edges.jaccard()
        return self.edges.jaccard() + DIVISION_WEIGHT * self.divisions.jaccard()


@dataclass(frozen=True)
class OfficialRuler:
    """Scores a predictor's tracks movie by movie, and micro-averages the counts.

    `score_movie` returns a movie's counts, or None when that movie has no prediction to score — a
    missing movie is skipped rather than counted as a total miss, so a partial run stays readable.
    """

    movies: Sequence[str]
    score_movie: Callable[[str], TrackingScore | None]

    def rows(self) -> list[tuple[str, TrackingScore]]:
        """One pass. The caller micro-averages what comes back — scoring a movie twice is not free."""
        scored = ((stem, self.score_movie(stem)) for stem in self.movies)
        return [(stem, score) for stem, score in scored if score is not None]


def main() -> None:
    import argparse  # noqa: PLC0415
    import logging  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import tracksdata as td  # noqa: PLC0415
    from biohub_tracking.io import open_dataset  # type: ignore[missing-import]  # noqa: PLC0415
    from biohub_tracking.metrics import evaluate  # type: ignore[missing-import]  # noqa: PLC0415

    from core.paths import DataRoot  # noqa: PLC0415

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = DataRoot.from_config(Path("paths.yaml"))
    champion = data.processed(COMPETITION) / CHAMPION_PREDICTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=champion, help="dir of <stem>.geff to score")
    arguments = parser.parse_args()
    ground_truth = data.raw(COMPETITION) / "train"

    def score_movie(stem: str) -> TrackingScore | None:
        geff = arguments.pred_dir / f"{stem}.geff"
        if not geff.exists():
            log.info("%s: MISSING from %s -- skipped", stem, arguments.pred_dir.name)
            return None
        predicted = td.graph.IndexedRXGraph.from_geff(geff)
        predicted = predicted[0] if isinstance(predicted, tuple) else predicted
        dataset = open_dataset(ground_truth / stem, require_tracks=True)
        result = evaluate(predicted, dataset.tracks, scale=dataset.scale, max_distance=MAX_MATCH_DISTANCE_UM)
        return TrackingScore(
            edges=MetricCounts(tp=result.edge_tp, fp=result.edge_fp, fn=result.edge_fn),
            divisions=MetricCounts(tp=result.division_tp, fp=result.division_fp, fn=result.division_fn),
        )

    rows = OfficialRuler(movies=TEST_MOVIES, score_movie=score_movie).rows()
    for stem, score in rows:
        log.info(
            "%s: edge_jac=%.4f eTP=%d eFP=%d eFN=%d divTP=%d divFP=%d divFN=%d",
            stem,
            score.edges.jaccard(),
            score.edges.tp,
            score.edges.fp,
            score.edges.fn,
            score.divisions.tp,
            score.divisions.fp,
            score.divisions.fn,
        )
    micro = TrackingScore.micro(score for _, score in rows)
    log.info(
        "MICRO: edge_jaccard=%.4f division_jaccard=%.4f score=%.4f  (champion scores 0.8463 here)",
        micro.edges.jaccard(),
        micro.divisions.jaccard(),
        micro.value(),
    )


if __name__ == "__main__":
    main()

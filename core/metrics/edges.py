"""Edge true/false positives and negatives against a sparse ground truth.

The competition's ground truth annotates a few percent of the cells, so most predicted links have no
annotated counterpart and are simply *ignored*: a predicted edge only enters the tally when one of its
endpoints matches a ground-truth node that the annotation actually continues through. Over-linking in
unannotated tissue therefore costs nothing here — the only price for predicting too much is the node-count
adjustment in `score.py`.

Before counting, the predicted edges are put through the same four repairs the reference implementation
applies, each closing a way of inflating the intersection: duplicate links collapse to one, links that
skip or reverse a timepoint are dropped, several links landing on one ground-truth link keep only the
first, and a node may keep at most the two children a division allows.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from jaxtyping import Bool, Int

from core.data.tracks import TrackGraph
from core.metrics.matching import UNMATCHED, NodeMatching

_MAX_CHILDREN = 2
_SOURCE, _TARGET = "source", "target"
_EDGE_ID, _MATCHED = "edge_id", "matched"
_MATCHED_SOURCE, _MATCHED_TARGET = "matched_source", "matched_target"


@dataclass(frozen=True)
class EdgeCounts:
    """The confusion counts the edge Jaccard is built from."""

    tp: int
    fp: int
    fn: int

    @classmethod
    def of(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching) -> "EdgeCounts":
        """Tally predicted links against the annotated ones under an established node matching."""
        annotated_links = len(truth.edge_rows())
        if len(prediction.edges) == 0:
            return cls(tp=0, fp=0, fn=annotated_links)

        links = cls._repaired(prediction, truth, matching)
        recovered = int(links[_MATCHED].sum())
        countable = int(cls._countable(links, truth).sum())
        return cls(tp=recovered, fp=countable - recovered, fn=annotated_links - recovered)

    @classmethod
    def pooled(cls, counts: Iterable["EdgeCounts"]) -> "EdgeCounts":
        """Sum counts across videos, so their Jaccard micro-averages the way the leaderboard does."""
        summed = cls(tp=0, fp=0, fn=0)
        for count in counts:
            summed = cls(tp=summed.tp + count.tp, fp=summed.fp + count.fp, fn=summed.fn + count.fn)
        return summed

    def jaccard(self) -> float:
        """`TP / (TP + FP + FN)`, or NaN when the ground truth and the prediction are both empty."""
        total = self.tp + self.fp + self.fn
        return self.tp / total if total > 0 else float("nan")

    def weight(self) -> int:
        """How much this sample counts for when adjusted Jaccards are averaged across videos."""
        return self.tp + self.fp + self.fn

    @classmethod
    def _repaired(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching) -> pl.DataFrame:
        """The predicted links that are eligible to be counted, after the four repairs."""
        edges = prediction.edge_rows()
        matched_gt = matching.gt_rows
        links = pl.DataFrame(
            {
                _EDGE_ID: np.arange(len(edges)),
                _SOURCE: edges[:, 0],
                _TARGET: edges[:, 1],
                _MATCHED_SOURCE: matched_gt[edges[:, 0]],
                _MATCHED_TARGET: matched_gt[edges[:, 1]],
                _MATCHED: cls._lands_on_an_annotated_link(edges, truth, matched_gt),
            }
        )
        deduplicated = links.sort(_MATCHED, descending=True).unique(
            subset=[_SOURCE, _TARGET], keep="first", maintain_order=True
        )
        consecutive = deduplicated.filter(cls._spans_one_timepoint(deduplicated, prediction))
        return cls._capped_out_degree(cls._collapsed_merges(consecutive))

    @staticmethod
    def _lands_on_an_annotated_link(
        edges: Int[np.ndarray, "e 2"],
        truth: TrackGraph,
        matched_gt: Int[np.ndarray, "n"],
    ) -> Bool[np.ndarray, "e"]:
        """Whether each predicted link lands exactly on an annotated link once both ends are translated."""
        annotated = truth.edge_rows()
        stride = max(len(truth.node_ids), 1)
        translated = matched_gt[edges]
        both_matched = (translated != UNMATCHED).all(axis=1)
        codes = np.where(both_matched, translated[:, 0] * stride + translated[:, 1], UNMATCHED)
        return np.isin(codes, annotated[:, 0] * stride + annotated[:, 1]) & both_matched

    @staticmethod
    def _spans_one_timepoint(links: pl.DataFrame, prediction: TrackGraph) -> Bool[np.ndarray, "e"]:
        """Backward links and links that skip a frame are not links the metric recognises."""
        timepoints = prediction.timepoints()
        return timepoints[links[_TARGET].to_numpy()] - timepoints[links[_SOURCE].to_numpy()] == 1

    @staticmethod
    def _collapsed_merges(links: pl.DataFrame) -> pl.DataFrame:
        """Several predicted links mapping onto one annotated link count once — the earliest of them."""
        both_matched = (pl.col(_MATCHED_SOURCE) != UNMATCHED) & (pl.col(_MATCHED_TARGET) != UNMATCHED)
        duplicate = both_matched & (pl.col(_EDGE_ID) != pl.col(_EDGE_ID).min().over(_MATCHED_SOURCE, _MATCHED_TARGET))
        return links.filter(~duplicate)

    @staticmethod
    def _capped_out_degree(links: pl.DataFrame) -> pl.DataFrame:
        """A cell has at most two daughters, so a node's extra outgoing links are not evaluable."""
        return links.filter(pl.col(_EDGE_ID).rank("ordinal").over(_SOURCE) <= _MAX_CHILDREN)

    @staticmethod
    def _countable(links: pl.DataFrame, truth: TrackGraph) -> Bool[np.ndarray, "e"]:
        """A link is countable when the annotation continues through one of its endpoints.

        Anywhere else the ground truth is silent about whether the link is right, so the metric ignores
        it rather than guessing — this is what makes recall in unannotated tissue free.
        """
        annotated = truth.edge_rows()
        count = len(truth.node_ids)
        continues = {
            _MATCHED_SOURCE: np.bincount(annotated[:, 0], minlength=count) > 0,
            _MATCHED_TARGET: np.bincount(annotated[:, 1], minlength=count) > 0,
        }
        source, target = (
            EdgeCounts._through_an_annotated_node(links[endpoint].to_numpy(), carries_on)
            for endpoint, carries_on in continues.items()
        )
        return source | target

    @staticmethod
    def _through_an_annotated_node(
        matched_rows: Int[np.ndarray, "e"],
        carries_on: Bool[np.ndarray, "g"],
    ) -> Bool[np.ndarray, "e"]:
        """Whether each endpoint matched an annotated node the ground truth carries on through."""
        matched = matched_rows != UNMATCHED
        return matched & carries_on[np.where(matched, matched_rows, 0)]

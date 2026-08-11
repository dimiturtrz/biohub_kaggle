"""The public CC0 local-association re-ranker — a 22-feature MLP scoring each gated candidate link.

The affinity transformer answers one question about a pair (from this target's frame, which source is the
parent?) out of the image. This ranker answers a different one out of the LINKING STATE: given the primary
link graph, the local crowding, the candidate list this source is choosing from, and where a constant-velocity
extrapolation says the source should have landed — is this particular candidate the true successor? It reads
no pixels; every one of its 22 inputs is derived from detections, the primary links, and the affinity.

Why it is worth porting rather than out-thinking: the public 0.915 notebook mounts the same detector and edge
packs we mount and scores 0.020 above us, and its association cost is
`motion_dist + 0.05*raw_dist - 0.75*(0.85*ranker_prob + 0.15*edge_prob)`. The transformer we have spent the
campaign improving survives there as ONE FEATURE at weight 0.15; this net carries 0.85 of the learned
association signal. Two of its features (`motion_dist_um`, `motion_gain_um`) are motion — which is not a
contradiction of our refuted `MotionHungarianLinker`: what was refuted is a HAND-CODED velocity model
overriding the cost, not motion as evidence a learned scorer weighs against 21 other facts.

The feature contract comes from the artifact's own manifest, never from a hard-coded list: every public
notebook carries a `_RANKER_FALLBACK_FEATURES` constant that is a stale guess (eight names wrong) which their
own loader overrides with the artifact metadata. `AssociationRanker.from_artifact` reads the manifest, so a
mismatch is an error rather than a silently permuted feature vector.

Two divergences from the notebook are deliberate and load-bearing:

* **Architecture.** The checkpoint's `state_dict` carries `net.1.{weight,bias}` of shape `(64,)` — a
  `LayerNorm` after the first `Linear`. The public loader collects only 2-D weight tensors and therefore
  DROPS that normalisation, running the trained weights through an architecture they were not trained in.
  We rebuild the real module and load `strict=True`, so the reconstruction is checked rather than assumed.
* **Where the motion prediction comes from.** Their relink sweeps frames in order and predicts from the
  predecessor IT just matched; a global min-cost flow has no such sequential state. Ours predicts from the
  parent in the CONTEXT graph — the primary linking we are re-ranking — which is the same quantity ("the
  source's previous position under the current best link set") sourced from the pass we actually run.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from celltrack.affinity import EdgeAffinity
from celltrack.models.edge_transformer import PrecomputedEdgeAffinity
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_MANIFEST_NAME = "ASSOCIATION_RANKER_MANIFEST.json"
# The radius the `*_density_7um` features count neighbours in — named by the feature itself, so it is the
# artifact's constant, not a choice of ours.
_DENSITY_RADIUS_UM = 7.0
# Damping of the constant-velocity extrapolation the motion features measure against: the prediction is
# `position + weight * (position - previous_position)`. 0.5 is the public pipeline's
# `MOTION_RELINK_VELOCITY_WEIGHT`, i.e. the value the ranker's own training features were computed with —
# changing it would move the inputs off the distribution the net was fitted on.
_VELOCITY_WEIGHT = 0.5
# A zero-variance feature would divide by zero; the public loader floors the std at the same value.
_STD_FLOOR = 1e-6


@dataclass(frozen=True)
class AssociationContext:
    """The per-video facts the 22 features read that a single frame gap does not carry.

    Built once per video from the detections, the PRIMARY link graph being re-ranked, and the edge affinity.
    The link graph is what makes the degree, `has_learned_edge`, `*_has_*` and `target_best_next_prob`
    features meaningful: in the public pipeline they are all derived from the already-selected edge list
    (`learned_edge_probs` is built from `edges`, the emitted links — not from every scored candidate), so a
    port that fed them the dense affinity instead would feed the net a feature that is constant by
    construction and shifted off its training distribution.
    """

    positions_um: Float[np.ndarray, "n 3"]
    timepoints: Int[np.ndarray, "n"]
    row_in_frame: Int[np.ndarray, "n"]
    frame_count: Int[np.ndarray, "n"]
    density: Float[np.ndarray, "n"]
    in_degree: Int[np.ndarray, "n"]
    out_degree: Int[np.ndarray, "n"]
    best_next_prob: Float[np.ndarray, "n"]
    predicted_um: Float[np.ndarray, "n 3"]
    link_sources: Int[np.ndarray, "e"]
    link_targets: Int[np.ndarray, "e"]
    link_prob: Float[np.ndarray, "e"]
    link_scored: Bool[np.ndarray, "e"]
    max_timepoint: int

    @classmethod
    def of(
        cls, spacing: Spacing, detections: TrackGraph, links: TrackGraph, affinity: EdgeAffinity | None
    ) -> "AssociationContext":
        """Every per-node quantity the features need, measured off one video's detections and primary links."""
        positions_um = spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        row_in_frame, frame_count = _frame_positions(timepoints)
        sources, targets = _link_rows(detections, links)
        probability, scored = _link_probabilities(timepoints, row_in_frame, sources, targets, affinity)
        count = len(timepoints)
        return cls(
            positions_um=positions_um,
            timepoints=timepoints,
            row_in_frame=row_in_frame,
            frame_count=frame_count,
            density=_frame_density(positions_um, timepoints),
            in_degree=np.bincount(targets, minlength=count),
            out_degree=np.bincount(sources, minlength=count),
            best_next_prob=_best_outgoing(count, sources, probability),
            predicted_um=_predicted_positions(positions_um, sources, targets),
            link_sources=sources,
            link_targets=targets,
            link_prob=probability,
            link_scored=scored,
            max_timepoint=max(1, int(timepoints.max())) if count else 1,
        )

    def rows_at(self, timepoint: int) -> Int[np.ndarray, "k"]:
        """The detection rows of one frame, in the ascending order every gap matrix is aligned to."""
        return np.flatnonzero(self.timepoints == timepoint)


@dataclass(frozen=True)
class AssociationFeatures:
    """One frame gap's 22 features per gated candidate pair, as `s x t` planes keyed by their manifest names.

    A pair outside the gate is still given a value (the planes are dense) but is never scored: `matrix`
    emits only the gated rows, exactly as the public pipeline builds one record per in-gate candidate.
    """

    context: AssociationContext
    sources: Int[np.ndarray, "s"]
    targets: Int[np.ndarray, "t"]
    timepoint: int
    gate_um: float
    delta_um: Float[np.ndarray, "s t 3"]

    @classmethod
    def of(
        cls,
        context: AssociationContext,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        timepoint: int,
        gate_um: float,
    ) -> "AssociationFeatures":
        """One gap's features, with the source→target displacement every geometric column is derived from."""
        delta = context.positions_um[targets][None, :, :] - context.positions_um[sources][:, None, :]
        return cls(
            context=context, sources=sources, targets=targets, timepoint=timepoint, gate_um=gate_um, delta_um=delta
        )

    def distance_um(self) -> Float[np.ndarray, "s t"]:
        """The physical source→target distance of every pair — the gate's quantity and the `edge_dist_um` column."""
        return np.linalg.norm(self.delta_um, axis=2)

    def planes(self) -> dict[str, Float[np.ndarray, "s t"]]:
        """Every feature as an `s x t` plane, named exactly as the manifest names it."""
        delta = self.delta_um
        distance = self.distance_um()
        motion = cdist(self.context.predicted_um[self.sources], self.context.positions_um[self.targets])
        edge_prob, has_edge = self._learned()
        rank, count = self._candidate_list(distance)
        return {
            "edge_prob": edge_prob,
            "has_learned_edge": has_edge,
            "source_out_degree": self._of_source(self.context.out_degree),
            "target_in_degree": self._of_target(self.context.in_degree),
            "source_frame_count": self._of_source(self.context.frame_count),
            "target_frame_count": self._of_target(self.context.frame_count),
            "source_density_7um": self._of_source(self.context.density),
            "target_density_7um": self._of_target(self.context.density),
            "candidate_rank_dist": rank,
            "candidate_count": count,
            "edge_dz_um": delta[:, :, 0],
            "edge_dy_um": delta[:, :, 1],
            "edge_dx_um": delta[:, :, 2],
            "edge_dist_um": distance,
            "edge_xy_um": np.linalg.norm(delta[:, :, 1:], axis=2),
            "edge_abs_z_um": np.abs(delta[:, :, 0]),
            "motion_dist_um": motion,
            "motion_gain_um": distance - motion,
            "source_has_prev": self._of_source((self.context.in_degree > 0).astype(np.float64)),
            "target_has_next": self._of_target((self.context.out_degree > 0).astype(np.float64)),
            "target_best_next_prob": self._of_target(self.context.best_next_prob),
            "t_norm": np.full(
                (len(self.sources), len(self.targets)), self.timepoint / self.context.max_timepoint, dtype=np.float64
            ),
        }

    def gated(self) -> Bool[np.ndarray, "s t"]:
        """Which pairs the distance gate admits — the candidate set the ranker is asked about."""
        return self.distance_um() <= self.gate_um

    def matrix(self, names: tuple[str, ...]) -> Float[np.ndarray, "k f"]:
        """The gated pairs' feature rows in the manifest's column order — row-major over the `s x t` grid.

        An unknown column name is an error rather than a zero-filled column: a silently wrong feature vector
        produces a plausible-looking null result, which is the one failure mode a port like this must not have.
        """
        planes = self.planes()
        missing = [name for name in names if name not in planes]
        if missing:
            raise ValueError(f"the artifact asks for features this port does not compute: {sorted(missing)}")
        gated = self.gated()
        return np.stack([planes[name][gated] for name in names], axis=1).astype(np.float64)

    def _learned(self) -> tuple[Float[np.ndarray, "s t"], Float[np.ndarray, "s t"]]:
        """`edge_prob` and `has_learned_edge`: the primary link's own probability, zero for a non-link.

        Both are facts about the CONTEXT graph, not about the affinity: the public pipeline keys them off the
        emitted edge list, so a candidate the primary linking did not select scores 0 / 0 here however
        confidently the transformer rated it.
        """
        shape = (len(self.sources), len(self.targets))
        probability = np.zeros(shape, dtype=np.float64)
        present = np.zeros(shape, dtype=np.float64)
        context = self.context
        crossing = (context.timepoints[context.link_sources] == self.timepoint) & (
            context.timepoints[context.link_targets] == self.timepoint + 1
        )
        rows = context.row_in_frame[context.link_sources[crossing]]
        columns = context.row_in_frame[context.link_targets[crossing]]
        probability[rows, columns] = context.link_prob[crossing]
        present[rows, columns] = context.link_scored[crossing]
        return probability, present

    def _candidate_list(
        self, distance: Float[np.ndarray, "s t"]
    ) -> tuple[Float[np.ndarray, "s t"], Float[np.ndarray, "s t"]]:
        """`candidate_rank_dist` and `candidate_count` over each source's gated candidate list.

        This is the one piece a global min-cost flow does not build for itself: the flow poses every in-gate
        transition as an arc and never materialises "the list this source is choosing between". The list is
        exactly the gated row, ranked by physical distance with the target's node order breaking ties — the
        public definition — so the rank is 1-based and the count is the row's number of gated candidates.
        A pair outside the gate is not in any list; it is given rank 0 and never reaches `matrix`.
        """
        gated = distance <= self.gate_um
        ordered = np.argsort(np.where(gated, distance, np.inf), axis=1, kind="stable")
        rank = np.zeros_like(distance)
        placement = np.argsort(ordered, axis=1, kind="stable") + 1.0
        np.copyto(rank, placement, where=gated)
        return rank, gated.sum(axis=1, keepdims=True) * np.ones_like(distance)

    def _of_source(self, per_node: Float[np.ndarray, "n"]) -> Float[np.ndarray, "s t"]:
        """A per-node quantity broadcast down the source axis of the gap."""
        return np.broadcast_to(
            per_node[self.sources][:, None].astype(np.float64), (len(self.sources), len(self.targets))
        )

    def _of_target(self, per_node: Float[np.ndarray, "n"]) -> Float[np.ndarray, "s t"]:
        """A per-node quantity broadcast across the target axis of the gap."""
        return np.broadcast_to(
            per_node[self.targets][None, :].astype(np.float64), (len(self.sources), len(self.targets))
        )


class _RankerMLP(nn.Module):
    """The artifact's own architecture — `Linear/LayerNorm/ReLU/Dropout/Linear/ReLU/Linear`, one logit out.

    The module names (`net.0`, `net.1`, `net.4`, `net.6`) are the checkpoint's, so a `strict=True` load is a
    check on this reconstruction: a wrong depth, width or a missing LayerNorm fails to load rather than
    quietly producing scores from an architecture the weights were never trained in.
    """

    def __init__(self, features: int, hidden: int, inner: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, inner),
            nn.ReLU(),
            nn.Linear(inner, 1),
        )

    @override
    def forward(self, features: Float[Tensor, "k f"]) -> Float[Tensor, "k"]:
        """One logit per candidate row."""
        return self.net(features)[:, 0]


@dataclass(frozen=True)
class AssociationRanker:
    """The mounted CC0 re-ranker: manifest-ordered features in, `P(true successor)` per candidate out.

    `affinities` turns it into an `EdgeAffinity` — per-gap `s x t` probability matrices in the linker's own
    node order — because that is how a second learned quantity already reaches the linkers (the fused mutual
    affinity does exactly this), and because `celltrack.linkers` may not import `celltrack.edges` at all.
    """

    module: _RankerMLP
    feature_names: tuple[str, ...]
    mean: Float[np.ndarray, "f"]
    std: Float[np.ndarray, "f"]

    @classmethod
    def from_artifact(cls, root: Path) -> "AssociationRanker":
        """Mount the downloaded artifact directory: manifest for the feature order, checkpoint for the rest."""
        manifest = json.loads((root / _MANIFEST_NAME).read_text(encoding="utf-8"))["model"]
        checkpoint = torch.load(root / manifest["path"], map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        names = tuple(manifest["feature_columns"])
        features, hidden = state["net.0.weight"].shape[1], state["net.0.weight"].shape[0]
        if features != len(names):
            raise ValueError(f"artifact takes {features} inputs but its manifest names {len(names)} features")
        module = _RankerMLP(features, hidden, state["net.4.weight"].shape[0], float(checkpoint["config"]["dropout"]))
        module.load_state_dict(state, strict=True)
        module.eval()
        mean = np.asarray(checkpoint["mean"], dtype=np.float64)
        std = np.maximum(np.asarray(checkpoint["std"], dtype=np.float64), _STD_FLOOR)
        return cls(module=module, feature_names=names, mean=mean, std=std)

    def probabilities(self, features: Float[np.ndarray, "k f"]) -> Float[np.ndarray, "k"]:
        """`P(true successor)` per candidate row — standardised by the artifact's stats, then sigmoid."""
        if features.shape[1] != len(self.feature_names):
            raise ValueError(f"got {features.shape[1]} features, the artifact takes {len(self.feature_names)}")
        standardised = (features - self.mean[None, :]) / self.std[None, :]
        with torch.no_grad():
            logits = self.module(torch.from_numpy(standardised.astype(np.float32)))
        return torch.sigmoid(logits).numpy().astype(np.float64)

    def affinities(
        self, spacing: Spacing, detections: TrackGraph, links: TrackGraph, affinity: EdgeAffinity | None, gate_um: float
    ) -> PrecomputedEdgeAffinity:
        """Re-rank one video: a per-gap `s x t` matrix of ranker probabilities, zero outside the gate.

        `links` is the primary linking this re-ranks — the graph the degree, `edge_prob` and motion features
        read — so the caller links once with the shipped cost, hands the result here, and links again with the
        ranker term priced in. `gate_um` must be the linker's own gate: it defines the candidate list the
        `candidate_rank_dist` / `candidate_count` features are computed over.
        """
        context = AssociationContext.of(spacing, detections, links, affinity)
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for timepoint in np.unique(context.timepoints)[:-1]:
            sources, targets = context.rows_at(int(timepoint)), context.rows_at(int(timepoint) + 1)
            if len(sources) == 0 or len(targets) == 0:
                continue
            by_timepoint[int(timepoint)] = self._gap(context, sources, targets, int(timepoint), gate_um)
        return PrecomputedEdgeAffinity(by_timepoint)

    def _gap(
        self,
        context: AssociationContext,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        timepoint: int,
        gate_um: float,
    ) -> Float[np.ndarray, "s t"]:
        """One gap's ranker probabilities, scattered back onto the `s x t` grid the linker cost is aligned to."""
        features = AssociationFeatures.of(context, sources, targets, timepoint, gate_um)
        gated = features.gated()
        scored = np.zeros(gated.shape, dtype=np.float64)
        if gated.any():
            scored[gated] = self.probabilities(features.matrix(self.feature_names))
        return scored


def _frame_positions(timepoints: Int[np.ndarray, "n"]) -> tuple[Int[np.ndarray, "n"], Int[np.ndarray, "n"]]:
    """Each node's index within its own frame, and how many detections that frame holds.

    The index is what maps a node row onto the `s x t` grid a gap's matrices are built on; the count is the
    `*_frame_count` feature (the public pipeline's `len(ids_by_t[t])`).
    """
    row_in_frame = np.zeros(len(timepoints), dtype=np.int64)
    frame_count = np.zeros(len(timepoints), dtype=np.int64)
    for timepoint in np.unique(timepoints):
        rows = np.flatnonzero(timepoints == timepoint)
        row_in_frame[rows] = np.arange(len(rows))
        frame_count[rows] = len(rows)
    return row_in_frame, frame_count


def _frame_density(positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
    """How many same-frame detections lie within 7 um of each node, excluding itself — the crowding feature."""
    density = np.zeros(len(timepoints), dtype=np.float64)
    for timepoint in np.unique(timepoints):
        rows = np.flatnonzero(timepoints == timepoint)
        tree = KDTree(positions_um[rows])
        counts = tree.query_ball_point(positions_um[rows], r=_DENSITY_RADIUS_UM, return_length=True)
        density[rows] = np.maximum(counts - 1, 0)
    return density


def _link_rows(detections: TrackGraph, links: TrackGraph) -> tuple[Int[np.ndarray, "e"], Int[np.ndarray, "e"]]:
    """The primary links as row indices into the DETECTION node arrays, whichever node order `links` carries."""
    if len(links.edges) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return detections.rows_of(links.edges[:, 0]), detections.rows_of(links.edges[:, 1])


def _link_probabilities(
    timepoints: Int[np.ndarray, "n"],
    row_in_frame: Int[np.ndarray, "n"],
    sources: Int[np.ndarray, "e"],
    targets: Int[np.ndarray, "e"],
    affinity: EdgeAffinity | None,
) -> tuple[Float[np.ndarray, "e"], Bool[np.ndarray, "e"]]:
    """The learned probability of each SELECTED link, and whether the affinity scored it at all.

    This is the public pipeline's `learned_edge_probs`, which it fills from the emitted edge list: a link whose
    gap the affinity never scored carries no probability, which is what makes `has_learned_edge` informative
    rather than constant.
    """
    probability = np.zeros(len(sources), dtype=np.float64)
    scored = np.zeros(len(sources), dtype=bool)
    if affinity is None:
        return probability, scored
    for index, (source, target) in enumerate(zip(sources.tolist(), targets.tolist(), strict=True)):
        matrix = affinity.probabilities(int(timepoints[source]))
        if matrix is not None and int(timepoints[target]) == int(timepoints[source]) + 1:
            probability[index] = float(matrix[row_in_frame[source], row_in_frame[target]])
            scored[index] = True
    return probability, scored


def _best_outgoing(
    count: int, sources: Int[np.ndarray, "e"], probability: Float[np.ndarray, "e"]
) -> Float[np.ndarray, "n"]:
    """The strongest outgoing link probability of each node — the `target_best_next_prob` feature.

    A target that already has a confident successor of its own is a different proposition than one with none,
    which is the crowding fact this feature carries into the score.
    """
    best = np.zeros(count, dtype=np.float64)
    np.maximum.at(best, sources, probability)
    return best


def _predicted_positions(
    positions_um: Float[np.ndarray, "n 3"], sources: Int[np.ndarray, "e"], targets: Int[np.ndarray, "e"]
) -> Float[np.ndarray, "n 3"]:
    """Where constant velocity says each node goes next: `position + 0.5 * (position - parent position)`.

    A node with no parent in the context graph predicts its own position — no velocity is known, so the motion
    distance degenerates to the raw distance and `motion_gain_um` to zero, exactly as the public pipeline's
    `predicted = source_pos` branch does.
    """
    previous = positions_um.copy()
    previous[targets] = positions_um[sources]
    return positions_um + _VELOCITY_WEIGHT * (positions_um - previous)

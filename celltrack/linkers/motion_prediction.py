"""The geometry a pair is PRICED in: distance to where the source was heading, not to where it sits.

Every linker in the family prices a candidate by `‖target − source‖`. That is the quantity a fast mover
breaks: its true successor lands FAR while a stationary neighbour sits NEAR, so raw distance ranks the wrong
target first and the learned terms have to overcome the geometry rather than refine it. `MotionHungarianLinker`
already answers this per frame — it competes pairs on `‖target − predicted(source)‖` — but it answers it
inside a per-frame matcher, whose ceiling on our movies is ~0.045 below the global flow solver's. This term
carries the same geometry into a cost the global solver can read, so the two choices (which distance, which
optimiser) stop being one bundled decision.

The velocity is NOT a hand-rolled one. `PriorVelocity.expected_incoming` already defines the displacement
that carried a cell into its current position as an affinity-weighted EXPECTED incoming step, deliberately
unnormalised: a target whose incoming probability mass falls short of one — an entering cell, or a previous
gap the head found ambiguous — gets a proportionally SHRUNKEN velocity rather than a confident guess at a
direction. Degrading toward zero is exactly the property a cost term wants, because a zero velocity is the
raw-distance cost this replaces, so an unreadable history costs nothing rather than mispricing the pair.

That definition needs only the PREVIOUS gap's probabilities, which the linker is already holding — so no
prior linking pass, no second affinity forward, and no place for a train/inference asymmetry to enter. The
prediction is damped by `VELOCITY_DAMPING`, the motion linker's argued half step, imported rather than
restated so the family has one damping constant and one argument for it.

GATE: this term prices, it does not admit. The caller gates on RAW distance, because the gate is a physical
admissibility statement — how far a cell can travel in one frame gap — while a prediction is an estimate that
can be wrong; gating on it would let a bad velocity make a physically reachable successor unlinkable, and
would silently widen the reachable set in the direction the cell was already moving. Prediction chooses among
admissible pairs; physics decides which pairs are admissible. This is the same split `MotionHungarianLinker`
makes (its tight/loose gates read `displacement`, its cost reads the predicted distance).
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.motion_linking import VELOCITY_DAMPING
from celltrack.models.prior_velocity import PriorVelocity


@dataclass(frozen=True)
class MotionPrediction:
    """The motion-predicted distance for one frame gap: `‖target − (source + damping · expected incoming step)‖`.

    `affinity` is the same directional source→target probability the cost's forward term prices — a softmax
    over the SOURCES of each target, which is the normalisation `PriorVelocity.expected_incoming` reads to
    make an incoming column's shortfall shrink the velocity. The PREVIOUS gap's matrix supplies the step that
    carried each source into place; a gap the affinity does not score (the video's first frame, or an unscored
    gap) yields exact zeros, so the pair is priced by raw distance there.
    """

    affinity: EdgeAffinity

    def distances(
        self,
        timepoint: int,
        sources: Int[np.ndarray, " s"],
        targets: Int[np.ndarray, " t"],
        timepoints: Int[np.ndarray, " n"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> Float[np.ndarray, "s t"]:
        """Every `source → target` distance measured from the source's PREDICTED position, in micrometres."""
        step = self._velocity(timepoint, sources, timepoints, positions_um)
        return cdist(positions_um[sources] + VELOCITY_DAMPING * step, positions_um[targets])

    def _velocity(
        self,
        timepoint: int,
        sources: Int[np.ndarray, " s"],
        timepoints: Int[np.ndarray, " n"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> Float[np.ndarray, "s 3"]:
        """The expected step that carried each source into place — exact zeros where the previous gap is unusable."""
        previous = np.flatnonzero(timepoints == timepoint - 1)
        probabilities = self.affinity.probabilities(timepoint - 1) if previous.size else None
        if probabilities is None:
            return np.zeros_like(positions_um[sources])
        step = PriorVelocity.expected_incoming(
            torch.from_numpy(positions_um[sources]),
            torch.from_numpy(positions_um[previous]),
            torch.from_numpy(np.asarray(probabilities)),
        )
        return step.numpy()

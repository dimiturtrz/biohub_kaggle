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
from jaxtyping import Float
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.frame_gap import FrameGap
from celltrack.linkers.motion_linking import VELOCITY_DAMPING
from celltrack.models.prior_velocity import PriorVelocity

# A target with no incoming mass has no parent to attribute a step to; the floor keeps the division finite.
_MASS_FLOOR = 1e-9


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
    # How many past gaps the velocity averages over. 1 = the previous gap alone (single step), byte-identical to
    # the original. A single step is DOMINATED by localisation noise here: the individual cell motion is ~1 um,
    # below the ~1.7 um integer-voxel annotation noise, so one displacement is mostly noise (measured on the
    # dense mislinks: velocity-to-true-step alignment cos +0.00 at one step, +0.30 at three). Averaging over more
    # gaps beats the noise down as ~1/sqrt(history) while a persistent velocity survives — the difference between
    # a single-step Kalman and a trajectory fit, chained probabilistically through the affinity so it needs no
    # prior linking pass. Each earlier gap's step is carried forward by the SAME affinity that prices the cost,
    # so an ambiguous history shrinks toward zero (the raw-distance cost) rather than guessing a direction.
    history: int = 1

    def distances(self, gap: FrameGap) -> Float[np.ndarray, "s t"]:
        """Every `source → target` distance measured from the source's PREDICTED position, in micrometres."""
        step = self._velocity(gap)
        return cdist(gap.positions_um[gap.sources] + VELOCITY_DAMPING * step, gap.positions_um[gap.targets])

    def _velocity(self, gap: FrameGap) -> Float[np.ndarray, "s 3"]:
        """The sources' velocity, averaged over up to `history` past gaps — exact zeros where none is usable.

        A `responsibility` matrix carries each source's soft ancestry back one frame per gap: initially each
        source answers only for itself, and after each gap it is spread over its affinity-weighted parents, so
        the step read at an earlier frame is attributed to the sources that most likely descend from it. The
        chain is entirely soft, so it costs one matrix product per gap and never commits a link.
        """
        velocity = np.zeros((len(gap.sources), 3))
        responsibility = np.eye(len(gap.sources))
        current = gap.sources
        levels = 0
        for step_back in range(self.history):
            prev_frame = gap.timepoint - 1 - step_back
            previous = np.flatnonzero(gap.timepoints == prev_frame)
            probabilities = self.affinity.probabilities(prev_frame) if previous.size else None
            if probabilities is None:
                break
            matrix = np.asarray(probabilities, dtype=np.float64)
            velocity = velocity + responsibility @ self._incoming(
                gap.positions_um[current], gap.positions_um[previous], matrix
            )
            responsibility = responsibility @ self._parent_distribution(matrix).T
            current = previous
            levels += 1
        return velocity / max(levels, 1)

    @staticmethod
    def _incoming(
        current_um: Float[np.ndarray, "c 3"],
        previous_um: Float[np.ndarray, "p 3"],
        matrix: Float[np.ndarray, "p c"],
    ) -> Float[np.ndarray, "c 3"]:
        """Each current-frame node's affinity-weighted incoming step from the previous frame."""
        return (
            PriorVelocity.expected_incoming(
                torch.from_numpy(current_um), torch.from_numpy(previous_um), torch.from_numpy(matrix)
            )
            .numpy()
            .astype(np.float64)
        )

    @staticmethod
    def _parent_distribution(matrix: Float[np.ndarray, "p c"]) -> Float[np.ndarray, "p c"]:
        """The affinity column-normalised over the previous frame — each current node's soft parent weights."""
        return matrix / np.maximum(matrix.sum(axis=0, keepdims=True), _MASS_FLOOR)

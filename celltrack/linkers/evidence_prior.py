"""The head's own verdict on whether a detection has a parent at all, priced as its appearance.

`FlowLinker` charges one flat price for a track to begin, everywhere, for every detection. The edge head
already estimates something much better per target, and we currently throw it away: its association softmax is
normalised over EVERY source in the frame, while the gate admits a handful, so the share of a target's
probability mass that lands on ADMISSIBLE parents is exactly the head saying "one of these is the parent" —
and a column holding almost none of it is the head saying "none of these is", which is what an appearance is.

MEASURED on the dense movie (`celltrack.eval.softmax_population`, 69164 target columns): the in-gate share is
mean 0.7796, median 0.9174, p25 0.7289, p5 0.0861. So the signal is real and strongly skewed — most targets
have a confident admissible parent and a twentieth have essentially none.

THIS IS THE CONSTRUCTIVE INVERSE OF A REFUTED ARM. `AdmissibleAffinity` renormalised the out-of-gate mass
AWAY, conditioning each column on the admissible parents, and cost 0.0083 — because conditioning asserts that
some admissible parent IS the parent, which is false for exactly the track starts this reads. Same quantity,
opposite operation: that one deleted the evidence, this one prices it.

The factor is BUDGET-PRESERVING and therefore has no strength knob: factors are normalised so their mean over
a frame's detections is 1, so an arm moves WHERE the appearance charge falls without changing how much of it
there is, and the flat case is its own control. That is the device `EvidenceRamp` uses for the affinity bonus,
for the same reason — an arm that also raised the total would be confounded with "cheaper births".

Speaks only about APPEARANCE. A source-normalised softmax says who a target's parent is; it says nothing about
whether a source has a child, so the disappearance factor is left at 1 rather than invented.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.boundary_prior import BoundaryFactors
from core.geometry import Spacing

_FLAT = 1.0
_NO_MASS = 0.0


@dataclass(frozen=True)
class EvidencePrior:
    """Per-detection appearance factors read off the edge head's in-gate probability mass."""

    affinity: EdgeAffinity

    def factors(
        self,
        spacing: Spacing,
        margin_um: float,
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
    ) -> BoundaryFactors:
        """Appearance scaled by how well the head believes each detection already has an admissible parent.

        `margin_um` is the linker's own gate, handed in rather than restated, so the admissible set this reads
        is exactly the one the linker will price. `spacing` is unused — the positions arrive in micrometres —
        and is accepted so this speaks the same vocabulary as the geometric prior it sits beside.
        """
        mass = self._in_gate_mass(margin_um, positions_um, timepoints)
        return BoundaryFactors(
            appearance=self._normalised(mass, timepoints),
            disappearance=np.full(len(timepoints), _FLAT),
        )

    def _in_gate_mass(
        self, margin_um: float, positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> Float[np.ndarray, "n"]:
        """Each detection's share of parent probability held by sources it could admissibly have come from.

        A detection in the first observed frame, or in a gap the scorer did not score, has no incoming gap to
        read and keeps the flat charge: there is no evidence about its parent either way, and inventing some
        would price the whole first frame on a number that does not exist.
        """
        mass = np.full(len(timepoints), _FLAT)
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probability = self.affinity.probabilities(timepoint)
            if probability is None:
                continue
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            admissible = self._within_gate(margin_um, positions_um[sources], positions_um[targets])
            mass[targets] = (probability * admissible).sum(axis=0)
        return mass

    @staticmethod
    def _within_gate(
        margin_um: float, sources_um: Float[np.ndarray, "s 3"], targets_um: Float[np.ndarray, "t 3"]
    ) -> Bool[np.ndarray, "s t"]:
        """Which candidate parents the linker would admit — its gate, applied to the same separations."""
        separation = np.linalg.norm(sources_um[:, None, :] - targets_um[None, :, :], axis=-1)
        return separation <= margin_um

    @staticmethod
    def _normalised(mass: Float[np.ndarray, "n"], timepoints: Int[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
        """Scale the mass to mean 1 WITHIN each frame, so the appearance budget only moves, never grows.

        Per frame rather than per movie because frames differ in crowding, and a movie-wide mean would quietly
        make births cheaper in sparse frames and dearer in dense ones — a density effect wearing an evidence
        argument. A frame whose mass is entirely zero has nothing to redistribute and stays flat.
        """
        factors = np.full(len(timepoints), _FLAT)
        for timepoint in np.unique(timepoints).tolist():
            rows = np.flatnonzero(timepoints == timepoint)
            average = float(mass[rows].mean())
            if average > _NO_MASS:
                factors[rows] = mass[rows] / average
        return factors

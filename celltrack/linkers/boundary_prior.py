"""Which track boundaries the observation window already explains — a per-detection discount on a flat boundary cost.

A flow linker charges one flat price for a track appearing and one for it disappearing, and that flat price is
the whole reason a global solver beats a per-frame one: a break costs an appearance plus a disappearance, so the
optimiser keeps a true track intact. But the price is a PRIOR — "a track should not start here" — and that prior
is not uniform over the video. A track that starts in the first observed frame was not born there; the movie
started mid-life. A track that starts one admissible step from a face of the imaged volume was not born there
either; the cell walked in from outside the field of view. Both are explained by WHAT WAS FILMED, not by a
linking error, and charging them the interior price is charging for the camera's limits.

A track that appears deep inside the volume, mid-movie, has no such excuse: for it to be real a cell must have
materialised in open space. That is the one the flat cost is meant to discourage, and pricing the two
identically is what denies the solver the single geometric fact separating them.

This prior is the same claim `ShortTrackFilter.keep_boundary` makes, moved from AFTER the linker to INSIDE its
cost: rather than exempting clip-truncated tracks from a length rule once the links are fixed, it tells the
solver during optimisation that such a track is cheap to open, so the boundary can be traded against a
transition instead of patched afterwards.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from core.geometry import Spacing

# What an explained track boundary costs, as a fraction of the flat boundary cost. Zero, and derived rather than
# tuned: the boundary cost prices the IMPLAUSIBILITY of a track terminating where it does, and at a clip boundary
# the observation window already accounts for the terminus — there is no implausibility left to charge. Any
# positive value would assert a prior against an entry or exit the geometry makes certain, and would push the
# solver to splice such a track onto whatever neighbour is cheapest purely to dodge the fee. Zero is also the
# value at which `FlowLinker` is documented to reduce to the per-frame assignment, so an explained boundary
# behaves exactly like that free-boundary regime — locally, where the window says boundaries really are free.
_EXPLAINED_FACTOR = 0.0
_INTERIOR_FACTOR = 1.0


@dataclass(frozen=True)
class BoundaryFactors:
    """A `BoundaryPrior`'s verdict per detection: what fraction of the flat cost its two boundary arcs each pay.

    Appearance and disappearance are separate because the temporal half of the prior is one-sided. A detection in
    the FIRST observed frame is free to appear (its cell predates the movie) but not free to vanish there; the
    last frame is the mirror. Only the spatial half is symmetric — a cell near a face can as easily enter through
    it as leave through it.
    """

    appearance: Float[np.ndarray, "n"]
    disappearance: Float[np.ndarray, "n"]


@dataclass(frozen=True)
class BoundaryPrior:
    """The imaged volume, as the fact that tells a linker which track boundaries its own observation window caused.

    Holds only what the linker cannot already see — the `(z, y, x)` voxel shape of a timepoint. The margin and the
    time range are derived from what it is handed (`factors`), so the band can never disagree with the gate that
    defines it.
    """

    volume_shape: tuple[int, int, int]

    def factors(
        self,
        spacing: Spacing,
        margin_um: float,
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
    ) -> BoundaryFactors:
        """The per-detection boundary discount: explained at a face or at the clip's first/last frame, else full.

        `margin_um` is the linker's own gate, not a new number. The gate is the linker's statement of the furthest
        a cell travels between consecutive frames — every longer transition is refused — so a detection further
        than the gate from every face CANNOT have been outside the volume on the previous frame: no admissible
        step reaches it from outside, and its track therefore did not enter the field of view. Inside the band the
        converse holds: an off-field cell is one admissible step away, so the appearance is fully explained by the
        field of view. The band is exactly the set the gate already defines; any independently chosen margin would
        be a second, unargued radius.
        """
        explained = self._within_a_face(spacing, margin_um, positions_um)
        return BoundaryFactors(
            appearance=self._factor(explained | (timepoints == timepoints.min())),
            disappearance=self._factor(explained | (timepoints == timepoints.max())),
        )

    def _within_a_face(
        self, spacing: Spacing, margin_um: float, positions_um: Float[np.ndarray, "n 3"]
    ) -> Bool[np.ndarray, "n"]:
        """Whether each detection sits within `margin_um` of any face of the imaged volume, in micrometres.

        The far face is the LAST ADDRESSABLE voxel centre, `(shape - 1) * spacing`, because a detection is a cell
        centre and no cell centre can be reported beyond it — that plane, not the array's outer edge, is the
        furthest a track can be observed before leaving. A degenerate axis (unit extent, or a zero spacing)
        collapses to a zero-width interior, so every detection is at that face: with no depth to cross, a cell
        genuinely is always one step from leaving through it.
        """
        extent_um = (np.asarray(self.volume_shape, dtype=np.float64) - 1.0) * spacing.as_array()
        to_nearest_face = np.minimum(positions_um, extent_um - positions_um)
        return np.any(to_nearest_face <= margin_um, axis=1)

    @staticmethod
    def _factor(explained: Bool[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
        """The discount as a multiplier — free where the window explains the boundary, full price in the interior."""
        return np.where(explained, _EXPLAINED_FACTOR, _INTERIOR_FACTOR)

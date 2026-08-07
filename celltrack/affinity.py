"""Shared vocabulary for learned edge association, spoken across edges, linkers, postproc, and eval.

An `EdgeAffinity` is a learned source->target association probability for one frame gap, aligned to the
linker's node order. It lives at the package root — not inside a linker — because it is a contract several
layers depend on: an edge scorer produces one, a linker consumes it, and postproc/eval read it back. Keeping
it here lets each of those layers import the vocabulary without importing a linker's implementation.
"""

from typing import Protocol

import numpy as np
from jaxtyping import Float


class EdgeAffinity(Protocol):
    """A learned source->target association probability for one frame gap, aligned to the linker's node order.

    Rows and columns follow the same ascending-index order the linker builds its cost from
    (`np.flatnonzero(timepoints == t)` for the sources, `== t + 1` for the targets), so the matrix drops
    straight onto the cost. `None` for a gap with no learned scores leaves that gap pure geometry.
    """

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        """P(source is the parent of target) for every candidate pair across the `t -> t+1` gap."""
        ...

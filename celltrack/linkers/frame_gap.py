"""One consecutive-frame gap: the sources, their next-frame candidates, and the geometry to price them.

Every per-frame linker walks the same sweep — for each timepoint but the last, the cells at `t` are the
sources and the cells at `t+1` the targets, matched under the shared `positions_um`. This record carries that
one gap so a linker iterates `FrameGap.sweep(...)` instead of re-deriving `sources`/`targets` and threading
`(timepoint, sources, targets, positions_um, timepoints)` through every cost method as five loose arguments.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True)
class FrameGap:
    """The sources at `timepoint`, their targets at `timepoint + 1`, and the frame geometry that prices them.

    `positions_um` and `timepoints` are the whole video's arrays, not the gap's slice: a motion model reads
    frames before this one, and `sources`/`targets` index into them, so both stay full-length and the gap
    holds only the two row sets a single matching couples.
    """

    timepoint: int
    sources: Int[np.ndarray, " s"]
    targets: Int[np.ndarray, " t"]
    positions_um: Float[np.ndarray, "n 3"]
    timepoints: Int[np.ndarray, " n"]

    @classmethod
    def sweep(cls, positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, " n"]) -> "Iterator[FrameGap]":
        """Every consecutive-frame gap in order — the loop body each per-frame linker shares."""
        for timepoint in np.unique(timepoints)[:-1]:
            yield cls(
                timepoint=int(timepoint),
                sources=np.flatnonzero(timepoints == timepoint),
                targets=np.flatnonzero(timepoints == timepoint + 1),
                positions_um=positions_um,
                timepoints=timepoints,
            )

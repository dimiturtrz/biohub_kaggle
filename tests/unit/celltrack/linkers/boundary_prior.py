import numpy as np

from celltrack.linkers.boundary_prior import BoundaryPrior
from core.geometry import Spacing

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)
# A 21-voxel cube: at an isotropic 1 um and a 5 um margin, the interior is the 11 um core (voxels 6..14).
CUBE = (21, 21, 21)
_CENTRE = [10, 10, 10]


def _factors(
    prior: BoundaryPrior, coordinates: list[list[int]], margin_um: float = 5.0, spacing: Spacing = ISOTROPIC
) -> tuple[list[float], list[float]]:
    """The prior's verdict for `t, z, y, x` rows, as the two lists of per-node multipliers."""
    array = np.array(coordinates, dtype=np.int64)
    factors = prior.factors(spacing, margin_um, spacing.to_micrometres(array[:, 1:]), array[:, 0])
    return factors.appearance.tolist(), factors.disappearance.tolist()


def _spatial(
    prior: BoundaryPrior,
    positions: list[list[int]],
    margin_um: float = 5.0,
    spacing: Spacing = ISOTROPIC,
    centre: list[int] | None = None,
) -> tuple[list[float], list[float]]:
    """The spatial half alone: probes sit mid-movie, between two anchors that own the clip's first and last frame."""
    anchor = centre if centre is not None else _CENTRE
    rows = [[0, *anchor], *[[5, *position] for position in positions], [9, *anchor]]
    appearance, disappearance = _factors(prior, rows, margin_um, spacing)
    return appearance[1:-1], disappearance[1:-1]


def test_factors():
    """A detection within one gate of a face is free to appear AND to vanish; the mid-volume one pays full price.

    Both directions are discounted at a face because the field of view is crossable both ways — a cell can walk
    in through it as easily as out.
    """
    appearance, disappearance = _spatial(BoundaryPrior(CUBE), [[10, 10, 2], [10, 10, 10]])
    assert appearance == [0.0, 1.0]
    assert disappearance == [0.0, 1.0]


def test_factors_margin_is_the_gate():
    """The band is exactly the gate wide: a cell one gate from the face could have been outside a frame ago.

    At margin 5 the voxel at x=5 is reachable from off-field in one admissible step and is discounted; x=6 is not
    reachable by any transition the linker would admit, so its appearance is a break, not an entry.
    """
    appearance, _ = _spatial(BoundaryPrior(CUBE), [[10, 10, 5], [10, 10, 6]])
    assert appearance == [0.0, 1.0]


def test_factors_measures_the_margin_in_micrometres():
    """The band is physical, not voxel-counted: under anisotropy the same voxel offset is a different distance.

    At a 3 um gate and 4 um z voxels, the slice two voxels in is 8 um deep — beyond the gate — while the identical
    offset along x (0.5 um voxels) is 1 um and still inside the band. Counting voxels would call the two the same.
    """
    anisotropic = Spacing(z=4.0, y=0.5, x=0.5)
    appearance, _ = _spatial(BoundaryPrior(CUBE), [[2, 10, 10], [10, 10, 2]], margin_um=3.0, spacing=anisotropic)
    assert appearance == [1.0, 0.0]


def test_factors_far_face_is_the_last_addressable_voxel():
    """The far face is `(shape - 1) * spacing` — the last voxel centre a cell can be reported in, not `shape`."""
    appearance, _ = _spatial(BoundaryPrior(CUBE), [[10, 10, 15], [10, 10, 14]])
    assert appearance == [0.0, 1.0]  # x=15 is 5 um from the x=20 face; x=14 is 6 um, past the gate


def test_factors_first_and_last_frame_are_one_sided():
    """The clip's ends discount one arc each: the first frame is free to open a track, the last free to close one.

    A cell in frame 0 predates the movie, so admitting it costs nothing — but it has no excuse for vanishing
    there, and the mirror holds at the end. A mid-movie interior node pays both.
    """
    rows = [[0, *_CENTRE], [5, *_CENTRE], [9, *_CENTRE]]
    appearance, disappearance = _factors(BoundaryPrior(CUBE), rows)
    assert appearance == [0.0, 1.0, 1.0]
    assert disappearance == [1.0, 1.0, 0.0]


def test_factors_clip_ends_are_the_observed_range():
    """First and last mean the first and last frame that HOLDS a detection — the window actually observed."""
    appearance, disappearance = _factors(BoundaryPrior(CUBE), [[3, *_CENTRE], [7, *_CENTRE]])
    assert appearance == [0.0, 1.0]
    assert disappearance == [1.0, 0.0]


def test_factors_degenerate_axis_has_no_interior():
    """An axis with no depth puts every detection at its face — with nothing to cross, a cell is always leaving.

    Both degeneracies collapse the same way: a single-voxel axis, and a zero spacing that gives a real axis no
    physical extent. The margin derivation stays a comparison of micrometres and needs no special case.
    """
    flat_axis, _ = _spatial(BoundaryPrior((1, 21, 21)), [[0, 10, 10]], centre=[0, 10, 10])
    zero_spacing, _ = _spatial(BoundaryPrior(CUBE), [_CENTRE], spacing=Spacing(z=0.0, y=1.0, x=1.0))
    assert flat_axis == [0.0]
    assert zero_spacing == [0.0]

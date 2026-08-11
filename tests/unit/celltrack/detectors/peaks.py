import numpy as np

from celltrack.detectors.peaks import PeakExtractor
from core.geometry import Spacing

EXTRACTOR = PeakExtractor(spacing=Spacing(z=1.0, y=1.0, x=1.0), scale_um=2.0)


def test_centres():
    """The strongest suppressed maxima come back as (z, y, x) voxel indices."""
    response = np.zeros((16, 16, 16))
    response[4, 4, 4] = 1.0
    response[4, 12, 12] = 2.0
    assert sorted(EXTRACTOR.centres(response, keep=2).tolist()) == [[4, 4, 4], [4, 12, 12]]


def test_above_threshold():
    """Every suppressed maximum above the threshold comes back; a weaker maximum and the background do not."""
    response = np.zeros((16, 16, 16))
    response[4, 4, 4] = 0.99
    response[4, 12, 12] = 0.30
    assert EXTRACTOR.above_threshold(response, threshold=0.5).tolist() == [[4, 4, 4]]


def test_above_threshold_collapses_a_plateau_to_one_centre():
    """A flat plateau of equal maxima is one blob, not a field of peaks — it collapses to a single centre."""
    response = np.zeros((16, 16, 16))
    response[4:8, 4:8, 4:8] = 1.0
    assert len(EXTRACTOR.above_threshold(response, threshold=0.5)) == 1


def test_maxima():
    """`maxima` returns each suppressed peak with its value; a higher threshold is then a mask, not a re-pool."""
    response = np.zeros((16, 16, 16))
    response[4, 4, 4] = 0.99
    response[4, 12, 12] = 0.60
    coordinates, values = EXTRACTOR.maxima(response, floor=0.5)
    assert sorted(coordinates.tolist()) == [[4, 4, 4], [4, 12, 12]]
    kept = coordinates[values >= 0.9]
    assert kept.tolist() == [[4, 4, 4]]


def test_maxima_nest_in_the_floor():
    """Raising the floor drops whole peaks and never splits one, so a threshold sweep masks instead of re-pooling.

    Each connected component of `volume == max_pool(volume)` is CONSTANT-valued: two adjacent marked voxels lie
    within each other's pooling kernel, so each is at least the other. A floor therefore keeps or removes a
    component whole — the maxima above a high floor are exactly the low-floor maxima whose value clears it, which
    is what lets one extraction serve a whole sweep. Holds while the suppression radius is at least one voxel on
    every axis (the isotropic downsample the pipeline reads out on); a zero-radius axis would leave neighbours
    uncompared, and a component could then straddle the floor.
    """
    response = np.zeros((16, 16, 16))
    response[2:5, 2:5, 2:5] = 0.9  # a plateau that clears the high floor
    response[10:13, 10:13, 10:13] = 0.5  # and one that does not
    response[8, 2, 2] = 0.85

    coordinates, values = EXTRACTOR.maxima(response, floor=0.2)
    masked = coordinates[values > 0.8]
    above, _ = EXTRACTOR.maxima(response, floor=0.8)

    assert sorted(masked.tolist()) == sorted(above.tolist())


def test_centres_suppresses_a_neighbour_within_a_cell():
    """Two maxima closer than the cell scale cannot both survive — the weaker is suppressed."""
    response = np.zeros((16, 16, 16))
    response[8, 8, 8] = 2.0
    response[8, 8, 9] = 1.0
    assert EXTRACTOR.centres(response, keep=2).tolist() == [[8, 8, 8]]


def test_centres_suppresses_anisotropically():
    """The radius reaches further in y/x than z, so a z-neighbour survives where a y-neighbour would not.

    With 4 um voxels in z against 1 um in y/x, a 4 um scale gives a suppression radius of 1 voxel in z and
    4 in y/x — so two peaks two voxels apart both survive along z but collapse to one along y.
    """
    extractor = PeakExtractor(spacing=Spacing(z=4.0, y=1.0, x=1.0), scale_um=4.0)
    along_z = np.zeros((16, 16, 16))
    along_z[6, 8, 8], along_z[8, 8, 8] = 2.0, 1.0
    assert len(extractor.centres(along_z, keep=2)) == 2
    along_y = np.zeros((16, 16, 16))
    along_y[8, 6, 8], along_y[8, 8, 8] = 2.0, 1.0
    assert len(extractor.centres(along_y, keep=2)) == 1

import numpy as np

from celltrack.peaks import PeakExtractor
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

import numpy as np

from core.geometry import Spacing

ISOTROPIC = Spacing(z=2.0, y=0.5, x=0.5)


def test_as_array():
    assert ISOTROPIC.as_array().tolist() == [2.0, 0.5, 0.5]


def test_to_micrometres():
    voxels = np.array([[1, 2, 4], [0, 0, 0]])
    assert ISOTROPIC.to_micrometres(voxels).tolist() == [[2.0, 1.0, 2.0], [0.0, 0.0, 0.0]]


def test_to_voxels():
    micrometres = np.array([[2.0, 1.0, 2.0]])
    assert ISOTROPIC.to_voxels(micrometres).tolist() == [[1.0, 2.0, 4.0]]


def test_to_voxels_inverts_to_micrometres():
    voxels = np.array([[3, 7, 11]])
    assert np.allclose(ISOTROPIC.to_voxels(ISOTROPIC.to_micrometres(voxels)), voxels)


def test_anisotropic_radius():
    """A physical radius covers fewer voxels along the coarse axis than along the fine ones."""
    assert ISOTROPIC.anisotropic_radius(4.0).tolist() == [2.0, 8.0, 8.0]

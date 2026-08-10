import numpy as np

from celltrack.data.targets import DetectionTarget
from core.geometry import Spacing

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


def a_dark_volume(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_build():
    """The heatmap peaks at an annotated centre and the mask supervises it."""
    centres = np.array([[8, 8, 8]])
    target = DetectionTarget.build(centres, a_dark_volume((16, 16, 16)), ISOTROPIC, scale_um=2.0)
    assert target.heatmap[8, 8, 8] == 1.0
    assert bool(target.mask[8, 8, 8])


def test_build_excludes_bright_unlabelled_regions():
    """A bright blob with no annotation is the ambiguous case — excluded from the mask, not called background."""
    volume = a_dark_volume((16, 16, 16))
    volume[2, 2, 2] = 0.9  # bright, but far from the single annotated centre
    target = DetectionTarget.build(np.array([[8, 8, 8]]), volume, ISOTROPIC, scale_um=2.0)
    assert not bool(target.mask[2, 2, 2])


def test_build_supervises_dark_background():
    """Clearly dark tissue far from any centre is a safe negative — supervised, so the model learns to be quiet."""
    target = DetectionTarget.build(np.array([[8, 8, 8]]), a_dark_volume((16, 16, 16)), ISOTROPIC, scale_um=2.0)
    assert bool(target.mask[2, 2, 2])
    assert target.heatmap[2, 2, 2] < 0.01


def test_supervised_fraction():
    """With an all-dark volume and one centre, nearly everything is supervised (dark negatives plus the peak)."""
    target = DetectionTarget.build(np.array([[8, 8, 8]]), a_dark_volume((16, 16, 16)), ISOTROPIC, scale_um=2.0)
    assert target.supervised_fraction() == 1.0

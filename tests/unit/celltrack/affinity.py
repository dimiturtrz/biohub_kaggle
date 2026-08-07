import numpy as np

from celltrack.affinity import EdgeAffinity


def test_probabilities():
    """A conforming implementation satisfies the `EdgeAffinity` contract: a source×target matrix per scored gap,
    `None` for an unscored one — the shape the linker drops onto its cost."""

    class _Scored:
        def probabilities(self, timepoint: int) -> np.ndarray | None:
            return np.zeros((2, 3), dtype=np.float32) if timepoint == 0 else None

    affinity: EdgeAffinity = _Scored()
    assert affinity.probabilities(0).shape == (2, 3)
    assert affinity.probabilities(5) is None

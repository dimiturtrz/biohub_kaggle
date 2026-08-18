import numpy as np

from celltrack.linkers.frame_gap import FrameGap


def test_sweep():
    """Each consecutive frame pair becomes one gap — sources at `t`, targets at `t+1`, the last frame no gap."""
    timepoints = np.array([0, 0, 1, 2], dtype=np.int64)  # two cells at t=0, one each at t=1 and t=2
    positions_um = np.zeros((4, 3), dtype=np.float64)

    gaps = list(FrameGap.sweep(positions_um, timepoints))

    assert [gap.timepoint for gap in gaps] == [0, 1]  # no gap opens on the final frame
    assert gaps[0].sources.tolist() == [0, 1]
    assert gaps[0].targets.tolist() == [2]
    assert gaps[1].sources.tolist() == [2]
    assert gaps[1].targets.tolist() == [3]
    assert gaps[0].timepoints is timepoints  # the whole-video arrays pass through, not a slice
    assert gaps[0].positions_um is positions_um

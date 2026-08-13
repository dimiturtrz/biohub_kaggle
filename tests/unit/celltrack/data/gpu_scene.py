import torch

from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.synthetic_scene import SceneConfig


def test_batch():
    """A generated batch is a list of on-device PairSamples: two rendered frames, both centre sets, and the
    identity edge matrix that names each cell's own successor — the exact tensors the joint trainer consumes.

    Run on the CPU with a tiny config so the render (scatter + separable Gaussian conv) exercises the real path
    without a GPU; the device is a field, so 'cpu' is a faithful stand-in for 'cuda'.
    """
    config = SceneConfig(n_cells=8, volume_shape=(16, 16, 16))
    scenes = GpuScenes(config, "cpu")
    generator = torch.Generator("cpu").manual_seed(0)

    pairs = scenes.batch(3, generator)

    assert len(pairs) == 3
    sample = pairs[0]
    assert sample.frame_t.shape == (16, 16, 16) and sample.frame_t1.shape == (16, 16, 16)
    assert sample.source_centres.shape == (8, 3) and sample.target_centres.shape == (8, 3)
    assert sample.edge_matrix.shape == (8, 8)
    assert torch.equal(sample.edge_matrix, torch.eye(8))  # each cell continues to itself
    assert float(sample.frame_t.max()) > 0.5  # blobs rendered

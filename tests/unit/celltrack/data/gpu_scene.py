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


def test_confusor_construction_poses_near_rivals():
    """On the GPU path, `confusor_rate` raises the share of sources with a next-frame node nearer than their true
    successor (the source-vs-target confusor `hard_fraction` names). rate=0 is the sparse uniform baseline. Run on
    'cpu' (the device is a field). The centres are lifted by (1,4,4), so divide back before measuring um."""

    def confusor_rate(rate: float) -> float:
        config = SceneConfig(n_cells=200, volume_shape=(48, 48, 48), confusor_rate=rate)
        generator = torch.Generator("cpu").manual_seed(0)
        sample = GpuScenes(config, "cpu").batch(1, generator)[0]
        lift = torch.tensor([1.0, 4.0, 4.0])
        source = sample.source_centres.float() / lift
        target = sample.target_centres.float() / lift
        distances = torch.cdist(source, target) * config.spacing_um
        own = distances.diagonal()
        in_gate = own <= 10.0
        hard = in_gate & (distances.argmin(dim=1) != torch.arange(len(own)))
        return hard.sum().item() / in_gate.sum().item()

    assert confusor_rate(0.0) < 0.15  # uniform placement barely poses it
    assert confusor_rate(0.4) > 0.25  # construction poses it


def test_appearance_diversity_on_gpu_path():
    """The GPU render honours the same appearance knobs as the CPU scene: intensity spread widens the peak
    range, and background + noise lift the empty floor off zero. A clean config renders identical blobs on a
    zero floor. Run on 'cpu' (the device is a field) so no GPU is required."""
    clean = SceneConfig(n_cells=6, volume_shape=(16, 16, 16))
    diverse = SceneConfig(
        n_cells=6,
        volume_shape=(16, 16, 16),
        intensity_log_std=0.5,
        background_level=0.05,
        noise_read_std=0.02,
    )
    generator = torch.Generator("cpu").manual_seed(0)
    clean_frame = GpuScenes(clean, "cpu").batch(1, generator)[0].frame_t
    generator = torch.Generator("cpu").manual_seed(0)
    diverse_frame = GpuScenes(diverse, "cpu").batch(1, generator)[0].frame_t

    assert float(clean_frame.min()) == 0.0  # clean render leaves the empty floor at zero
    assert float(diverse_frame.mean()) > float(clean_frame.mean())  # background + noise lift the floor
    assert float(diverse_frame.max()) > float(clean_frame.max())  # intensity spread brightens the top cell

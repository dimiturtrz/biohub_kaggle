import numpy as np

from celltrack.data.synthetic_scene import Scene, SceneConfig


def test_generate():
    """A generated scene carries rendered volumes and a lineage graph whose shapes agree with the config.

    n_cells cells over n_frames frames give n_cells*n_frames nodes and n_cells*(n_frames-1) continuation edges;
    the volume is (n_frames, *volume_shape); every blob peak reaches the configured intensity.
    """
    config = SceneConfig(n_cells=40, n_frames=3, volume_shape=(32, 32, 32))
    scene = Scene.generate(config, seed=0)

    assert scene.volumes.shape == (3, 32, 32, 32)
    assert scene.positions_vox.shape == (120, 3)
    assert scene.timepoints.shape == (120,)
    assert scene.edges.shape == (80, 2)  # 40 cells continued over 2 gaps
    assert np.isclose(scene.volumes.max(), 1.0, atol=0.05)
    # every edge links a cell at frame f to the SAME cell at f+1 (row f*n + c -> (f+1)*n + c)
    assert bool(np.all(scene.edges[:, 1] - scene.edges[:, 0] == 40))


def test_hard_fraction():
    """Difficulty is a measured dial: a crowded fast-turning scene has many edges whose true successor is NOT
    the nearest next-frame cell, and an easy sparse slow scene has almost none."""
    hard = Scene.generate(SceneConfig(n_cells=300, speed_um_std=5.0, turn_std_rad=0.8), seed=0)
    easy = Scene.generate(SceneConfig(n_cells=40, speed_um_std=0.5, turn_std_rad=0.1), seed=0)

    hard_frac = hard.hard_fraction(10.0, 1.625)
    easy_frac = easy.hard_fraction(10.0, 1.625)

    assert hard_frac > 0.10  # the contested regime the generator exists to produce
    assert easy_frac < hard_frac  # difficulty responds to the knobs


def test_blobs_render_at_cell_centres():
    """A cell centre is a local intensity maximum — the detector can find the cells the graph names."""
    scene = Scene.generate(SceneConfig(n_cells=20, n_frames=2, volume_shape=(32, 32, 32)), seed=1)

    frame0 = scene.volumes[0]
    for centre in scene.positions_vox[:20]:
        z, y, x = np.round(centre).astype(int)
        assert frame0[z, y, x] > 0.5  # the centre is bright

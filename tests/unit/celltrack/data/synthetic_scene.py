import numpy as np

from celltrack.data.synthetic_scene import Scene, SceneConfig, SceneCorpus, SceneFrames


def test_scene_generate():
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


def test_scene_corpus_generate():
    """SceneCorpus.generate flattens N scenes into every consecutive-frame pair they hold (n_scenes * gaps)."""
    corpus = SceneCorpus.generate(SceneConfig(n_cells=10, n_frames=3, volume_shape=(16, 16, 16)), n_scenes=4, seed=0)
    assert len(corpus) == 8  # 4 scenes * 2 gaps
    assert int(corpus[0].edge_matrix.sum()) == 10


def test_hard_fraction():
    """Difficulty is a measured dial driven by DENSITY (faithful motion makes near-rivals rare, as in reality):
    a crowded scene has edges whose true successor is NOT the nearest next-frame cell, an easy sparse slow scene
    has almost none. The contested case is genuine geometric crowding, not overspeed chaos."""
    hard = Scene.generate(SceneConfig(n_cells=1500, speed_lognorm_sigma=0.7, turn_std_rad=0.8), seed=0)
    easy = Scene.generate(SceneConfig(n_cells=40, speed_lognorm_sigma=0.15, turn_std_rad=0.1), seed=0)

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


def _peaks(scene: Scene, n: int) -> np.ndarray:
    frame = scene.volumes[0]
    return np.array([frame[tuple(np.round(c).astype(int))] for c in scene.positions_vox[:n]])


def test_intensity_diversity():
    """`intensity_log_std` spreads the per-cell peak — a uniform blob has near-zero peak spread, a diverse one
    a wide one, so a detector cannot lean on a single fixed brightness."""
    uniform = Scene.generate(SceneConfig(n_cells=50, n_frames=2, volume_shape=(32, 32, 32)), seed=3)
    diverse = Scene.generate(
        SceneConfig(n_cells=50, n_frames=2, volume_shape=(32, 32, 32), intensity_log_std=0.4), seed=3
    )
    assert _peaks(diverse, 50).std() > 5.0 * _peaks(uniform, 50).std()


def test_background_and_noise_lift_the_floor():
    """Background + read noise lift empty voxels off zero — the detector sees a peak over a noisy floor, not a
    trivially separable noise-free blob. A clean scene leaves those voxels at exactly zero."""
    clean = Scene.generate(SceneConfig(n_cells=8, n_frames=2, volume_shape=(32, 32, 32)), seed=4)
    noisy = Scene.generate(
        SceneConfig(n_cells=8, n_frames=2, volume_shape=(32, 32, 32), background_level=0.05, noise_read_std=0.02),
        seed=4,
    )
    corner_clean = clean.volumes[0, :4, :4, :4]  # far from any cell
    corner_noisy = noisy.volumes[0, :4, :4, :4]
    assert float(corner_clean.max()) == 0.0
    assert float(corner_noisy.mean()) > 0.02  # lifted by background + noise


def test_anisotropic_psf_elongates_z():
    """`sigma_z_ratio` > 1 blurs z more than x/y — a lone blob spans more voxels above half-max along z than x."""
    config = SceneConfig(n_cells=1, n_frames=2, volume_shape=(40, 40, 40), margin_vox=18.0, sigma_z_ratio=2.5)
    scene = Scene.generate(config, seed=0)
    z, y, x = np.round(scene.positions_vox[0]).astype(int)
    frame = scene.volumes[0]
    half = frame[z, y, x] / 2.0
    z_span = int((frame[:, y, x] > half).sum())
    x_span = int((frame[z, y, :] > half).sum())
    assert z_span > x_span  # z is the anisotropic (blurrier) axis


def test_track_graph():
    """The lineage as a TrackGraph with node coordinates LIFTED into the raw grid (y, x by 4) the pipeline reads."""
    scene = Scene.generate(SceneConfig(n_cells=10, n_frames=2, volume_shape=(16, 32, 32)), seed=0)

    graph = scene.track_graph()

    assert graph.coordinates.shape == (20, 4)  # (t, z, y, x) per node
    assert bool(np.all(graph.coordinates[:, 0] == scene.timepoints))  # t column preserved
    # y and x lifted by 4 (the (1,4,4) downsample the volumes were rendered at), z and t unchanged
    assert bool(np.all(graph.coordinates[:, 2] == np.rint(scene.positions_vox[:, 1] * 4)))
    assert bool(np.all(graph.coordinates[:, 1] == np.rint(scene.positions_vox[:, 0])))


def test_pair_targets():
    """Each consecutive-frame gap becomes a PairTarget whose edge matrix names every cell's own successor."""
    scene = Scene.generate(SceneConfig(n_cells=12, n_frames=3, volume_shape=(16, 32, 32)), seed=1)

    targets = scene.pair_targets()

    assert len(targets) == 2  # 3 frames -> 2 gaps
    target = targets[0]
    assert int(target.edge_matrix.sum()) == 12  # every cell continues once
    frame = target.frames.frame(target.timepoint, (1, 4, 4))
    assert frame.shape == (16, 32, 32)


def test_of():
    """SceneFrames.of precomputes the intensity window (low <= high) so a per-frame read is a subtract-and-scale."""
    scene = Scene.generate(SceneConfig(n_cells=8, n_frames=2, volume_shape=(16, 16, 16)), seed=2)

    frames = SceneFrames.of(scene.volumes)

    assert frames.q_low <= frames.q_high
    assert frames.volumes.shape == (2, 16, 16, 16)


def test_frame():
    """SceneFrames precomputes the quantile window once (`of`), then a frame read is a subtract-and-scale that
    clamps to non-negative and matches the shipped normalisation, refusing a downsample the coords don't live on."""
    scene = Scene.generate(SceneConfig(n_cells=8, n_frames=2, volume_shape=(16, 16, 16)), seed=2)

    frames = SceneFrames.of(scene.volumes)
    frame = frames.frame(0, (1, 4, 4))

    assert frame.shape == (16, 16, 16)
    assert float(frame.min()) >= 0.0  # clamped non-negative

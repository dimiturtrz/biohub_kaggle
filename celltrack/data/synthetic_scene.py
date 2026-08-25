"""A procedural cell-scene generator — hard association examples with a KNOWN answer.

The static synthetic corpus (`synthetic_pairs`) is too easy: its cells are well-separated, so the true
successor is always distinguishable and the head learns nothing it does not already know. The real failure —
the dense mislinks — is the opposite: a fast-moving cell whose true successor lands FAR while a slow neighbour
sits NEAR, so appearance and proximity both point at the wrong target. On REAL data that case is unlabelable by
us (we cannot tell true from decoy at the resolution); GENERATED, it is fully known, because we wrote the
trajectory.

This constructs exactly that case on demand. Cells are placed CROWDED, given per-axis velocities with a
fast-mover tail and per-frame DIRECTION CHANGES (the specific thing that defeats a constant-velocity prior),
advanced frame by frame, and rendered as Gaussian blobs on the same isotropic 1.625um / 64^3 grid the detector
sees. The emitted `TrackGraph` carries the true `t -> t+1` continuation for every cell, so the association head
can be trained to separate a case that LOOKS ambiguous — and the only cue that resolves it is the motion
history, which is why this pairs with the temporal-position / velocity features rather than appearance.

WHAT IT DOES NOT CLAIM. Rendered blobs are not real nuclei; the appearance distribution differs, so this is not
a detector-quality substitute (that is the static-synthetic domain gap, unchanged). The bet is narrower: the
RULE "when appearance is ambiguous, follow the trajectory" is about motion, not appearance, so a head that
learns it on generated hard cases may apply it on real ones. That bet is what a training run measures; this
module only manufactures the cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.data.joint_dataset import PairTarget
from core.data.tracks import TrackGraph

# The generated volumes are ALREADY on the isotropic 64^3 post-downsample grid, but the pipeline reads pooled
# frames against RAW-grid node coordinates divided by this factor — so scene node positions are lifted into the
# raw grid (y, x times 4; z unchanged) and a run must use this downsample, exactly as the static synthetic does.
_SCENE_DOWNSAMPLE: tuple[int, int, int] = (1, 4, 4)
_Q_LOW, _Q_HIGH = 0.001, 0.999
_MIN_FRAMES_FOR_HEADING = 3  # inversion confusor needs t-1, t, t+1: two prior frames to establish a heading


@dataclass(frozen=True)
class SceneConfig:
    """The difficulty knobs of one generated sequence — every one is an argument, none a magic constant.

    Defaults target the dense-movie regime: ~700 cells in a 64^3 isotropic volume (crowded), a FAITHFUL
    lognormal step (median 1.7um, real fast-mover tail), and a per-frame turn spread that makes a
    constant-velocity prediction wrong often enough to force the head onto the full trajectory. Difficulty is
    density- and appearance-driven: raising `n_cells` (genuine near-rivals) or `speed_lognorm_sigma` (tail) or
    `turn_std_rad` makes the successor harder; lowering them recovers the easy static-synthetic case. The
    confusor is NOT manufactured by overspeed — that produced unlearnable chaos (see the speed comment below).
    """

    volume_shape: tuple[int, int, int] = (64, 64, 64)
    spacing_um: float = 1.625  # the isotropic post-downsample grid the detector reads
    n_frames: int = 4
    # Defaults are the DENSE-HARD regime, chosen by measurement: at ~700 cells / 64^3 (the dense movie's own
    # density) with a faithful lognormal step and large turns, in-gate edges get genuine near-rivals from
    # DENSITY (not overshoot). Lowering n_cells / speed_lognorm_sigma / turn_std_rad recovers the easy
    # static-synthetic case, so difficulty is a dial, not a fixed property.
    n_cells: int = 700
    blob_sigma_vox: float = 1.4  # a nucleus is ~2-3 voxels across on this grid
    peak_intensity: float = 1.0
    # Per-frame displacement is LOGNORMAL, derived from the real annotated corpus (card-free audit 2026-08-24):
    # real step um med 1.68 / p90 2.87 / p99 5.02 / max 7.81. Physics = slow diffusion + a directed fast-mover
    # tail -> multiplicative, heavy-tailed -> lognormal (a half-normal is too light-tailed for the real max).
    # median = speed_um_mean (exp(mu)); speed_lognorm_sigma sets the tail. sigma 0.46 reproduces p99 4.9 / max
    # 7.3. The prior additive half-normal (speed_um_std=6.0) ran ~3.3x too fast (synth median 5.6um > real MAX)
    # and manufactured 48% rival-nearer OVERSHOOT chaos (real ~0.1%) — an unlearnable confusor that poisoned the
    # association head (eqdx gap-widened). The REAL confusor is density- + appearance-driven, not speed-driven.
    speed_um_mean: float = 1.7  # the measured median annotated step = lognormal median exp(mu)
    speed_lognorm_sigma: float = 0.46  # lognormal shape; the faithful fast-mover tail (fit to real p99/max)
    turn_std_rad: float = 0.8  # per-frame heading change; the constant-velocity prior fails when this is large
    margin_vox: float = 3.0  # keep centres off the volume face so a blob is not clipped
    # APPEARANCE DIVERSITY. The blob above is one identical, noise-free shape — trivially detectable, so a
    # detector trained on it never has to learn the real signal and does not transfer. These break that: real
    # nuclei vary in brightness, the microscope PSF is z-anisotropic, and the frame carries background + noise.
    # Values live on the quantile-normalised [0, 1] scale the frames are read at (q0.001/q0.999 window). All
    # default to the identical-blob regime (off) so an existing run is unchanged; an appearance-diverse run
    # turns them on. Per-cell SIZE variation (`size_log_std`) is available on this numpy render path — measured
    # against the josefreitas faithful set, cell size (volume_um3) was the dominant single-cell fidelity gap
    # (KS D~0.79, no other knob touches it). It stays off on the separable-blur GPU path, which needs one shared
    # kernel for conv3d speed; brightness + anisotropy + noise remain the free knobs shared by both paths.
    intensity_log_std: float = 0.0  # per-cell peak = peak_intensity * exp(N(0, s)); nucleus brightness spread
    size_log_std: float = 0.0  # per-cell sigma = blob_sigma_vox * exp(N(0, s)); nucleus SIZE spread (numpy path)
    sigma_z_ratio: float = 1.0  # z sigma = blob_sigma_vox * ratio; microscope PSF blurs z more than x/y
    background_level: float = 0.0  # additive constant baseline before noise
    noise_read_std: float = 0.0  # additive Gaussian read noise std (post-blur, [0, 1] scale)
    noise_shot_photons: float = 0.0  # Poisson shot noise; photons at unit signal (0 = off, higher = cleaner)
    # CONFUSOR CONSTRUCTION. Uniform placement poses the association confusor (a next-frame node nearer the
    # source than its true successor — the `hard_fraction` case) only ~4% of the time: density alone almost never
    # puts a rival inside the true step, so the head trains on trivially separable edges and P_true never moves
    # (joint_confusor_synth_v1, card-free audit 2026-08-24). These CONSTRUCT it instead of hoping density emits
    # it. For `confusor_rate` of cells: boost the cell to a fast-mover (own step >= min_step, true successor
    # departs FAR) and place a SLOW distractor within `radius_um` of its start (the distractor stays put, so its
    # next-frame node sits near the source as a false near-successor). The source->distractor pair is a LABELED
    # hard negative real annotation cannot give us. rate=0 (default) is the plain uniform generator, unchanged.
    confusor_rate: float = 0.0  # fraction of cells made hard sources (each consumes one distractor; rate <= 0.5)
    confusor_radius_um: float = 1.6  # distractor placed within this of the source start (~real mislink sep, 1 vox)
    confusor_min_step_um: float = 3.0  # a source's forced step (>= radius, so rival <= own); real p90 2.87/p99 5.0
    confusor_distractor_step_um: float = 0.5  # distractor's slow step, so it stays a near-source rival (real p10)
    # VELOCITY-INVERSION mode. The default construction above is velocity-CONTINUOUS: the source's true successor
    # departs ALONG its heading (align ~ +1), so a velocity prior SOLVES it (synth oracle 0.634). The REAL dense
    # confusor is velocity-DEFEATING: the true successor departs OFF the source's heading (measured align ~ -0.17,
    # own motion ~1um << 6.4um gap) while the near-static rival sits ON the old-trajectory extrapolation, exactly
    # where a velocity prior predicts the source lands. Training on the continuous confusor teaches a rule the
    # head solves with velocity and that inverts on real data (v2 null, 2026-08-24). This mode poses the faithful
    # case: at the last transition a confusor source jumps off-heading (align ~ -confusor_anti_align) and its
    # paired distractor is the static cell on the predicted old-trajectory landing.
    confusor_invert_velocity: bool = False
    confusor_anti_align: float = 0.17  # target -cos(true_step, incoming heading); real dense mislinks ~ 0.17


# Appearance-faithful preset. One named decision, not a per-run sweep: the values are frozen from the KS fit
# against the josefreitas set (eqdx audit 2026-08-24). They kill the identical-blob degeneracy that made prior
# synth-training refutations shallow — size SPREAD + z-shape + brightness/noise — without chasing a fragile
# multi-knob KS-tie (size and confusor density trade off physically; volume_um3 KS D 0.79 -> 0.52 is enough to
# remove the degeneracy). n_cells / speed / turn keep their dense-hard defaults: the confusor axis josef lacks.
FAITHFUL_APPEARANCE = SceneConfig(
    blob_sigma_vox=2.2,
    size_log_std=0.25,
    sigma_z_ratio=1.0,
    intensity_log_std=0.4,
    background_level=0.12,
    noise_read_std=0.04,
    noise_shot_photons=150.0,
)


@dataclass(frozen=True)
class Scene:
    """One generated sequence: the rendered volumes and the ground-truth lineage over them."""

    volumes: Float[np.ndarray, "t z y x"]
    positions_vox: Float[np.ndarray, "n 3"]  # every node's (z, y, x) centre, full-resolution voxels of this grid
    timepoints: Int[np.ndarray, " n"]
    edges: Int[np.ndarray, "e 2"]  # true t -> t+1 continuations, as row indices into positions_vox

    @classmethod
    def generate(cls, config: SceneConfig, seed: int) -> "Scene":
        """Place crowded cells, move them (fast-movers + turns), render blobs, and record the true edges."""
        rng = np.random.default_rng(seed)
        lo, hi = cls._box(config)

        # Per-cell state: a start position and an initial velocity, the velocity in VOXELS per frame so it is
        # commensurate with the grid the blobs render on (um / um-per-voxel = voxels).
        start = rng.uniform(lo, hi, size=(config.n_cells, 3))
        mu = math.log(config.speed_um_mean)
        speed_vox = np.exp(mu + config.speed_lognorm_sigma * rng.normal(0.0, 1.0, size=(config.n_cells, 1)))
        speed_vox /= config.spacing_um
        velocity = rng.normal(0.0, 1.0, size=(config.n_cells, 3))
        velocity *= speed_vox / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-9)

        # Two confusor constructions. The default (velocity-CONTINUOUS) is posed on the initial velocity before
        # the walk; the inversion (velocity-DEFEATING, faithful to real dense mislinks) is posed on the built
        # trajectory afterwards, because it needs the source's heading history to depart from.
        if config.confusor_invert_velocity:
            positions = cls._advance(start, velocity, config, rng)
            cls._apply_confusor_inversion(positions, config, rng)
        else:
            start, velocity = cls._construct_confusors(start, velocity, config, rng)
            positions = cls._advance(start, velocity, config, rng)

        # A cell keeps its brightness across the sequence (a nucleus does not flicker), so the per-cell peak is
        # drawn once here and indexed by cell in every frame's render.
        intensities = config.peak_intensity * np.exp(rng.normal(0.0, config.intensity_log_std, config.n_cells))

        positions_vox = np.concatenate(positions, axis=0)
        timepoints = np.repeat(np.arange(config.n_frames), config.n_cells)
        # Cell c at frame f is row f*n_cells + c; its continuation is the same cell at f+1.
        edges = np.array(
            [
                [f * config.n_cells + c, (f + 1) * config.n_cells + c]
                for f in range(config.n_frames - 1)
                for c in range(config.n_cells)
            ],
            dtype=np.int64,
        )
        volumes = cls._render(positions, config, intensities, rng)
        return cls(volumes=volumes, positions_vox=positions_vox, timepoints=timepoints, edges=edges)

    @staticmethod
    def _construct_confusors(
        start: Float[np.ndarray, "n 3"],
        velocity: Float[np.ndarray, "n 3"],
        config: SceneConfig,
        rng: np.random.Generator,
    ) -> tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 3"]]:
        """Turn a `confusor_rate` share of cells into labelled hard sources (fast) + slow near-source distractors.

        The canonical difficulty is `hard_fraction`: an edge is hard when a next-frame node sits nearer the
        SOURCE than the true successor does. Uniform placement almost never produces it, so we place it: pick
        disjoint (source, distractor) pairs, force the source to a real fast-mover step (its true successor
        departs), and drop the distractor within `radius_um` of the source with a near-zero step so its
        next-frame node stays put — a false near-successor that competes on proximity. Both edges stay truthfully
        labelled (each cell to itself), so the source->distractor candidate is a KNOWN negative. rate=0 is a no-op.
        """
        if config.confusor_rate <= 0.0:
            return start, velocity
        n_pairs = min(round(config.confusor_rate * config.n_cells), config.n_cells // 2)
        if n_pairs == 0:
            return start, velocity
        picked = rng.permutation(config.n_cells)[: 2 * n_pairs]
        src, dis = picked[:n_pairs], picked[n_pairs:]
        lo, hi = config.margin_vox, np.asarray(config.volume_shape, dtype=np.float64) - config.margin_vox
        min_step_vox = config.confusor_min_step_um / config.spacing_um
        speed = np.linalg.norm(velocity[src], axis=1, keepdims=True)
        velocity[src] = velocity[src] / (speed + 1e-9) * np.maximum(speed, min_step_vox)  # depart FAR
        offset = rng.normal(0.0, 1.0, size=(n_pairs, 3))
        offset /= np.linalg.norm(offset, axis=1, keepdims=True) + 1e-9
        radius = rng.uniform(0.0, config.confusor_radius_um / config.spacing_um, size=(n_pairs, 1))
        start[dis] = np.clip(start[src] + offset * radius, lo, hi)  # sit next to the source
        slow = np.linalg.norm(velocity[dis], axis=1, keepdims=True)
        slow_vox = config.confusor_distractor_step_um / config.spacing_um
        velocity[dis] = velocity[dis] / (slow + 1e-9) * slow_vox  # stay put -> false near-successor
        return start, velocity

    @staticmethod
    def _box(config: SceneConfig) -> tuple[Float[np.ndarray, " 3"], Float[np.ndarray, " 3"]]:
        """The (lo, hi) voxel bounds a cell centre is clipped to — the volume inset by the render margin."""
        shape = np.asarray(config.volume_shape, dtype=np.float64)
        return np.full(3, config.margin_vox, dtype=np.float64), shape - config.margin_vox

    @classmethod
    def _advance(
        cls,
        start: Float[np.ndarray, "n 3"],
        velocity: Float[np.ndarray, "n 3"],
        config: SceneConfig,
        rng: np.random.Generator,
    ) -> list[Float[np.ndarray, "n 3"]]:
        """Walk every cell forward n_frames-1 steps under its (turning) velocity, clipping to the box."""
        lo, hi = cls._box(config)
        positions = [start]
        current = start
        for _ in range(1, config.n_frames):
            velocity = cls._turn(velocity, config.turn_std_rad, rng)
            current = np.clip(current + velocity, lo, hi)  # a cell that would leave stays at the face
            positions.append(current)
        return positions

    @classmethod
    def _apply_confusor_inversion(
        cls,
        positions: list[Float[np.ndarray, "n 3"]],
        config: SceneConfig,
        rng: np.random.Generator,
    ) -> None:
        """Pose the velocity-DEFEATING confusor in place: at the last transition a source jumps OFF its heading
        while its paired rival sits static on the old-trajectory extrapolation, where a velocity prior predicts.

        This is the faithful dense mislink ([[celltrack-relative-pe-radial-confusor-directional]]): the true
        successor's step is anti-correlated with the incoming heading (cos ~ -confusor_anti_align, real ~ -0.17),
        and own motion (~1um) is dwarfed by the off-trajectory jump. The default _construct_confusors poses the
        OPPOSITE (successor departs ALONG the heading), which a velocity prior solves and which every prior synth
        motion arm therefore learned nothing from. Needs >=3 frames so frames 0..t give the source a heading.
        """
        if config.confusor_rate <= 0.0 or config.n_frames < _MIN_FRAMES_FOR_HEADING:
            return
        n_pairs = min(round(config.confusor_rate * config.n_cells), config.n_cells // 2)
        if n_pairs == 0:
            return
        lo, hi = cls._box(config)
        picked = rng.permutation(config.n_cells)[: 2 * n_pairs]
        src, dis = picked[:n_pairs], picked[n_pairs:]
        t = config.n_frames - 2  # confusor transition t -> t+1
        incoming = positions[t][src] - positions[t - 1][src]
        inc_dir = incoming / (np.linalg.norm(incoming, axis=1, keepdims=True) + 1e-9)
        # True successor departs off-heading: a random direction with its along-heading part removed (cos ~ 0),
        # then tilted back so cos(step, incoming) = -anti_align exactly.
        rand = rng.normal(0.0, 1.0, size=(n_pairs, 3))
        perp = rand - (rand * inc_dir).sum(axis=1, keepdims=True) * inc_dir
        perp_dir = perp / (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9)
        anti = config.confusor_anti_align
        step_dir = perp_dir * math.sqrt(max(0.0, 1.0 - anti * anti)) - inc_dir * anti  # unit, cos with inc = -anti
        step_vox = config.confusor_min_step_um / config.spacing_um
        positions[t + 1][src] = np.clip(positions[t][src] + step_dir * step_vox, lo, hi)
        # Rival: static cell on the old-trajectory landing (where a constant-velocity prior extrapolates the src).
        old_landing = positions[t][src] + incoming
        offset = rng.normal(0.0, 1.0, size=(n_pairs, 3))
        offset /= np.linalg.norm(offset, axis=1, keepdims=True) + 1e-9
        radius = rng.uniform(0.0, config.confusor_radius_um / config.spacing_um, size=(n_pairs, 1))
        landing = np.clip(old_landing + offset * radius, lo, hi)
        for frame in positions:
            frame[dis] = landing  # fully static rival -> its own history never betrays it as the wrong successor

    @staticmethod
    def _turn(
        velocity: Float[np.ndarray, "n 3"], turn_std_rad: float, rng: np.random.Generator
    ) -> Float[np.ndarray, "n 3"]:
        """Rotate each velocity by a small random 3D angle — a persistent walk that occasionally changes heading.

        A Gaussian perturbation added to the unit direction and renormalised turns the heading without changing
        the speed, so `turn_std_rad` controls how badly a constant-velocity prediction misses while the step
        length distribution (the fast-mover tail) is preserved.
        """
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        direction = velocity / (speed + 1e-9)
        perturbed = direction + rng.normal(0.0, turn_std_rad, size=velocity.shape)
        return perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-9) * speed

    @staticmethod
    def _render(
        positions: list[Float[np.ndarray, "n 3"]],
        config: SceneConfig,
        intensities: Float[np.ndarray, " n"],
        rng: np.random.Generator,
    ) -> Float[np.ndarray, "t z y x"]:
        """Paint a Gaussian blob at every cell centre, per frame — the max over cells, then background + noise.

        Overlapping crowded cells are combined by MAXIMUM rather than addition, so two near cells read as two
        peaks of the right height instead of one bright merged blob — the crowding the detector must separate is
        preserved, and each peak stays calibrated to its own `intensities[cell]` regardless of density.

        Each blob is computed only inside a `+-radius` window around its centre (the Gaussian is negligible
        beyond ~3 sigma), so the cost is O(cells * window) rather than O(cells * volume) — the difference between
        a generator that can run per training step and one that cannot at the dense density this targets.

        The blob is z-anisotropic (`sigma_z_ratio`) to mimic the microscope PSF, and once every cell is painted
        the frame gets a constant `background_level` and Poisson-then-Gaussian noise — so the detector sees the
        real signal (a peak over a noisy floor), not a trivially separable noise-free blob.
        """
        base_sigma_axis = np.asarray(
            [config.blob_sigma_vox * config.sigma_z_ratio, config.blob_sigma_vox, config.blob_sigma_vox]
        )
        # Per-cell size: a nucleus's radius varies (lognormal), so `size_log_std` scales every axis of its blob
        # together (an isotropic size draw, not a new shape). Off (=0) collapses to one shared kernel, unchanged.
        scale = (
            np.exp(rng.normal(0.0, config.size_log_std, config.n_cells))
            if config.size_log_std > 0.0
            else np.ones(config.n_cells)
        )
        sigma_per_cell = base_sigma_axis[None, :] * scale[:, None]
        two_sigma_sq_per_cell = 2.0 * sigma_per_cell**2
        radius_per_cell = np.ceil(3.0 * sigma_per_cell).astype(int)
        shape = config.volume_shape
        volumes = np.zeros((len(positions), *shape), dtype=np.float32)
        for frame, centres in enumerate(positions):
            for cell, centre in enumerate(centres):
                radius = radius_per_cell[cell]
                two_sigma_sq_axis = two_sigma_sq_per_cell[cell]
                base = np.floor(centre).astype(int)
                lo = np.maximum(base - radius, 0)
                hi = np.minimum(base + radius + 1, shape)
                axes = [np.arange(lo[d], hi[d]) for d in range(3)]
                offsets = np.meshgrid(*axes, indexing="ij")
                squared = sum((offsets[d] - centre[d]) ** 2 / two_sigma_sq_axis[d] for d in range(3))
                blob = (intensities[cell] * np.exp(-squared)).astype(np.float32)
                window = volumes[frame, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
                np.maximum(window, blob, out=window)
        return Scene._corrupt(volumes, config, rng)

    @staticmethod
    def _corrupt(
        volumes: Float[np.ndarray, "t z y x"], config: SceneConfig, rng: np.random.Generator
    ) -> Float[np.ndarray, "t z y x"]:
        """Add a constant background then camera noise — Poisson shot noise, then Gaussian read noise.

        Shot noise scales with signal (variance = mean / photons at unit intensity), so bright cell cores are
        noisier in absolute terms but cleaner in SNR — the real camera behaviour. Read noise is a flat Gaussian
        floor. Both are off at their zero defaults, leaving the clean blob unchanged.
        """
        out = volumes + config.background_level
        if config.noise_shot_photons > 0.0:
            out = rng.poisson(np.clip(out, 0.0, None) * config.noise_shot_photons) / config.noise_shot_photons
        if config.noise_read_std > 0.0:
            out = out + rng.normal(0.0, config.noise_read_std, out.shape)
        return np.clip(out, 0.0, None).astype(np.float32)

    def hard_fraction(self, gate_um: float, spacing_um: float) -> float:
        """Share of `t -> t+1` edges whose true successor is NOT the nearest next-frame cell — the difficulty.

        This is the property the whole generator exists to produce: an edge is HARD when a different cell of the
        next frame sits closer to the source than the true successor does, so proximity points at the wrong
        target. Reported so a config's difficulty is a measured number, not an assumed one.
        """
        frames = int(self.timepoints.max()) + 1
        n_cells = len(self.timepoints) // frames
        hard = 0
        total = 0
        for frame in range(frames - 1):
            sources = self.positions_vox[frame * n_cells : (frame + 1) * n_cells] * spacing_um
            targets = self.positions_vox[(frame + 1) * n_cells : (frame + 2) * n_cells] * spacing_um
            distances = np.linalg.norm(sources[:, None, :] - targets[None, :, :], axis=-1)
            nearest = np.argmin(distances, axis=1)
            for cell in range(n_cells):
                if distances[cell, cell] <= gate_um:  # the true successor is in-gate (a linkable case)
                    total += 1
                    hard += int(nearest[cell] != cell)
        return hard / total if total else float("nan")

    def track_graph(self) -> TrackGraph:
        """The scene's lineage as a `TrackGraph`, node coordinates lifted into the RAW grid the pipeline expects.

        The volumes are on the 64^3 grid, but the trainer divides node positions by `_SCENE_DOWNSAMPLE` to reach
        the frame grid, so `(z, y, x)` are lifted by `(1, 4, 4)` to cancel that division and land back on the
        rendered voxel. `edges` are already row indices into the node array, which is what the enumeration reads.
        """
        lift = np.asarray([1, *_SCENE_DOWNSAMPLE], dtype=np.float64)  # (t, z, y, x); t and z unchanged, y/x by 4
        coordinates = np.column_stack([self.timepoints, self.positions_vox]) * lift
        return TrackGraph(
            node_ids=np.arange(len(self.timepoints), dtype=np.int64),
            coordinates=np.rint(coordinates).astype(np.int64),
            edges=self.edges,
        )

    def pair_targets(self) -> list[PairTarget]:
        """This scene as the consecutive-frame `PairTarget`s the joint trainer consumes — frames from its volumes."""
        return PairTarget.enumerate(self.track_graph(), SceneFrames.of(self.volumes))


@dataclass(frozen=True)
class SceneFrames:
    """A generated scene's volumes as a `FrameSource` — quantile-normalised the way a competition video's are."""

    volumes: Float[np.ndarray, "t z y x"]
    q_low: float
    q_high: float

    @classmethod
    def of(cls, volumes: Float[np.ndarray, "t z y x"]) -> "SceneFrames":
        """Precompute the intensity window once so a per-frame read is a subtract-and-scale, not a re-quantile."""
        low, high = np.quantile(volumes, [_Q_LOW, _Q_HIGH])
        return cls(volumes=volumes, q_low=float(low), q_high=float(high))

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """That frame, quantile-normalised and non-negative — already on the grid, so never strided again."""
        if tuple(downsample) != _SCENE_DOWNSAMPLE:
            raise ValueError(
                f"generated scenes render on the {_SCENE_DOWNSAMPLE}-equivalent 64^3 grid with raw-lifted nodes; "
                f"a run at downsample {tuple(downsample)} would read a grid the coordinates do not live on"
            )
        normed = (torch.from_numpy(self.volumes[timepoint]) - self.q_low) / (self.q_high - self.q_low + 1e-6)
        return normed.clamp(0.0)


class SceneCorpus:
    """A pool of generated scenes as the trainer's own `PairTarget`s — the dynamic synthetic corpus."""

    @staticmethod
    def generate(config: SceneConfig, n_scenes: int, seed: int) -> list[PairTarget]:
        """`n_scenes` scenes at the config's difficulty, flattened into every consecutive-frame pair they hold.

        Each scene is a fresh seed, so the pool is varied; the volumes stay resident (referenced by their
        `SceneFrames`) so the loop reads them without a re-decode, at ~4 MB per scene.
        """
        targets: list[PairTarget] = []
        for index in range(n_scenes):
            targets.extend(Scene.generate(config, seed + index).pair_targets())
        return targets

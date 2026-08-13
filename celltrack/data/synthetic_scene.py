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
from jaxtyping import Float, Int


@dataclass(frozen=True)
class SceneConfig:
    """The difficulty knobs of one generated sequence — every one is an argument, none a magic constant.

    Defaults target the dense-movie regime: ~80 cells in a 64^3 isotropic volume (crowded), a mean step near
    the measured 1.7um with a heavy fast-mover tail, and a per-frame turn spread that makes a constant-velocity
    prediction wrong often enough to force the head onto the full trajectory. Raising `n_cells`, `speed_um_std`
    or `turn_std_rad` makes the successor harder to pick; lowering them recovers the easy static-synthetic case.
    """

    volume_shape: tuple[int, int, int] = (64, 64, 64)
    spacing_um: float = 1.625  # the isotropic post-downsample grid the detector reads
    n_frames: int = 4
    # Defaults are the DENSE-HARD regime, chosen by measurement: at ~700 cells / 64^3 (the dense movie's own
    # density) with a fast-mover tail and large turns, ~36% of in-gate edges have the true successor NOT nearest
    # — the contested case the real corpus lacks. Lowering n_cells / speed_um_std / turn_std_rad recovers the
    # easy static-synthetic case, so difficulty is a dial, not a fixed property.
    n_cells: int = 700
    blob_sigma_vox: float = 1.4  # a nucleus is ~2-3 voxels across on this grid
    peak_intensity: float = 1.0
    speed_um_mean: float = 1.7  # the measured median annotated step
    speed_um_std: float = 6.0  # the fast-mover tail — where the true successor lands far
    turn_std_rad: float = 0.8  # per-frame heading change; the constant-velocity prior fails when this is large
    margin_vox: float = 3.0  # keep centres off the volume face so a blob is not clipped


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
        shape = np.asarray(config.volume_shape, dtype=np.float64)
        lo, hi = config.margin_vox, shape - config.margin_vox

        # Per-cell state: a start position and an initial velocity, the velocity in VOXELS per frame so it is
        # commensurate with the grid the blobs render on (um / um-per-voxel = voxels).
        start = rng.uniform(lo, hi, size=(config.n_cells, 3))
        step_vox = config.speed_um_mean / config.spacing_um
        spread_vox = config.speed_um_std / config.spacing_um
        velocity = rng.normal(0.0, 1.0, size=(config.n_cells, 3))
        velocity *= (step_vox + np.abs(rng.normal(0.0, spread_vox, size=(config.n_cells, 1)))) / (
            np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-9
        )

        positions = [start]
        current = start
        for _ in range(1, config.n_frames):
            velocity = cls._turn(velocity, config.turn_std_rad, rng)
            current = np.clip(current + velocity, lo, hi)  # a cell that would leave stays at the face
            positions.append(current)

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
        volumes = cls._render(positions, config)
        return cls(volumes=volumes, positions_vox=positions_vox, timepoints=timepoints, edges=edges)

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
    def _render(positions: list[Float[np.ndarray, "n 3"]], config: SceneConfig) -> Float[np.ndarray, "t z y x"]:
        """Paint an isotropic Gaussian blob at every cell centre, per frame — the max over cells, not the sum.

        Overlapping crowded cells are combined by MAXIMUM rather than addition, so two near cells read as two
        peaks of the right height instead of one bright merged blob — the crowding the detector must separate is
        preserved, and the peak intensity stays calibrated to `peak_intensity` regardless of density.

        Each blob is computed only inside a `+-radius` window around its centre (the Gaussian is negligible
        beyond ~3 sigma), so the cost is O(cells * window) rather than O(cells * volume) — the difference between
        a generator that can run per training step and one that cannot at the dense density this targets.
        """
        radius = math.ceil(3.0 * config.blob_sigma_vox)
        two_sigma_sq = 2.0 * config.blob_sigma_vox**2
        shape = config.volume_shape
        volumes = np.zeros((len(positions), *shape), dtype=np.float32)
        for frame, centres in enumerate(positions):
            for centre in centres:
                base = np.floor(centre).astype(int)
                lo = np.maximum(base - radius, 0)
                hi = np.minimum(base + radius + 1, shape)
                axes = [np.arange(lo[d], hi[d]) for d in range(3)]
                offsets = np.meshgrid(*axes, indexing="ij")
                squared = sum((offsets[d] - centre[d]) ** 2 for d in range(3))
                blob = (config.peak_intensity * np.exp(-squared / two_sigma_sq)).astype(np.float32)
                window = volumes[frame, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
                np.maximum(window, blob, out=window)
        return volumes

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

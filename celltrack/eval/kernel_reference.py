"""Run a published Kaggle kernel's OWN function bytes without importing its notebook scaffolding.

The fidelity oracle proves our reimplemented postproc makes the same decisions as the published kernel by
diffing our stage against the kernel's actual function. That requires executing the kernel's code — but the
reference is a flattened notebook whose import-time code installs a heavy dependency stack (tracksdata, zarr,
ilpy, pyscipopt). `KernelReference` parses the reference file with `ast` and execs ONLY the named function's
own source segment in a namespace seeded with the kernel's constants and its small pure-geometry helpers, so
the diff runs the kernel's ACTUAL bytes with nothing transcribed but the constants (which ARE the fidelity
subject and are cited to the reference line that sets them). It also casts a `TrackGraph` into the node/edge
dicts those bytes consume.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.data.tracks import TrackGraph
from core.geometry import Spacing

# The imaging's voxel-to-micrometre scale, identical on both sides: kernel `VOXEL_SCALE_UM` (divsub reference
# line 1238) and our `Spacing`. Named here so the oracle's own geometry and the kernel namespace share one value.
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
SPACING = Spacing(z=VOXEL_SCALE_UM[0], y=VOXEL_SCALE_UM[1], x=VOXEL_SCALE_UM[2])

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "research" / "frontier_kernels"
DIVSUB_REFERENCE = REFERENCE_ROOT / "rockerritesh_0-926-biohub-divsub" / "0-926-biohub-divsub_code.py"
MOTION_REFERENCE = REFERENCE_ROOT / "evgendvorkin_biohub-0-927-lb" / "biohub-0-927-lb_code.py"


class KernelReference:
    """Extract-and-exec bridge to a flattened Kaggle kernel: run the kernel's OWN function bytes without
    importing its notebook scaffolding, and cast a `TrackGraph` into the node/edge dicts those bytes consume.

    Every method here transcribes NOTHING but the constants the kernel sets — the functions themselves are
    exec'd from the reference file, so the diff runs the frontier's actual decision logic."""

    @staticmethod
    def kernel_namespace(path: Path, function_names: list[str], constants: dict[str, object]) -> dict[str, object]:
        """Exec named top-level functions OUT of a kernel reference file, WITHOUT importing the module.

        The reference is a flattened notebook whose import-time code installs a heavy dependency stack; only the
        functions named here are extracted (via `ast.get_source_segment`) and exec'd in a namespace seeded with
        `constants` and the standard scientific libs — so the kernel's own decision logic runs while none of its
        notebook scaffolding does, and nothing but the constants is transcribed.
        """
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        namespace: dict[str, object] = {
            "np": np,
            "math": math,
            "linear_sum_assignment": linear_sum_assignment,
            **constants,
        }
        wanted = set(function_names)
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        missing = wanted - {node.name for node in defs}
        if missing:
            raise ValueError(f"{path.name} has no top-level function(s) {sorted(missing)}")
        segments = [segment for node in defs if (segment := ast.get_source_segment(source, node)) is not None]
        exec("\n\n".join(segments), namespace)
        return namespace

    @staticmethod
    def nodes_by_id(graph: TrackGraph) -> dict[int, dict[str, float]]:
        """The tracker state as the kernel's node representation: id -> {t, z, y, x} in voxel indices."""
        return {
            int(node_id): {"t": float(row[0]), "z": float(row[1]), "y": float(row[2]), "x": float(row[3])}
            for node_id, row in zip(graph.node_ids.tolist(), graph.coordinates.tolist(), strict=True)
        }

    @staticmethod
    def edge_dicts(graph: TrackGraph) -> list[dict[str, object]]:
        """The graph's edges as the kernel's edge representation, carrying the distance the kernel records."""
        positions_um = SPACING.to_micrometres(graph.positions())
        rows = graph.edge_rows()
        return [
            {
                "source_id": int(source_id),
                "target_id": int(target_id),
                "distance_um": float(np.linalg.norm(positions_um[dst] - positions_um[src])),
                "edge_prob": None,
            }
            for (source_id, target_id), (src, dst) in zip(graph.edges.tolist(), rows.tolist(), strict=True)
        ]

    @staticmethod
    def divsub_helpers() -> dict[str, object]:
        """The divsub kernel's pure-geometry helpers, extracted so nothing but constants is transcribed."""
        return KernelReference.kernel_namespace(
            DIVSUB_REFERENCE,
            ["edge_distance_um", "node_point", "_position_um"],
            {"VOXEL_SCALE_UM": VOXEL_SCALE_UM},
        )

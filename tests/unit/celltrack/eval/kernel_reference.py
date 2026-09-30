from pathlib import Path

import pytest

from celltrack.eval.fidelity_oracle import FidelityOracle
from celltrack.eval.kernel_reference import DIVSUB_REFERENCE, MOTION_REFERENCE, KernelReference

# The donor references are other competitors' notebooks: fetched into the gitignored donor tree, never
# vendored (research/frontier_kernels/README.md), so they are absent on a fresh clone and in CI. Every test
# that runs their actual bytes skips without them — the rest of the bridge is exercised on our own state.
needs_references = pytest.mark.skipif(
    not (DIVSUB_REFERENCE.is_file() and MOTION_REFERENCE.is_file()),
    reason="donor kernel references not fetched (see research/frontier_kernels/README.md)",
)


def test_require(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="kaggle kernels pull"):
        KernelReference.require(tmp_path / "absent_code.py")


@needs_references
def test_kernel_namespace():
    # Extracts a named top-level function out of a reference file without importing the notebook module...
    namespace = KernelReference.kernel_namespace(MOTION_REFERENCE, ["motion_relink_edges"], {})
    assert callable(namespace["motion_relink_edges"])
    # ...and raises, naming the culprit, when a requested function is absent from the reference.
    try:
        KernelReference.kernel_namespace(DIVSUB_REFERENCE, ["not_a_real_kernel_function"], {})
    except ValueError as error:
        assert "not_a_real_kernel_function" in str(error)
    else:
        raise AssertionError("missing function did not raise")


def test_nodes_by_id():
    nodes = KernelReference.nodes_by_id(FidelityOracle.synthetic_state())
    assert nodes[1] == {"t": 1.0, "z": 10.0, "y": 10.0, "x": 100.0}
    assert set(nodes[1]) == {"t", "z", "y", "x"}


def test_edge_dicts():
    edges = KernelReference.edge_dicts(FidelityOracle.synthetic_state())
    first = next(edge for edge in edges if (edge["source_id"], edge["target_id"]) == (0, 1))
    assert first["edge_prob"] is None
    assert first["distance_um"] == 0.0  # nodes 0 and 1 share (z, y, x), differ only in t


@needs_references
def test_divsub_helpers():
    helpers = KernelReference.divsub_helpers()
    assert callable(helpers["edge_distance_um"])
    assert callable(helpers["node_point"])

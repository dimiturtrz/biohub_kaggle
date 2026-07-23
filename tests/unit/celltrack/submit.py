from pathlib import Path

from celltrack.linking import NearestNeighbourLinker
from celltrack.submit import NodeBudget, TestPredictor
from core.paths import DataRoot

SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def test_from_training(dataset_root: Path):
    """The budget is the mean estimated cell count of each acquisition across the training videos."""
    budget = NodeBudget.from_training(DataRoot(dataset_root))
    assert budget.per_prefix == {"44b6": 30, "6bba": 30}


def test_for_video(dataset_root: Path):
    budget = NodeBudget.from_training(DataRoot(dataset_root))
    assert budget.for_video(Path("elsewhere/44b6_0000abcd.zarr")) == 30


def test_classical(dataset_root: Path):
    """The classical predictor wires a LoG detector, an NN linker and a training-derived budget."""
    predictor = TestPredictor.classical(DataRoot(dataset_root), scale_um=4.0, gate_um=15.0)
    assert predictor.detector.scale_um == 4.0
    assert isinstance(predictor.linker, NearestNeighbourLinker)
    assert predictor.linker.max_distance_um == 15.0
    assert set(predictor.budget.per_prefix) == {"44b6", "6bba"}


def test_predict(dataset_root: Path):
    """A test video is detected and linked into a track graph without touching any annotation."""
    root = DataRoot(dataset_root)
    predictor = TestPredictor.classical(root, scale_um=2.0, gate_um=10.0)
    graph = predictor.predict(root.videos("test")[0])
    assert graph.coordinates.shape[1] == 4


def test_submission(dataset_root: Path):
    """Every test video becomes one keyed entry of the submission, in the competition's column order."""
    root = DataRoot(dataset_root)
    predictor = TestPredictor.classical(root, scale_um=2.0, gate_um=10.0)
    submission = predictor.submission(root)
    assert set(submission.graphs) == {"44b6_0000ef01", "6bba_0000ef01"}
    assert submission.to_frame().columns == SUBMISSION_COLUMNS

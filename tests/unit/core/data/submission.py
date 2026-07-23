from pathlib import Path

from core.data.submission import Submission
from core.data.tracks import TrackGraph


def test_to_frame(graph: TrackGraph):
    """Two nodes and one edge become three rows, with the unused columns filled by -1."""
    frame = Submission(graphs={"aaaa_1": graph}).to_frame()
    assert frame.columns == ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    assert frame["row_type"].to_list() == ["node", "node", "edge"]
    assert frame["id"].to_list() == [0, 1, 2]
    assert frame.row(2, named=True)["node_id"] == -1


def test_to_frame_numbers_rows_across_videos(graph: TrackGraph):
    frame = Submission(graphs={"bbbb_2": graph, "aaaa_1": graph}).to_frame()
    assert frame["id"].to_list() == [0, 1, 2, 3, 4, 5]
    assert frame["dataset"].to_list()[:3] == ["aaaa_1"] * 3


def test_write_csv(graph: TrackGraph, tmp_path: Path):
    written = Submission(graphs={"aaaa_1": graph}).write_csv(tmp_path / "submission.csv")
    assert written.read_text(encoding="utf-8").splitlines()[1].startswith("0,aaaa_1,node,7,0,0,0,0,-1,-1")


def test_read_csv(graph: TrackGraph, tmp_path: Path):
    """The CSV round-trips: it is the transport, not a lossy report."""
    source = Submission(graphs={"aaaa_1": graph, "bbbb_2": graph})
    restored = Submission.read_csv(source.write_csv(tmp_path / "submission.csv"))
    assert sorted(restored.graphs) == ["aaaa_1", "bbbb_2"]
    assert restored.graphs["aaaa_1"].node_ids.tolist() == graph.node_ids.tolist()
    assert restored.graphs["aaaa_1"].coordinates.tolist() == graph.coordinates.tolist()
    assert restored.graphs["bbbb_2"].edges.tolist() == graph.edges.tolist()

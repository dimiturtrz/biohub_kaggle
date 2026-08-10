import logging
from pathlib import Path

from core.obs import Obs


def test_setup(tmp_path: Path):
    logfile = tmp_path / "sub" / "train.log"
    logfile.parent.mkdir(parents=True)
    logfile.write_text("stale contents", encoding="utf-8")
    log = Obs.setup(logfile, level=logging.WARNING)
    assert log.name == "celltrack"
    assert log.propagate is False
    assert log.level == logging.WARNING
    assert len(log.handlers) == 2  # stream + file
    assert logfile.read_text(encoding="utf-8") == ""  # truncated at setup


def test_setup_without_file_has_only_a_stream_handler():
    assert len(Obs.setup().handlers) == 1


def test_setup_repeated_does_not_accumulate_handlers(tmp_path: Path):
    Obs.setup(tmp_path / "a.log")
    assert len(Obs.setup(tmp_path / "b.log").handlers) == 2  # cleared, not appended


def test_emit(tmp_path: Path):
    logfile = tmp_path / "train.log"
    handler = Obs._AppendHandler(logfile)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("celltrack", logging.INFO, __file__, 1, "hello %d", (7,), None)
    handler.emit(record)
    assert "hello 7" in logfile.read_text(encoding="utf-8")


def test_progress():
    assert list(Obs.progress(range(3), "count", total=3)) == [0, 1, 2]


def test_timed(tmp_path: Path):
    logfile = tmp_path / "train.log"
    log = Obs.setup(logfile)
    with Obs.timed(log, "load"):
        pass
    contents = logfile.read_text(encoding="utf-8")
    assert "START load" in contents
    assert "DONE  load" in contents


def test_setup_can_append_for_a_resumed_run(tmp_path: Path):
    """`truncate=False` keeps what a killed run wrote, so a resumed run's log continues rather than replaces."""
    logfile = tmp_path / "train.log"
    logfile.write_text("first attempt\n", encoding="utf-8")
    log = Obs.setup(logfile, truncate=False)
    log.warning("second attempt")
    contents = logfile.read_text(encoding="utf-8")
    assert "first attempt" in contents
    assert "second attempt" in contents

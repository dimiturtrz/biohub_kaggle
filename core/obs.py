"""Observability: file+console logging and a timing context manager, for watching runs live.

Subprocess wrappers (`uv run`, CI capture) buffer child stdout, so progress printed to stdout is
invisible until exit. Logging through a per-emit file handler writes straight from the process to
disk — tail the file to watch a training run live.

    from core.obs import Obs
    log = Obs.setup("runs/foo/train.log")
    with Obs.timed(log, "load data"):
        ...
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import override

from tqdm import tqdm

_LOGGER_NAME = "celltrack"


class Obs:
    """Observability entrypoints: logger `setup`, a `progress` bar, and a `timed` block — plus the
    per-emit file handler they write through (nested: it exists only to serve this subject)."""

    class _AppendHandler(logging.Handler):
        """Opens the file, writes the line, closes — every emit. Immune to logging.shutdown() closing a
        long-lived handle (torch/third-party libs call it mid-run, silently killing a normal
        FileHandler). Fine here: logging is per-phase/per-epoch (low frequency), so re-open cost is
        negligible."""

        def __init__(self, path: str | Path) -> None:
            super().__init__()
            self.path = str(path)

        @override
        def emit(self, record: logging.LogRecord) -> None:
            try:
                with Path(self.path).open("a", encoding="utf-8") as handle:
                    handle.write(self.format(record) + "\n")
            except OSError:
                self.handleError(record)

    @staticmethod
    def setup(logfile: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
        """Configure the `celltrack` logger -> console (stdout) + optional file. Returns it.

        Handlers go on the NAMED logger with propagate=False (not the root) — third-party libs call
        `logging.basicConfig(force=True)`, wiping root handlers; keeping ours off the root makes them
        survive that. `celltrack.*` children propagate up to here.
        """
        log = logging.getLogger(_LOGGER_NAME)
        log.setLevel(level)
        log.propagate = False
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        log.addHandler(stream)
        if logfile is not None:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            Path(logfile).write_text("", encoding="utf-8")  # truncate at start
            file_handler = Obs._AppendHandler(logfile)
            file_handler.setFormatter(fmt)
            log.addHandler(file_handler)
        return log

    @staticmethod
    def progress(iterable: Iterable[object], desc: str, total: int | None = None, every: float = 5.0):
        """tqdm progress bar (degrades gracefully in non-tty). `every` = min seconds between bar
        refreshes so file logs stay readable."""
        return tqdm(iterable, desc=desc, total=total, mininterval=every, dynamic_ncols=True)

    @staticmethod
    @contextlib.contextmanager
    def timed(log: logging.Logger, msg: str) -> Iterator[None]:
        """Log START/DONE + elapsed seconds around a block — the basic bottleneck probe."""
        start = time.perf_counter()
        log.info("START %s", msg)
        try:
            yield
        finally:
            log.info("DONE  %s (%.2fs)", msg, time.perf_counter() - start)

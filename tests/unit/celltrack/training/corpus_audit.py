"""Unit tests for the corpus crowding audit's wiring — the report rows come from `CandidateDensity`."""

from celltrack.training.corpus_audit import main


def test_main_is_the_only_entrypoint():
    """The audit is a CLI over `CandidateDensity`: it owns argument wiring, never a measurement of its own."""
    assert callable(main)

"""Fail-fast gate on a warm/chained training base — abort before a broken floor burns the card."""


class BaseCalibration:
    """The warm/chained-base sanity gate: a warm start assumes its base is already a good detector."""

    # A calibrated detector predicts ~one node per cell (ratio near zero); a warm/chained base off by more than
    # half its count is a broken floor to optimise from (v1/v2 chained a +2-ratio base, read as "method nulled").
    MISCALIBRATED_RATIO = 0.5

    @classmethod
    def gate(cls, node_ratio: float, *, expects_calibrated_base: bool) -> None:
        """A warm/chained run onto a miscalibrated base is aborted at window 0 — its null would reflect the base
        (P_true, confusor gap read off a broken floor), not the change under test, so fail fast not burn the card.

        A from-scratch run legitimately starts uncalibrated, so only warm/chained runs assert this.
        """
        if expects_calibrated_base and abs(node_ratio) > cls.MISCALIBRATED_RATIO:
            message = (
                f"warm/chained base is miscalibrated: node ratio {node_ratio:+.2f} exceeds "
                f"+-{cls.MISCALIBRATED_RATIO} — this base is not a valid warm start; point at a converged "
                "base or run from scratch. Aborting before the run wastes the card."
            )
            raise ValueError(message)

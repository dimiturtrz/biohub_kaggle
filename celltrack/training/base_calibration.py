"""Fail-fast gate on a warm/chained training base — abort before a broken floor burns the card."""

_REFERENCE_INPLANE_AREA = 16  # the (1, 4, 4) decode grid the +-0.5 band was calibrated on (dy*dx = 4*4)


class BaseCalibration:
    """The warm/chained-base sanity gate: a warm start assumes its base is already a good detector."""

    # A calibrated (1,4,4) detector predicts ~one node per cell (ratio near zero); a warm/chained base off by more
    # than half its count is a broken floor to optimise from (v1/v2 chained a +2-ratio base, read as "method
    # nulled"). The band is asymmetric on a finer grid — see gate().
    MISCALIBRATED_RATIO = 0.5

    @classmethod
    def upper_band(cls, inplane_area: int) -> float:
        """The most a VALID warm base may over-detect, scaled to its decode grid.

        A finer grid over-detects BY DESIGN: more in-plane pixels -> more local maxima before NMS, and the linker
        drops the surplus (that is how finer decode separates cells merged at (1,4,4)). Candidate count scales
        ~linearly with in-plane pixel count, so the allowed positive ratio scales with the area ratio vs the
        (1,4,4) reference the +-0.5 band was set on: (1,2,2) has 4x the pixels -> band +2.0, and a truly broken
        +3 base is still caught. Under-detection is never grid-explained, so the lower band stays fixed (gate()).
        """
        return cls.MISCALIBRATED_RATIO * (_REFERENCE_INPLANE_AREA / inplane_area)

    @classmethod
    def gate(
        cls, node_ratio: float, *, expects_calibrated_base: bool, inplane_area: int = _REFERENCE_INPLANE_AREA
    ) -> None:
        """A warm/chained run onto a miscalibrated base is aborted at window 0 — its null would reflect the base
        (P_true, confusor gap read off a broken floor), not the change under test, so fail fast not burn the card.

        The band is asymmetric: over-detection (positive ratio) is grid-explained and allowed up to `upper_band`
        for the run's decode grid; under-detection (negative ratio) is always a broken floor, held at the fixed
        -MISCALIBRATED_RATIO. A from-scratch run legitimately starts uncalibrated, so only warm/chained runs
        assert this.
        """
        if not expects_calibrated_base:
            return
        upper = cls.upper_band(inplane_area)
        if node_ratio > upper or node_ratio < -cls.MISCALIBRATED_RATIO:
            message = (
                f"warm/chained base is miscalibrated: node ratio {node_ratio:+.2f} outside "
                f"[{-cls.MISCALIBRATED_RATIO:+.2f}, {upper:+.2f}] for this decode grid — this base is not a valid "
                "warm start; point at a converged base or run from scratch. Aborting before the run wastes the card."
            )
            raise ValueError(message)

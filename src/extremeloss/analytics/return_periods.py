from __future__ import annotations

from ..estimation.metrics import exceedance_probability
from ..results import GPDFit


def return_period(probability: float) -> float:
    """The return period ``1 / p`` of an event with exceedance probability ``p``.

    The reciprocal convention: an event with per-observation exceedance
    probability ``p`` recurs once per ``1/p`` observations on average.
    Raises ``ValueError`` outside ``0 < p < 1``.
    """
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be strictly between 0 and 1")
    return float(1.0 / probability)


def exceedance_frequency(losses, threshold: float) -> float:
    return exceedance_probability(losses, threshold)


def return_level(period: float, fit: GPDFit) -> float:
    """The loss exceeded on average once per ``period`` observations, under a POT fit.

    Thin wrapper over :meth:`GPDFit.return_level` -- the fit's
    unconditional quantile at ``1 - 1/period`` -- with the domain check
    that ``period`` exceeds one. Point estimates agree exactly with
    :func:`gpd_return_level`, which adds period units
    (``observations_per_period``) and delta-method confidence intervals.
    """
    if period <= 1.0:
        raise ValueError("period must exceed 1.0")
    return fit.return_level(period)

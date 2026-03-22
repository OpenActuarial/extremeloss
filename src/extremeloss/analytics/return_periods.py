from __future__ import annotations

from ..estimation.metrics import exceedance_probability
from ..results import GPDFit


def return_period(probability: float) -> float:
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be strictly between 0 and 1")
    return float(1.0 / probability)


def exceedance_frequency(losses, threshold: float) -> float:
    return exceedance_probability(losses, threshold)


def return_level(period: float, fit: GPDFit) -> float:
    if period <= 1.0:
        raise ValueError("period must exceed 1.0")
    return fit.return_level(period)
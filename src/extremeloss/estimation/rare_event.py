from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..results import TailEstimateResult
from ..utils.validation import coerce_losses, validate_alpha, validate_q, validate_threshold
from .metrics import empirical_tvar, empirical_var, exceedance_probability


def _normal_ci(estimate: float, stderr: float, alpha: float) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = estimate - z * stderr
    upper = estimate + z * stderr
    return float(lower), float(upper)


def estimate_tail_probability(
    data,
    threshold: float,
    *,
    size: int | None = None,
    alpha: float = 0.05,
) -> TailEstimateResult:
    """Estimate P(X > threshold) from simulated or observed losses."""
    validate_threshold(threshold)
    validate_alpha(alpha)
    losses = coerce_losses(data, size=size)
    indicators = (losses > threshold).astype(float)
    estimate = float(np.mean(indicators))
    stderr = float(np.std(indicators, ddof=0) / math.sqrt(losses.size))
    ci = _normal_ci(estimate, stderr, alpha)
    return TailEstimateResult(
        estimate=estimate,
        method="empirical",
        stderr=stderr,
        ci=ci,
        n=int(losses.size),
        threshold=float(threshold),
        diagnostics={"n_exceedances": int(np.sum(indicators))},
    )


def estimate_var(
    data,
    q: float,
    *,
    size: int | None = None,
    alpha: float = 0.05,
) -> TailEstimateResult:
    validate_q(q)
    validate_alpha(alpha)
    losses = coerce_losses(data, size=size)
    estimate = empirical_var(losses, q)
    return TailEstimateResult(
        estimate=estimate,
        method="empirical",
        n=int(losses.size),
        quantile=float(q),
    )


def estimate_tvar(
    data,
    q: float,
    *,
    size: int | None = None,
    alpha: float = 0.05,
) -> TailEstimateResult:
    validate_q(q)
    validate_alpha(alpha)
    losses = coerce_losses(data, size=size)
    threshold = empirical_var(losses, q)
    tail = losses[losses >= threshold]
    estimate = empirical_tvar(losses, q)
    stderr = None
    ci = None
    if tail.size > 1:
        stderr = float(np.std(tail, ddof=1) / math.sqrt(tail.size))
        ci = _normal_ci(estimate, stderr, alpha)
    return TailEstimateResult(
        estimate=estimate,
        method="empirical",
        stderr=stderr,
        ci=ci,
        n=int(losses.size),
        threshold=float(threshold),
        quantile=float(q),
        diagnostics={"tail_sample_size": int(tail.size)},
    )


def estimate_var_tvar(
    data,
    q: float,
    *,
    size: int | None = None,
    alpha: float = 0.05,
) -> dict[str, TailEstimateResult]:
    losses = coerce_losses(data, size=size)
    return {
        "var": estimate_var(losses, q, alpha=alpha),
        "tvar": estimate_tvar(losses, q, alpha=alpha),
        "tail_probability": TailEstimateResult(
            estimate=exceedance_probability(losses, empirical_var(losses, q)),
            method="empirical",
            n=int(losses.size),
            quantile=float(q),
        ),
    }
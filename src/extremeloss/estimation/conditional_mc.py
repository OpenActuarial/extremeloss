from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..results import TailEstimateResult
from ..utils.validation import validate_alpha, validate_probabilities, validate_threshold, validate_q


def _normal_ci(estimate: float, stderr: float, alpha: float) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(estimate - z * stderr), float(estimate + z * stderr)


def estimate_tail_probability_cmc(
    conditional_probabilities,
    *,
    threshold: float | None = None,
    alpha: float = 0.05,
) -> TailEstimateResult:
    """Estimate an exceedance probability from conditional exceedance probabilities.

    Parameters
    ----------
    conditional_probabilities:
        Samples of P(X > u | Y) from a conditioning variable Y.
    threshold:
        Optional tail threshold associated with the conditional probabilities.
    """
    validate_alpha(alpha)
    if threshold is not None:
        validate_threshold(threshold)
    probs = validate_probabilities(conditional_probabilities, name="conditional_probabilities")
    estimate = float(np.mean(probs))
    stderr = float(np.std(probs, ddof=0) / math.sqrt(probs.size))
    return TailEstimateResult(
        estimate=estimate,
        method="conditional_monte_carlo",
        stderr=stderr,
        ci=_normal_ci(estimate, stderr, alpha),
        n=int(probs.size),
        threshold=float(threshold) if threshold is not None else None,
        diagnostics={
            "min_conditional_probability": float(np.min(probs)),
            "max_conditional_probability": float(np.max(probs)),
        },
    )


def estimate_tvar_cmc(
    conditional_tail_expectations,
    *,
    q: float,
    threshold: float | None = None,
    alpha: float = 0.05,
) -> TailEstimateResult:
    """Estimate TVaR from conditional expectations of tail losses.

    `conditional_tail_expectations` should contain draws of E[X | X >= VaR_q, Y]
    or another conditionally unbiased TVaR contribution.
    """
    validate_q(q)
    validate_alpha(alpha)
    if threshold is not None:
        validate_threshold(threshold)
    vals = np.asarray(conditional_tail_expectations, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("conditional_tail_expectations must be a non-empty 1D array")
    estimate = float(np.mean(vals))
    stderr = float(np.std(vals, ddof=0) / math.sqrt(vals.size))
    return TailEstimateResult(
        estimate=estimate,
        method="conditional_monte_carlo",
        stderr=stderr,
        ci=_normal_ci(estimate, stderr, alpha),
        n=int(vals.size),
        threshold=float(threshold) if threshold is not None else None,
        quantile=float(q),
        diagnostics={
            "min_conditional_expectation": float(np.min(vals)),
            "max_conditional_expectation": float(np.max(vals)),
        },
    )
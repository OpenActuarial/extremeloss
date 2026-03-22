from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..results import TailEstimateResult
from ..utils.validation import (
    as_1d_float_array,
    validate_alpha,
    validate_q,
    validate_threshold,
    validate_weights,
)


def normalized_weights(weights) -> np.ndarray:
    w = validate_weights(weights)
    return w / np.sum(w)


def effective_sample_size(weights) -> float:
    w = normalized_weights(weights)
    return float(1.0 / np.sum(w ** 2))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(idx, x.size - 1)
    return float(x[idx])


def _normal_ci(estimate: float, stderr: float, alpha: float) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(estimate - z * stderr), float(estimate + z * stderr)


def estimate_tail_probability_is(
    losses,
    weights,
    threshold: float,
    *,
    alpha: float = 0.05,
) -> TailEstimateResult:
    validate_threshold(threshold)
    validate_alpha(alpha)
    x = as_1d_float_array(losses, name="losses")
    w = normalized_weights(weights)
    if x.size != w.size:
        raise ValueError("losses and weights must have the same length")
    indicators = (x > threshold).astype(float)
    estimate = float(np.sum(w * indicators))
    ess = effective_sample_size(w)
    variance = float(np.sum(w * (indicators - estimate) ** 2))
    stderr = float(math.sqrt(variance / ess)) if ess > 0 else 0.0
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        stderr=stderr,
        ci=_normal_ci(estimate, stderr, alpha),
        n=int(x.size),
        effective_n=ess,
        threshold=float(threshold),
        diagnostics={"n_exceedances": int(np.sum(indicators))},
    )


def estimate_var_is(losses, weights, q: float) -> TailEstimateResult:
    validate_q(q)
    x = as_1d_float_array(losses, name="losses")
    w = normalized_weights(weights)
    if x.size != w.size:
        raise ValueError("losses and weights must have the same length")
    estimate = _weighted_quantile(x, w, q)
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        n=int(x.size),
        effective_n=effective_sample_size(w),
        quantile=float(q),
    )


def estimate_tvar_is(losses, weights, q: float) -> TailEstimateResult:
    validate_q(q)
    x = as_1d_float_array(losses, name="losses")
    w = normalized_weights(weights)
    if x.size != w.size:
        raise ValueError("losses and weights must have the same length")
    threshold = _weighted_quantile(x, w, q)
    mask = x >= threshold
    tail_weights = w[mask]
    tail_losses = x[mask]
    if tail_losses.size == 0:
        estimate = threshold
    else:
        estimate = float(np.sum(tail_weights * tail_losses) / np.sum(tail_weights))
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        n=int(x.size),
        effective_n=effective_sample_size(w),
        threshold=float(threshold),
        quantile=float(q),
        diagnostics={"tail_weight": float(np.sum(tail_weights))},
    )
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
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


def stabilize_weights(weights, *, clip_quantile: float | None = None, renormalize: bool = True) -> np.ndarray:
    w = validate_weights(weights).copy()
    if clip_quantile is not None:
        if not (0.0 < clip_quantile <= 1.0):
            raise ValueError("clip_quantile must lie in (0, 1]")
        upper = float(np.quantile(w, clip_quantile))
        w = np.minimum(w, upper)
    if renormalize:
        w = w / np.sum(w)
    return w


def log_importance_weights(log_target_density, log_proposal_density, *, normalize: bool = True) -> np.ndarray:
    log_t = as_1d_float_array(log_target_density, name="log_target_density")
    log_p = as_1d_float_array(log_proposal_density, name="log_proposal_density")
    if log_t.size != log_p.size:
        raise ValueError("log_target_density and log_proposal_density must have the same length")
    log_w = log_t - log_p
    if normalize:
        log_w = log_w - logsumexp(log_w)
        return np.exp(log_w)
    return np.exp(log_w)


def effective_sample_size(weights) -> float:
    w = normalized_weights(weights)
    return float(1.0 / np.sum(w ** 2))


def importance_sampling_diagnostics(weights) -> dict[str, float]:
    w = normalized_weights(weights)
    ess = float(1.0 / np.sum(w ** 2))
    cv = float(np.std(w, ddof=0) / np.mean(w))
    entropy = float(-np.sum(w * np.log(w + 1e-300)))
    return {
        "effective_n": ess,
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w)),
        "coefficient_of_variation": cv,
        "entropy": entropy,
    }


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


def _validate_losses_weights(losses, weights) -> tuple[np.ndarray, np.ndarray]:
    x = as_1d_float_array(losses, name="losses")
    w = normalized_weights(weights)
    if x.size != w.size:
        raise ValueError("losses and weights must have the same length")
    return x, w


def estimate_mean_is(values, weights, *, alpha: float = 0.05) -> TailEstimateResult:
    validate_alpha(alpha)
    x, w = _validate_losses_weights(values, weights)
    estimate = float(np.sum(w * x))
    ess = effective_sample_size(w)
    variance = float(np.sum(w * (x - estimate) ** 2))
    stderr = float(math.sqrt(variance / ess)) if ess > 0 else 0.0
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        stderr=stderr,
        ci=_normal_ci(estimate, stderr, alpha),
        n=int(x.size),
        effective_n=ess,
        diagnostics=importance_sampling_diagnostics(w),
    )


def estimate_tail_probability_is(
    losses,
    weights,
    threshold: float,
    *,
    alpha: float = 0.05,
) -> TailEstimateResult:
    validate_threshold(threshold)
    validate_alpha(alpha)
    x, w = _validate_losses_weights(losses, weights)
    indicators = (x > threshold).astype(float)
    estimate = float(np.sum(w * indicators))
    ess = effective_sample_size(w)
    variance = float(np.sum(w * (indicators - estimate) ** 2))
    stderr = float(math.sqrt(variance / ess)) if ess > 0 else 0.0
    diagnostics = importance_sampling_diagnostics(w)
    diagnostics["n_exceedances"] = int(np.sum(indicators))
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        stderr=stderr,
        ci=_normal_ci(estimate, stderr, alpha),
        n=int(x.size),
        effective_n=ess,
        threshold=float(threshold),
        diagnostics=diagnostics,
    )


def estimate_exceedance_curve_is(losses, weights, thresholds) -> dict[str, np.ndarray]:
    x, w = _validate_losses_weights(losses, weights)
    grid = as_1d_float_array(thresholds, name="thresholds")
    probs = np.array([np.sum(w * (x > u)) for u in grid], dtype=float)
    return {"thresholds": grid, "probabilities": probs}


def estimate_var_is(losses, weights, q: float) -> TailEstimateResult:
    validate_q(q)
    x, w = _validate_losses_weights(losses, weights)
    estimate = _weighted_quantile(x, w, q)
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        n=int(x.size),
        effective_n=effective_sample_size(w),
        quantile=float(q),
        diagnostics=importance_sampling_diagnostics(w),
    )


def estimate_tvar_is(losses, weights, q: float) -> TailEstimateResult:
    validate_q(q)
    x, w = _validate_losses_weights(losses, weights)
    threshold = _weighted_quantile(x, w, q)
    mask = x >= threshold
    tail_weights = w[mask]
    tail_losses = x[mask]
    if tail_losses.size == 0:
        estimate = threshold
    else:
        estimate = float(np.sum(tail_weights * tail_losses) / np.sum(tail_weights))
    diagnostics = importance_sampling_diagnostics(w)
    diagnostics["tail_weight"] = float(np.sum(tail_weights))
    return TailEstimateResult(
        estimate=estimate,
        method="importance_sampling",
        n=int(x.size),
        effective_n=effective_sample_size(w),
        threshold=float(threshold),
        quantile=float(q),
        diagnostics=diagnostics,
    )


def estimate_var_tvar_is(losses, weights, q: float) -> dict[str, TailEstimateResult]:
    return {
        "var": estimate_var_is(losses, weights, q=q),
        "tvar": estimate_tvar_is(losses, weights, q=q),
    }
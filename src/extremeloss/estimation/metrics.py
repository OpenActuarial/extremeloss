from __future__ import annotations

import numpy as np

from ..utils.validation import as_1d_float_array, validate_q


def _var_rank(n: int, q: np.ndarray) -> np.ndarray:
    """Rank k of the VaR order statistic: k = ceil(n*q), guarded against
    floating-point error when n*q is an exact integer."""
    nq = n * q
    k = np.where(np.abs(nq - np.round(nq)) < 1e-8, np.round(nq), np.ceil(nq))
    return np.clip(k.astype(np.intp), 1, n)


def empirical_var(losses, q):
    """Empirical VaR: the order statistic ``x_(ceil(n*q))``.

    Implements ``VaR_q = inf{x : F(x) >= q}`` on the empirical distribution
    (identical to ``np.quantile(..., method="inverted_cdf")``). This is the
    ecosystem-standard estimator shared with ``risksim`` and ``lossmodels``;
    it serves as the crude-Monte-Carlo baseline that the variance-reduced
    estimators in this subpackage are compared against.

    ``q`` may be scalar (returns ``float``) or array-like (returns array).
    """
    validate_q(q)
    arr = np.sort(as_1d_float_array(losses, name="losses"))
    q_arr = np.asarray(q, dtype=float)
    k = _var_rank(arr.size, q_arr)
    out = arr[k - 1]
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def empirical_tvar(losses, q):
    """Empirical TVaR via the average-quantile (Acerbi-Tasche) plug-in.

    Implements ``TVaR_q = (1/(1-q)) * integral_q^1 VaR_u du`` exactly on the
    empirical distribution: with sorted losses and ``k = ceil(n*q)``,

        TVaR_q = [ sum_{i>k} x_(i) + x_(k) * (k - n*q) ] / (n * (1 - q)).

    Correct in the presence of ties/atoms and always ``>= empirical_var``.
    Ecosystem-standard estimator shared with ``risksim`` and ``lossmodels``.

    ``q`` may be scalar (returns ``float``) or array-like (returns array).
    """
    validate_q(q)
    arr = np.sort(as_1d_float_array(losses, name="losses"))
    n = arr.size
    q_arr = np.asarray(q, dtype=float)
    k = _var_rank(n, q_arr)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    tail_sum = csum[n] - csum[k]
    var_vals = arr[k - 1]
    nq = n * q_arr
    weight = np.where(np.abs(nq - np.round(nq)) < 1e-8, 0.0, k - nq)
    out = (tail_sum + var_vals * weight) / (n * (1.0 - q_arr))
    # TVaR >= VaR holds as a theorem in exact arithmetic; enforce it so
    # floating-point noise (e.g. a constant tail at a layer limit) can never
    # produce tvar infinitesimally below var.
    out = np.maximum(out, var_vals)
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def exceedance_probability(losses, threshold: float) -> float:
    """Empirical exceedance probability: the fraction of losses strictly above ``threshold``.

    The plug-in estimator ``mean(losses > threshold)`` for
    ``P(X > threshold)``. Strict inequality, matching the exceedance
    convention used across the package.
    """
    arr = as_1d_float_array(losses, name="losses")
    return float(np.mean(arr > threshold))


def exceedance_curve(losses, thresholds) -> dict[str, np.ndarray]:
    arr = as_1d_float_array(losses, name="losses")
    grid = as_1d_float_array(thresholds, name="thresholds")
    probs = np.array([np.mean(arr > u) for u in grid], dtype=float)
    return {"thresholds": grid, "probabilities": probs}


def survival_function(losses, grid) -> dict[str, np.ndarray]:
    return exceedance_curve(losses, grid)

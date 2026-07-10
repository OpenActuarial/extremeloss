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
    """Clip extreme importance weights at an upper quantile, then renormalize.

    With ``clip_quantile=c``, weights are capped at their own
    ``c``-quantile -- the standard variance-for-bias trade when a few
    weights dominate the sample; ``renormalize=True`` (default) rescales
    to sum to one afterwards. With ``clip_quantile=None`` this only
    (optionally) normalizes.

    Parameters
    ----------
    weights : array-like
        Nonnegative importance weights (need not be normalized).
    clip_quantile : float, optional
        Cap level in ``(0, 1]``; ``None`` disables clipping.
    renormalize : bool, optional
        Rescale to sum to one after clipping (default True).

    Returns
    -------
    numpy.ndarray
        The stabilized weights.

    See Also
    --------
    importance_sampling_diagnostics : Check whether stabilization is needed.
    """
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
    r"""Importance weights from log-densities, computed stably in log space.

    :math:`w_i \propto \exp(\log f(x_i) - \log g(x_i))` for target
    :math:`f` and proposal :math:`g`. With ``normalize=True`` (default)
    the weights are self-normalized via ``logsumexp`` -- the numerically
    safe route when the densities span many orders of magnitude, exactly
    the regime importance sampling is used for. With ``normalize=False``
    the raw (exponentiated) ratios are returned.

    Parameters
    ----------
    log_target_density, log_proposal_density : array-like
        Log-densities of the two distributions at the sampled points;
        equal lengths.
    normalize : bool, optional
        Self-normalize the weights to sum to one (default True).

    Returns
    -------
    numpy.ndarray
        The importance weights.
    """
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
    r"""Kish effective sample size of a set of importance weights.

    :math:`\mathrm{ESS} = 1 / \sum_i \bar w_i^2` for normalized weights
    :math:`\bar w_i`: equal to ``n`` when all weights are equal,
    approaching one as a single weight dominates. The headline diagnostic
    for weight degeneracy -- the importance-sampling standard errors in
    this module scale by :math:`\sqrt{\mathrm{ESS}}`, not
    :math:`\sqrt{n}`.
    """
    w = normalized_weights(weights)
    return float(1.0 / np.sum(w ** 2))


def importance_sampling_diagnostics(weights) -> dict[str, float]:
    """Weight-degeneracy diagnostics for an importance sample.

    Normalizes the weights and returns ``effective_n`` (Kish ESS),
    ``max_weight`` / ``min_weight``, ``coefficient_of_variation``, and
    ``entropy`` (natural log). A max weight near one, or an ESS far below
    ``n``, says the proposal is missing the target's mass and the
    estimates carry more uncertainty than their nominal ``n`` suggests.
    These diagnostics ride along on every ``estimate_*_is`` result.
    """
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


def _weighted_var_index(values: np.ndarray, weights: np.ndarray, q: float):
    """Sorted values/weights and the index of the weighted lower quantile.

    The index is the first ``k`` with cumulative weight ``W_k >= q`` -- the
    weighted analogue of the ecosystem VaR convention ``inf{x : F(x) >= q}``.
    """
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf[-1] = 1.0  # guard cumulative rounding of normalized weights
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(idx, x.size - 1)
    return x, w, cdf, idx


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    x, _, _, idx = _weighted_var_index(values, weights, q)
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
    r"""Self-normalized importance-sampling estimate of a mean, with a Wald CI.

    :math:`\hat\mu = \sum_i \bar w_i x_i` for normalized weights
    :math:`\bar w_i`. The standard error uses the weighted second moment
    about :math:`\hat\mu` scaled by the *effective* sample size (Kish
    ESS), not ``n`` -- degenerate weights widen the interval as they
    should.

    Parameters
    ----------
    values, weights : array-like
        Sampled values and their importance weights (normalization not
        required); equal lengths.
    alpha : float, optional
        Two-sided Wald level (default ``0.05``, a 95% interval).

    Returns
    -------
    TailEstimateResult
        ``estimate``, ``stderr``, ``ci``, ``n``, ``effective_n``, and the
        weight diagnostics.
    """
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
    r"""Self-normalized importance-sampling estimate of :math:`P(X > u)`, with a Wald CI.

    The weighted exceedance-indicator mean
    :math:`\sum_i \bar w_i \, 1\{x_i > u\}`, with the standard error
    scaled by the effective sample size exactly as in
    :func:`estimate_mean_is`. This is the payoff case for importance
    sampling: a proposal tilted into the tail makes rare exceedances
    common, so far fewer simulations are needed for a given precision.
    The raw exceedance count rides in ``diagnostics``.

    Parameters
    ----------
    losses, weights : array-like
        Sampled losses and their importance weights; equal lengths.
    threshold : float
        The level :math:`u` whose exceedance probability is estimated.
    alpha : float, optional
        Two-sided Wald level (default ``0.05``).

    Returns
    -------
    TailEstimateResult
        ``estimate``, ``stderr``, ``ci``, ``n``, ``effective_n``,
        ``threshold``, and the weight diagnostics.
    """
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
    """Weighted exceedance curve: self-normalized ``P(X > u)`` over a threshold grid.

    The importance-sampling analogue of the empirical exceedance curve --
    each probability is the normalized-weight mass strictly above the
    threshold. Returns ``{"thresholds", "probabilities"}`` as arrays.
    Point estimates only; for a standard error at a single threshold use
    :func:`estimate_tail_probability_is`.
    """
    x, w = _validate_losses_weights(losses, weights)
    grid = as_1d_float_array(thresholds, name="thresholds")
    probs = np.array([np.sum(w * (x > u)) for u in grid], dtype=float)
    return {"thresholds": grid, "probabilities": probs}


def estimate_var_is(losses, weights, q: float) -> TailEstimateResult:
    r"""Weighted value-at-risk: the ``q``-quantile under the importance weights.

    The weighted analogue of the ecosystem VaR convention
    :math:`\inf\{x : F(x) \ge q\}`: with values sorted and weights
    normalized, the first value whose cumulative weight reaches ``q``.
    Reduces exactly to the empirical VaR under equal weights. No standard
    error is attached (quantile standard errors under importance sampling
    require a density estimate); the effective sample size and weight
    diagnostics ride along.
    """
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
    """Weighted TVaR under the ecosystem average-quantile convention.

    Implements the weighted Acerbi-Tasche plug-in for
    ``TVaR_q = (1/(1-q)) * integral_q^1 VaR_u du``: with values sorted, weights
    normalized, cumulative weights ``W_i`` and ``k`` the weighted-VaR index,

        TVaR_q = [ sum_{i>k} w_i x_i + x_k (W_k - q) ] / (1 - q).

    The atom at VaR contributes only the weight mass above level ``q``, which
    keeps the estimator coherent with ties and makes it reduce *exactly* to
    ``empirical_tvar`` when all weights are equal.
    """
    validate_q(q)
    x, w = _validate_losses_weights(losses, weights)
    xs, ws, cdf, k = _weighted_var_index(x, w, q)
    threshold = float(xs[k])
    tail = float(np.sum(ws[k + 1 :] * xs[k + 1 :]))
    atom = threshold * max(float(cdf[k]) - q, 0.0)
    estimate = (tail + atom) / (1.0 - q)
    # TVaR >= VaR holds as a theorem; enforce it against floating-point noise,
    # matching the empirical estimators across the ecosystem.
    estimate = max(estimate, threshold)
    diagnostics = importance_sampling_diagnostics(w)
    diagnostics["tail_weight"] = float(np.sum(ws[k:]))
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
    """Weighted VaR and TVaR at level ``q`` in one call.

    Returns ``{"var": estimate_var_is(...), "tvar":
    estimate_tvar_is(...)}`` -- the pair share the sorting and weighting
    conventions, so the TVaR always sits at or above the VaR.
    """
    return {
        "var": estimate_var_is(losses, weights, q=q),
        "tvar": estimate_tvar_is(losses, weights, q=q),
    }

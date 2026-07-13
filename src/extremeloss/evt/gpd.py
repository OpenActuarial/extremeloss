from __future__ import annotations

import math

import numpy as np
from scipy.stats import genpareto

from ..results import GPDFit
from ..utils.validation import (
    as_1d_float_array,
    validate_gpd_params,
    validate_q,
    validate_threshold,
)


def fit_gpd(excesses, threshold: float = 0.0, method: str = "mle") -> GPDFit:
    """Fit a generalized Pareto distribution to excess losses."""
    if method != "mle":
        raise ValueError("only method='mle' is currently supported")
    validate_threshold(threshold)
    x = as_1d_float_array(excesses, name="excesses")
    if np.any(x <= 0.0):
        raise ValueError("excesses must be strictly positive")
    xi_hat, loc_hat, beta_hat = genpareto.fit(x, floc=0.0)
    if loc_hat != 0.0:
        raise RuntimeError("GPD fit returned nonzero location despite floc=0")
    return GPDFit(
        threshold=float(threshold),
        xi=float(xi_hat),
        beta=float(beta_hat),
        exceedance_fraction=1.0,
        n_exceedances=int(x.size),
        fit_method=method,
        covariance=_gpd_covariance(x, float(xi_hat), float(beta_hat)),
    )


def _gpd_covariance(excesses, xi: float, beta: float):
    """Observed-information covariance of (xi, beta) at the MLE.

    The numerical Hessian of the GPD log-likelihood, inverted. Returns
    ``None`` when the information is not positive definite -- the MLE is
    irregular for xi <= -1/2 and near-boundary fits, and a covariance that
    means nothing is worse than no covariance.
    """
    x = np.asarray(excesses, dtype=float)

    def loglik(theta):
        c, b = float(theta[0]), float(theta[1])
        if b <= 0.0:
            return -np.inf
        return float(np.sum(genpareto.logpdf(x, c=c, scale=b)))

    theta = np.array([xi, beta], dtype=float)
    h = 1e-5 * np.maximum(np.abs(theta), 1e-3)
    f0 = loglik(theta)
    hess = np.empty((2, 2))
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = h[i]
        hess[i, i] = (loglik(theta + ei) - 2 * f0 + loglik(theta - ei)) / h[i] ** 2
    ej = np.array([0.0, h[1]])
    ei = np.array([h[0], 0.0])
    hess[0, 1] = hess[1, 0] = (
        loglik(theta + ei + ej)
        - loglik(theta + ei - ej)
        - loglik(theta - ei + ej)
        + loglik(theta - ei - ej)
    ) / (4 * h[0] * h[1])
    if not np.all(np.isfinite(hess)):
        return None
    info = -hess
    try:
        np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.inv(info)


def gpd_tail_probability(
    x: float,
    threshold: float,
    xi: float,
    beta: float,
    exceedance_fraction: float,
) -> float:
    r"""Unconditional GPD tail probability :math:`P(X > x)` above a POT threshold.

    For :math:`x > u`, returns
    :math:`\zeta_u \, (1 + \xi (x - u)/\beta)^{-1/\xi}` (exponential form
    as :math:`\xi \to 0`), where :math:`\zeta_u` is the exceedance rate;
    for :math:`x \le u` it returns :math:`\zeta_u` itself -- the GPD says
    nothing below its threshold. Zero beyond the finite upper endpoint when
    :math:`\xi < 0`.

    Parameters
    ----------
    x : float
        The loss level.
    threshold, xi, beta : float
        POT threshold :math:`u` and the fitted GPD shape and scale.
    exceedance_fraction : float
        The exceedance rate :math:`\zeta_u = P(X > u)`, e.g.
        ``GPDFit.exceedance_fraction``.

    Returns
    -------
    float
        Ground-up (unconditional) :math:`P(X > x)`.
    """
    validate_gpd_params(threshold, xi, beta, exceedance_fraction, allow_zero_exceedance=True)
    if x <= threshold:
        return float(exceedance_fraction)
    y = (x - threshold) / beta
    if abs(xi) < 1e-10:
        surv = math.exp(-y)
    else:
        term = 1.0 + xi * y
        if term <= 0.0:
            return 0.0
        surv = term ** (-1.0 / xi)
    return float(exceedance_fraction * surv)


def gpd_var(
    p: float,
    threshold: float,
    xi: float,
    beta: float,
    exceedance_fraction: float,
) -> float:
    r"""Unconditional GPD value-at-risk: the ``p``-quantile of the ground-up loss.

    Inverts the POT tail: with exceedance rate :math:`\zeta_u`,

    .. math::
        \mathrm{VaR}_p = u + \frac{\beta}{\xi}
        \left[\left(\frac{1 - p}{\zeta_u}\right)^{-\xi} - 1\right]

    (:math:`u + \beta \log(\zeta_u / (1 - p))` as :math:`\xi \to 0`).
    Valid only when the quantile lands in the fitted tail, i.e.
    :math:`1 - p < \zeta_u`; otherwise a ``ValueError`` -- below the
    threshold the GPD has nothing to say.

    Parameters
    ----------
    p : float
        Quantile level in ``(0, 1)``, e.g. ``0.995``.
    threshold, xi, beta : float
        POT threshold :math:`u` and the fitted GPD shape and scale.
    exceedance_fraction : float
        The exceedance rate :math:`\zeta_u`, e.g.
        ``GPDFit.exceedance_fraction``.

    Returns
    -------
    float
        The ground-up ``p``-quantile.

    See Also
    --------
    gpd_tvar : The matching expected shortfall.
    gpd_return_level : Return levels with delta-method intervals.
    """
    validate_q(p)
    validate_gpd_params(threshold, xi, beta, exceedance_fraction)
    tail_prob = 1.0 - p
    if tail_prob >= exceedance_fraction:
        raise ValueError(
            "p is not far enough into the tail for the specified threshold and exceedance_fraction"
        )
    ratio = tail_prob / exceedance_fraction
    if abs(xi) < 1e-10:
        return float(threshold + beta * math.log(1.0 / ratio))
    return float(threshold + (beta / xi) * (ratio ** (-xi) - 1.0))


def gpd_tvar(
    p: float,
    threshold: float,
    xi: float,
    beta: float,
    exceedance_fraction: float,
) -> float:
    r"""Unconditional GPD tail value-at-risk (expected shortfall) at level ``p``.

    Closed form on top of :func:`gpd_var`:

    .. math::
        \mathrm{TVaR}_p = \frac{\mathrm{VaR}_p + \beta - \xi u}{1 - \xi},
        \qquad \xi < 1.

    Infinite for :math:`\xi \ge 1` (the tail has no mean) -- a
    ``ValueError`` rather than a silent ``inf``. Arguments as for
    :func:`gpd_var`.
    """
    if xi >= 1.0:
        raise ValueError("TVaR is infinite for xi >= 1")
    var_p = gpd_var(p, threshold, xi, beta, exceedance_fraction)
    return float((var_p + beta - xi * threshold) / (1.0 - xi))


def gpd_return_level(
    fit,
    return_periods,
    observations_per_period: float = 1.0,
    confidence_level: float = 0.95,
):
    r"""Return levels with confidence intervals from a POT/GPD fit.

    The ``T``-period return level is the loss exceeded once per ``T``
    periods on average: with exceedance rate :math:`\zeta_u` and ``m``
    observations per period, it solves
    :math:`P(X > r) = 1/(T\,m)`:

    .. math::
        r_T = u + \frac{\beta}{\xi}\left[(T\,m\,\zeta_u)^{\xi} - 1\right]

    (:math:`u + \beta\log(T m \zeta_u)` as :math:`\xi \to 0`).
    Confidence intervals are by the delta method over
    :math:`(\zeta_u, \xi, \beta)`, including the binomial variance of the
    exceedance rate (Coles, 2001, §4.3.3); they require the fit to carry a
    parameter ``covariance`` (populated by :func:`fit_gpd` / ``fit_pot``
    when the information matrix is positive definite).

    Parameters
    ----------
    fit : GPDFit
        A fit whose ``exceedance_fraction`` reflects the full dataset
        (i.e. from ``fit_pot``, not raw ``fit_gpd`` on excesses alone).
    return_periods : float or array-like
        Periods ``T`` in the same period unit as ``observations_per_period``.
    observations_per_period : float
        Observations per period (e.g. claims per year), so ``T *
        observations_per_period * exceedance_fraction`` is the expected
        number of threshold exceedances in ``T`` periods -- it must exceed
        1 for the return level to sit above the threshold.
    confidence_level : float
        Wald interval level.

    Returns
    -------
    dict of str -> numpy.ndarray
        ``return_period``, ``return_level``, ``se``, ``ci_low``,
        ``ci_high`` (``se``/bounds are ``nan`` without a covariance).
    """
    from statistics import NormalDist

    periods = np.atleast_1d(np.asarray(return_periods, dtype=float))
    if np.any(periods <= 0):
        raise ValueError("return periods must be positive")
    if observations_per_period <= 0:
        raise ValueError("observations_per_period must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    zeta = float(fit.exceedance_fraction)
    if not 0 < zeta <= 1:
        raise ValueError("fit.exceedance_fraction must be in (0, 1]")
    xi, beta, u = float(fit.xi), float(fit.beta), float(fit.threshold)
    n_total = int(round(fit.n_exceedances / zeta))

    expected = periods * observations_per_period * zeta
    if np.any(expected <= 1.0):
        raise ValueError(
            "return period too short for this threshold: fewer than one "
            "expected exceedance; lower the threshold or lengthen the period"
        )

    def level(z, c, b):
        arg = periods * observations_per_period * z
        if abs(c) < 1e-10:
            return u + b * np.log(arg)
        return u + (b / c) * (arg**c - 1.0)

    r = level(zeta, xi, beta)
    se = np.full_like(r, np.nan)
    if fit.covariance is not None:
        cov = np.zeros((3, 3))
        cov[0, 0] = zeta * (1.0 - zeta) / max(n_total, 1)
        cov[1:, 1:] = np.asarray(fit.covariance, dtype=float)
        # numeric gradient of the return level in (zeta, xi, beta)
        theta = np.array([zeta, xi, beta])
        grad = np.empty((3, r.size))
        for i in range(3):
            h = 1e-6 * max(abs(theta[i]), 1e-6)
            up, dn = theta.copy(), theta.copy()
            up[i] += h
            dn[i] -= h
            grad[i] = (level(*up) - level(*dn)) / (2 * h)
        var = np.einsum("ik,ij,jk->k", grad, cov, grad)
        se = np.sqrt(np.maximum(var, 0.0))
    z_crit = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {
        "return_period": periods,
        "return_level": r,
        "se": se,
        "ci_low": r - z_crit * se,
        "ci_high": r + z_crit * se,
    }
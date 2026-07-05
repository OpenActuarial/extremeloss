from __future__ import annotations

import numpy as np
from scipy.stats import genextreme

from ..results import GEVFit
from ..utils.validation import as_1d_float_array, validate_positive


def make_blocks(data, block_size: int, *, drop_last: bool = True) -> np.ndarray:
    validate_positive(block_size, name="block_size")
    x = as_1d_float_array(data, name="data")
    block_size = int(block_size)
    n_blocks = x.size // block_size
    if not drop_last and x.size % block_size:
        n_blocks += 1
    if n_blocks < 1:
        raise ValueError("block_size is larger than the data length")
    maxima = []
    for i in range(n_blocks):
        start = i * block_size
        stop = min((i + 1) * block_size, x.size)
        block = x[start:stop]
        if block.size == 0:
            continue
        maxima.append(float(np.max(block)))
    out = np.asarray(maxima, dtype=float)
    if out.size < 2:
        raise ValueError("at least two blocks are required for GEV fitting")
    return out


def fit_gev(block_maxima, method: str = "mle", *, block_size: int | None = None) -> GEVFit:
    if method != "mle":
        raise ValueError("only method='mle' is currently supported")
    x = as_1d_float_array(block_maxima, name="block_maxima")
    if x.size < 2:
        raise ValueError("at least two block maxima are required")
    c_hat, loc_hat, scale_hat = genextreme.fit(x)
    if scale_hat <= 0.0:
        raise RuntimeError("GEV fit returned a nonpositive scale parameter")
    return GEVFit(
        xi=float(-c_hat),
        loc=float(loc_hat),
        scale=float(scale_hat),
        n_blocks=int(x.size),
        block_size=int(block_size) if block_size is not None else None,
        fit_method=method,
        covariance=_gev_covariance(x, float(-c_hat), float(loc_hat),
                                   float(scale_hat)),
    )


def _gev_covariance(block_maxima, xi: float, loc: float, scale: float):
    """Observed-information covariance of ``(xi, loc, scale)`` at the MLE.

    Numerical Hessian of the GEV log-likelihood in the package's own
    parameterization (``xi`` positive for heavy tails; SciPy's
    ``genextreme`` shape is ``c = -xi``). ``None`` when the information is
    not positive definite -- the GEV MLE is irregular for ``xi <= -1/2``,
    and a covariance that means nothing is worse than none.
    """
    x = np.asarray(block_maxima, dtype=float)

    def loglik(theta):
        c_xi, c_loc, c_scale = float(theta[0]), float(theta[1]), float(theta[2])
        if c_scale <= 0.0:
            return -np.inf
        return float(np.sum(genextreme.logpdf(x, c=-c_xi, loc=c_loc,
                                              scale=c_scale)))

    theta = np.array([xi, loc, scale], dtype=float)
    h = 1e-5 * np.maximum(np.abs(theta), 1e-3)
    f0 = loglik(theta)
    hess = np.empty((3, 3))
    for i in range(3):
        ei = np.zeros(3)
        ei[i] = h[i]
        hess[i, i] = (loglik(theta + ei) - 2 * f0 + loglik(theta - ei)) / h[i] ** 2
        for j in range(i + 1, 3):
            ej = np.zeros(3)
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (
                loglik(theta + ei + ej)
                - loglik(theta + ei - ej)
                - loglik(theta - ei + ej)
                + loglik(theta - ei - ej)
            ) / (4 * h[i] * h[j])
    if not np.all(np.isfinite(hess)):
        return None
    info = -hess
    try:
        np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.inv(info)


def fit_block_maxima(data, block_size: int, method: str = "mle", *, drop_last: bool = True) -> GEVFit:
    maxima = make_blocks(data, block_size=block_size, drop_last=drop_last)
    return fit_gev(maxima, method=method, block_size=block_size)


def block_return_level(period: float, fit: GEVFit) -> float:
    if period <= 1.0:
        raise ValueError("period must exceed 1.0")
    return fit.return_level(period)


def gev_return_level(
    fit,
    return_periods,
    confidence_level: float = 0.95,
):
    r"""Block-maxima return levels with confidence intervals.

    The ``T``-block return level is the GEV quantile at ``1 - 1/T`` --
    identically what :meth:`GEVFit.return_level` computes for a single
    period. Confidence intervals are by the delta method over
    ``(xi, loc, scale)`` (Coles, 2001, section 3.3.3); they require the
    fit to carry a parameter ``covariance`` (populated by
    :func:`fit_gev` when the information matrix is positive definite).
    Periods are in *blocks*: with annual maxima, ``T = 100`` is the
    100-year level.

    Returns
    -------
    dict of str -> numpy.ndarray
        ``return_period``, ``return_level``, ``se``, ``ci_low``,
        ``ci_high`` (``se``/bounds are ``nan`` without a covariance).
    """
    from statistics import NormalDist

    periods = np.atleast_1d(np.asarray(return_periods, dtype=float))
    if np.any(periods <= 1.0):
        raise ValueError("return periods must exceed 1 block")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    xi, loc, scale = float(fit.xi), float(fit.loc), float(fit.scale)

    def level(c_xi, c_loc, c_scale):
        y = -np.log(1.0 - 1.0 / periods)  # reduced variate
        if abs(c_xi) < 1e-10:
            return c_loc - c_scale * np.log(y)
        return c_loc + (c_scale / c_xi) * (y ** (-c_xi) - 1.0)

    r = level(xi, loc, scale)
    se = np.full_like(r, np.nan)
    if fit.covariance is not None:
        cov = np.asarray(fit.covariance, dtype=float)
        theta = np.array([xi, loc, scale])
        grad = np.empty((3, r.size))
        for i in range(3):
            h = 1e-6 * max(abs(theta[i]), 1e-6)
            up, dn = theta.copy(), theta.copy()
            up[i] += h
            dn[i] -= h
            grad[i] = (level(*up) - level(*dn)) / (2 * h)
        var = np.einsum("ik,ij,jk->k", grad, cov, grad)
        se = np.sqrt(np.maximum(var, 0.0))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {
        "return_period": periods,
        "return_level": r,
        "se": se,
        "ci_low": r - z * se,
        "ci_high": r + z * se,
    }

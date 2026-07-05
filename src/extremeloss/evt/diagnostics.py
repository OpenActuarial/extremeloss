"""Fitted-model diagnostics: did the tail fit actually succeed.

The threshold tools answer "where does GPD behavior start"; nothing in the
package answered "did this fit describe the data". These functions do,
numerically (QQ/PP point sets and a parametric-bootstrap goodness-of-fit
test) with graphical companions in :mod:`extremeloss.plotting`.

The p-values are parametric-bootstrap p-values: each replicate simulates
from the fitted model and **refits** before computing its statistic.
Comparing a statistic computed with estimated parameters against the
known-parameter null distribution (what a naive ``scipy.stats.kstest``
p-value does) is anti-conservative -- the fit has already chased the
sample -- and the refit-per-replicate design is the standard correction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genextreme, genpareto

from ..results import GEVFit, GPDFit
from ..utils.validation import as_1d_float_array

__all__ = ["parametric_bootstrap_gof", "pp_points", "qq_points"]


def _conditional_pieces(fit, data):
    """(sorted sample, cdf callable, simulate callable, refit-cdf factory)."""
    if isinstance(fit, GPDFit):
        x = as_1d_float_array(data)
        exc = np.sort(x[x > fit.threshold] - fit.threshold)
        if exc.size < 2:
            raise ValueError("fewer than two exceedances above the threshold")

        def cdf(v):
            return genpareto.cdf(v, c=fit.xi, scale=fit.beta)

        def simulate(rng, n):
            return genpareto.rvs(c=fit.xi, scale=fit.beta, size=n,
                                 random_state=rng)

        def refit_cdf(sample):
            c_hat, _, b_hat = genpareto.fit(sample, floc=0.0)
            return lambda v: genpareto.cdf(v, c=c_hat, scale=b_hat)

        return exc, cdf, simulate, refit_cdf, fit.threshold
    if isinstance(fit, GEVFit):
        x = np.sort(as_1d_float_array(data))
        if x.size < 2:
            raise ValueError("fewer than two block maxima")

        def cdf(v):
            return genextreme.cdf(v, c=-fit.xi, loc=fit.loc, scale=fit.scale)

        def simulate(rng, n):
            return genextreme.rvs(c=-fit.xi, loc=fit.loc, scale=fit.scale,
                                  size=n, random_state=rng)

        def refit_cdf(sample):
            c_hat, l_hat, s_hat = genextreme.fit(sample)
            return lambda v: genextreme.cdf(v, c=c_hat, loc=l_hat,
                                            scale=s_hat)

        return x, cdf, simulate, refit_cdf, 0.0
    raise TypeError(
        f"expected a GPDFit or GEVFit; got {type(fit).__name__}"
    )


def qq_points(fit, data) -> dict[str, np.ndarray]:
    """Quantile-quantile point set for a fitted tail.

    For a :class:`GPDFit`, empirical quantiles are the sorted losses above
    the threshold and theoretical quantiles are the fitted conditional GPD
    quantiles at the plotting positions ``(i - 0.5) / n`` (both in
    original loss units, threshold included). For a :class:`GEVFit`, the
    block maxima against fitted GEV quantiles. A good fit puts the points
    on the 45-degree line -- deviations in the upper corner are exactly
    where a tail model earns or loses its keep.

    Returns
    -------
    dict
        ``theoretical``, ``empirical`` (equal-length arrays), ``n``.
    """
    sample, _, _, _, shift = _conditional_pieces(fit, data)
    n = sample.size
    p = (np.arange(1, n + 1) - 0.5) / n
    if isinstance(fit, GPDFit):
        theo = shift + genpareto.ppf(p, c=fit.xi, scale=fit.beta)
        emp = shift + sample
    else:
        theo = genextreme.ppf(p, c=-fit.xi, loc=fit.loc, scale=fit.scale)
        emp = sample
    return {"theoretical": theo, "empirical": emp, "n": n}


def pp_points(fit, data) -> dict[str, np.ndarray]:
    """Probability-probability point set: model cdf vs plotting positions.

    Complements :func:`qq_points`: PP is most sensitive in the body of the
    fitted range, QQ in the tail. Both on the unit square.
    """
    sample, cdf, _, _, _ = _conditional_pieces(fit, data)
    n = sample.size
    return {
        "empirical": (np.arange(1, n + 1) - 0.5) / n,
        "model": np.asarray(cdf(sample), dtype=float),
        "n": n,
    }


def _ks_ad(u: np.ndarray) -> tuple[float, float]:
    """KS and Anderson-Darling statistics from model-cdf values (sorted)."""
    n = u.size
    i = np.arange(1, n + 1)
    ks = float(np.max(np.maximum(i / n - u, u - (i - 1) / n)))
    eps = 1e-12
    u_c = np.clip(u, eps, 1 - eps)
    ad = float(-n - np.mean((2 * i - 1) * (np.log(u_c)
                                           + np.log(1 - u_c[::-1]))))
    return ks, ad


def parametric_bootstrap_gof(fit, data, n_boot: int = 500, rng=None) -> dict:
    """Goodness-of-fit test for a fitted GPD or GEV, done honestly.

    Kolmogorov-Smirnov and Anderson-Darling statistics of the data against
    the fitted model, with p-values from a parametric bootstrap that
    refits inside every replicate (see the module docstring for why the
    refit is not optional). A-D weights the tails, K-S the body; report
    both, because a tail model can pass one and fail the other.

    Returns
    -------
    dict
        ``ks``, ``ks_pvalue``, ``ad``, ``ad_pvalue``, ``n``, ``n_boot``.
    """
    if n_boot < 19:
        raise ValueError("n_boot must be at least 19 for a meaningful p-value")
    sample, cdf, simulate, refit_cdf, _ = _conditional_pieces(fit, data)
    gen = np.random.default_rng(rng)
    ks_obs, ad_obs = _ks_ad(np.asarray(cdf(sample), dtype=float))
    ks_ge = ad_ge = 0
    n = sample.size
    for _ in range(n_boot):
        sim = np.sort(simulate(gen, n))
        try:
            boot_cdf = refit_cdf(sim)
        except Exception:  # rare degenerate refit: count as extreme
            ks_ge += 1
            ad_ge += 1
            continue
        ks_b, ad_b = _ks_ad(np.asarray(boot_cdf(sim), dtype=float))
        ks_ge += ks_b >= ks_obs
        ad_ge += ad_b >= ad_obs
    return {
        "ks": ks_obs,
        "ks_pvalue": (1 + ks_ge) / (n_boot + 1),
        "ad": ad_obs,
        "ad_pvalue": (1 + ad_ge) / (n_boot + 1),
        "n": int(n),
        "n_boot": int(n_boot),
    }

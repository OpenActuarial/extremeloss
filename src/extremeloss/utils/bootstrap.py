from __future__ import annotations

import numpy as np

from ..estimation.metrics import empirical_tvar, empirical_var, exceedance_probability
from ..results import BootstrapResult
from .validation import as_1d_float_array, validate_alpha, validate_positive


def bootstrap_statistic(
    data,
    statistic,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    r"""Nonparametric bootstrap of an arbitrary statistic, with a percentile CI.

    Resamples ``data`` with replacement ``n_resamples`` times, evaluates
    ``statistic`` on each resample, and summarizes the bootstrap distribution
    with a point estimate (the statistic on the original sample), a
    percentile confidence interval, and a bootstrap standard error. The
    interval is the equal-tailed percentile interval with endpoints at the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the resampled statistics.

    Parameters
    ----------
    data : array-like
        The sample to resample.
    statistic : callable
        A function mapping a 1-D array to a float, e.g. ``np.mean`` or a
        tail measure. It is called once on the original data for the point
        estimate and once per resample.
    n_resamples : int, optional
        Number of bootstrap resamples (default 1000). Must be positive.
    alpha : float, optional
        Significance level for the two-sided percentile interval (default
        0.05, i.e. a 95% interval). Must lie in ``(0, 1)``.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator controlling the resampling. An existing
        ``Generator`` is used as given; anything else is passed to
        :func:`numpy.random.default_rng`. Fix it for reproducible intervals.

    Returns
    -------
    BootstrapResult
        Carries ``estimate`` (statistic on the original sample),
        ``bootstrap_estimates`` (the array of resampled values), ``ci``
        (the ``(low, high)`` percentile bounds), ``stderr`` (standard
        deviation of the resampled values, ``ddof=1``), ``method``
        (``"percentile"``), and ``alpha``.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not positive or ``alpha`` is not in ``(0, 1)``.

    Notes
    -----
    The percentile interval is simple and distribution-free but can
    under-cover for strongly skewed sampling distributions -- the same
    caveat that applies to Wald tail intervals elsewhere in the package.

    See Also
    --------
    bootstrap_var : Bootstrap CI for value-at-risk.
    bootstrap_tvar : Bootstrap CI for tail value-at-risk.
    bootstrap_tail_probability : Bootstrap CI for an exceedance probability.
    """
    validate_positive(n_resamples, name="n_resamples")
    validate_alpha(alpha)
    x = as_1d_float_array(data, name="data")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    point = float(statistic(x))
    boot = np.empty(int(n_resamples), dtype=float)
    n = x.size
    for i in range(int(n_resamples)):
        sample = x[rng.integers(0, n, size=n)]
        boot[i] = float(statistic(sample))
    ci = tuple(np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())
    stderr = float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0
    return BootstrapResult(
        estimate=point,
        bootstrap_estimates=boot,
        method="percentile",
        ci=(float(ci[0]), float(ci[1])),
        stderr=stderr,
        alpha=float(alpha),
    )


def bootstrap_tail_probability(losses, threshold: float, **kwargs) -> BootstrapResult:
    r"""Bootstrap confidence interval for an exceedance probability.

    Convenience wrapper over :func:`bootstrap_statistic` with the statistic
    fixed to the empirical probability :math:`P(X > u)` of exceeding
    ``threshold``.

    Parameters
    ----------
    losses : array-like
        Loss sample to resample.
    threshold : float
        The level :math:`u` whose exceedance probability is bootstrapped.
    **kwargs
        Forwarded to :func:`bootstrap_statistic` (``n_resamples``, ``alpha``,
        ``random_state``).

    Returns
    -------
    BootstrapResult
        As returned by :func:`bootstrap_statistic`, for the exceedance
        probability.

    See Also
    --------
    bootstrap_statistic : The general routine this delegates to.
    """
    return bootstrap_statistic(losses, lambda x: exceedance_probability(x, threshold), **kwargs)


def bootstrap_var(losses, q: float, **kwargs) -> BootstrapResult:
    r"""Bootstrap confidence interval for value-at-risk (a loss quantile).

    Convenience wrapper over :func:`bootstrap_statistic` with the statistic
    fixed to the empirical value-at-risk at level ``q`` -- the ``q``-quantile
    of the loss distribution.

    Parameters
    ----------
    losses : array-like
        Loss sample to resample.
    q : float
        VaR level in ``(0, 1)``, e.g. ``0.99`` for the 99% VaR.
    **kwargs
        Forwarded to :func:`bootstrap_statistic` (``n_resamples``, ``alpha``,
        ``random_state``).

    Returns
    -------
    BootstrapResult
        As returned by :func:`bootstrap_statistic`, for the VaR.

    See Also
    --------
    bootstrap_tvar : Bootstrap CI for the tail value-at-risk at the same level.
    bootstrap_statistic : The general routine this delegates to.
    """
    return bootstrap_statistic(losses, lambda x: empirical_var(x, q), **kwargs)


def bootstrap_tvar(losses, q: float, **kwargs) -> BootstrapResult:
    r"""Bootstrap confidence interval for tail value-at-risk.

    Convenience wrapper over :func:`bootstrap_statistic` with the statistic
    fixed to the empirical tail value-at-risk at level ``q`` -- the mean loss
    in the worst ``1 - q`` fraction of outcomes (also called CVaR or expected
    shortfall).

    Parameters
    ----------
    losses : array-like
        Loss sample to resample.
    q : float
        TVaR level in ``(0, 1)``, e.g. ``0.99`` for the 99% TVaR.
    **kwargs
        Forwarded to :func:`bootstrap_statistic` (``n_resamples``, ``alpha``,
        ``random_state``).

    Returns
    -------
    BootstrapResult
        As returned by :func:`bootstrap_statistic`, for the TVaR.

    See Also
    --------
    bootstrap_var : Bootstrap CI for the value-at-risk at the same level.
    bootstrap_statistic : The general routine this delegates to.
    """
    return bootstrap_statistic(losses, lambda x: empirical_tvar(x, q), **kwargs)

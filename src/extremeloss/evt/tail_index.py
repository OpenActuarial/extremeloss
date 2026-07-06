from __future__ import annotations

import numpy as np

from ..utils.validation import as_1d_float_array


def _sorted_positive_tail(data) -> np.ndarray:
    x = as_1d_float_array(data, name="data")
    if np.any(x <= 0.0):
        raise ValueError("data must contain only positive values")
    return np.sort(x)


def hill_estimator(data, k: int) -> float:
    r"""Hill estimator of the tail index from the ``k`` largest observations.

    For positive data with an approximately Pareto upper tail, the Hill
    estimator of the tail index :math:`\gamma = 1/\alpha` (equivalently the
    shape parameter :math:`\xi` of the corresponding GPD) is the mean log
    spacing of the ``k`` largest order statistics above the ``(k+1)``-th:

    .. math::
        \hat{\gamma}_k = \frac{1}{k} \sum_{i=1}^{k}
        \bigl(\log X_{(n-i+1)} - \log X_{(n-k)}\bigr)

    where :math:`X_{(1)} \le \dots \le X_{(n)}` are the sorted values.
    The estimate is sensitive to the choice of ``k``: small ``k`` gives high
    variance, large ``k`` introduces bias by reaching into the distribution
    body. Sweep ``k`` with :func:`hill_curve` and read the tail index from a
    stable region of the resulting plot.

    Parameters
    ----------
    data : array-like
        Strictly positive observations. Order is irrelevant; the values are
        sorted internally.
    k : int
        Number of upper order statistics used, satisfying
        ``1 <= k < len(data)``.

    Returns
    -------
    float
        The Hill tail-index estimate :math:`\hat{\gamma}_k`.

    Raises
    ------
    ValueError
        If ``data`` contains a non-positive value, or if ``k`` is outside
        ``1 <= k < len(data)``.

    See Also
    --------
    hill_curve : Hill estimate across a grid of ``k``.
    pickands_estimator : Alternative tail-index estimator valid for any
        real tail index.
    """
    x = _sorted_positive_tail(data)
    n = x.size
    if k <= 0 or k >= n:
        raise ValueError("k must satisfy 1 <= k < len(data)")
    x_top = x[-k:]
    x_k1 = x[-k - 1]
    estimate = np.mean(np.log(x_top) - np.log(x_k1))
    return float(estimate)


def pickands_estimator(data, k: int) -> float:
    r"""Pickands estimator of the tail index from ordered tail spacings.

    Unlike the Hill estimator, the Pickands estimator is valid for any real
    tail index :math:`\gamma` (light, heavy, or bounded tails) and requires
    no positivity of the data beyond the internal ordering. Using the sorted
    values :math:`X_{(1)} \le \dots \le X_{(n)}`, it compares spacings at the
    ``k``-th, ``2k``-th, and ``4k``-th largest observations:

    .. math::
        \hat{\gamma}_k = \frac{1}{\log 2}\,
        \log\!\left(
        \frac{X_{(n-k+1)} - X_{(n-2k+1)}}{X_{(n-2k+1)} - X_{(n-4k+1)}}
        \right)

    As with any tail-index estimator the result depends on ``k``; it is
    typically read from a stable region of a plot over ``k``.

    Parameters
    ----------
    data : array-like
        Strictly positive observations. Order is irrelevant; the values are
        sorted internally.
    k : int
        Order-statistic spacing parameter, satisfying ``4k < len(data) + 1``
        so that the ``4k``-th largest observation exists.

    Returns
    -------
    float
        The Pickands tail-index estimate :math:`\hat{\gamma}_k`.

    Raises
    ------
    ValueError
        If ``data`` contains a non-positive value, if ``k`` violates
        ``4k < len(data) + 1``, or if the ordered tail spacings are not
        both strictly positive (which can occur with ties or a short tail).

    See Also
    --------
    hill_estimator : Log-spacing estimator for heavy (positive-index) tails.
    """
    x = _sorted_positive_tail(data)
    n = x.size
    if k <= 0 or 4 * k >= n + 1:
        raise ValueError("k must satisfy 4k < len(data) + 1")
    x1 = x[-k]
    x2 = x[-2 * k]
    x4 = x[-4 * k]
    numerator = x1 - x2
    denominator = x2 - x4
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("Pickands estimator requires ordered tail spacings to be positive")
    return float(np.log(numerator / denominator) / np.log(2.0))


def hill_curve(data, k_grid=None) -> dict[str, np.ndarray]:
    r"""Hill tail-index estimate across a grid of ``k``, for a Hill plot.

    Evaluates :func:`hill_estimator` at each ``k`` in ``k_grid``. Plotting the
    returned ``hill`` values against ``k`` produces the standard *Hill plot*;
    the tail index is read from a region where the estimate is roughly flat,
    balancing the variance of small ``k`` against the bias of large ``k``.

    Parameters
    ----------
    data : array-like
        Strictly positive observations.
    k_grid : array-like of int, optional
        Values of ``k`` at which to evaluate the estimator. When ``None``
        (the default), uses ``1, 2, ..., max(2, len(data) // 4)`` so the grid
        stays within the upper quarter of the sample.

    Returns
    -------
    dict of str -> numpy.ndarray
        ``k`` -- the grid of order-statistic counts used, and ``hill`` --
        the corresponding Hill estimates, aligned elementwise.

    Raises
    ------
    ValueError
        If ``data`` contains a non-positive value, or if any ``k`` in the
        grid violates ``1 <= k < len(data)``.

    See Also
    --------
    hill_estimator : The per-``k`` estimator evaluated here.
    """
    x = _sorted_positive_tail(data)
    n = x.size
    if k_grid is None:
        upper = max(2, n // 4)
        k_grid = np.arange(1, upper + 1, dtype=int)
    else:
        k_grid = np.asarray(k_grid, dtype=int)
    estimates = np.array([hill_estimator(x, int(k)) for k in k_grid], dtype=float)
    return {"k": k_grid, "hill": estimates}

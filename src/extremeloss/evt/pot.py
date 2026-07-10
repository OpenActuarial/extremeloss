from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from ..results import GPDFit
from ..utils.validation import as_1d_float_array, validate_threshold


def extract_exceedances(data, threshold: float) -> np.ndarray:
    r"""Excesses of the data over a threshold (peaks-over-threshold data).

    Returns the positive excesses :math:`X_i - u` for every observation above
    the threshold :math:`u` -- the exceedance data on which a generalized
    Pareto tail is fitted. Note the return is *excesses* (measured from the
    threshold), not the raw exceeding values, matching the convention of the
    GPD fitters, which take excesses with location fixed at zero.

    Parameters
    ----------
    data : array-like
        Observations to threshold.
    threshold : float
        The threshold :math:`u`. Only strictly-exceeding observations
        (``x > threshold``) contribute.

    Returns
    -------
    numpy.ndarray
        The excesses ``x - threshold`` for all ``x > threshold``, in the
        original data order.

    Raises
    ------
    ValueError
        If ``threshold`` is invalid, or if no observation exceeds it (an
        empty exceedance set cannot support a tail fit).

    See Also
    --------
    fit_gpd : Fit a GPD to a set of excesses.
    """
    validate_threshold(threshold)
    x = as_1d_float_array(data, name="data")
    exceedances = x[x > threshold] - threshold
    if exceedances.size == 0:
        raise ValueError("no exceedances found above the specified threshold")
    return exceedances


def fit_pot(data, threshold: float, method: str = "mle") -> GPDFit:
    r"""Peaks-over-threshold fit: a GPD for the excesses of ``data`` over ``threshold``.

    Selects the observations strictly above the threshold :math:`u`, fits a
    generalized Pareto distribution to their excesses :math:`x - u` by maximum
    likelihood (location fixed at zero), and returns a :class:`GPDFit` carrying
    the empirical exceedance rate ``n_exceedances / n`` -- so the fit quotes
    *unconditional*, ground-up tail quantities (``sf``, VaR/TVaR, return
    levels) out of the box. The observed-information covariance of
    ``(xi, beta)`` is attached when the information matrix is positive
    definite (``None`` otherwise; the MLE is irregular for
    :math:`\xi \le -1/2`).

    Parameters
    ----------
    data : array-like
        Ground-up loss observations (the full sample, not pre-extracted
        excesses -- for excesses already in hand use :func:`fit_gpd`).
    threshold : float
        The POT threshold :math:`u`. Choose it with
        :func:`threshold_diagnostic_table`.
    method : str, optional
        Only ``"mle"`` is currently supported.

    Returns
    -------
    GPDFit
        Fitted ``(xi, beta)`` above ``threshold``, with
        ``exceedance_fraction``, ``n_exceedances``, and (when available)
        the parameter ``covariance``.

    Raises
    ------
    ValueError
        If no observation exceeds the threshold, or ``method`` is not
        ``"mle"``.

    See Also
    --------
    fit_gpd : The same fit when the excesses are already extracted.
    threshold_diagnostic_table : Scan candidate thresholds before committing.
    """
    if method != "mle":
        raise ValueError("only method='mle' is currently supported")
    x = as_1d_float_array(data, name="data")
    exceedances = extract_exceedances(x, threshold)
    xi_hat, loc_hat, beta_hat = genpareto.fit(exceedances, floc=0.0)
    if loc_hat != 0.0:
        raise RuntimeError("GPD fit returned nonzero location despite floc=0")
    from .gpd import _gpd_covariance

    return GPDFit(
        threshold=float(threshold),
        xi=float(xi_hat),
        beta=float(beta_hat),
        covariance=_gpd_covariance(exceedances, float(xi_hat), float(beta_hat)),
        exceedance_fraction=float(exceedances.size / x.size),
        n_exceedances=int(exceedances.size),
        fit_method=method,
    )

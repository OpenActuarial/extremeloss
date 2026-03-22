from __future__ import annotations

import math

import numpy as np
from scipy.stats import genpareto

from ..results import GPDFit
from ..utils.validation import as_1d_float_array, validate_q, validate_threshold


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
    )


def gpd_tail_probability(
    x: float,
    threshold: float,
    xi: float,
    beta: float,
    exceedance_fraction: float,
) -> float:
    validate_threshold(threshold)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if x <= threshold:
        return float(min(1.0, exceedance_fraction))
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
    validate_q(p)
    validate_threshold(threshold)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if exceedance_fraction <= 0.0 or exceedance_fraction >= 1.0:
        raise ValueError("exceedance_fraction must lie strictly between 0 and 1")
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
    if xi >= 1.0:
        raise ValueError("TVaR is infinite for xi >= 1")
    var_p = gpd_var(p, threshold, xi, beta, exceedance_fraction)
    return float((var_p + beta - xi * threshold) / (1.0 - xi))
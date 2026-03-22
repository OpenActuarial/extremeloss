from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from ..results import GPDFit
from ..utils.validation import as_1d_float_array, validate_threshold


def extract_exceedances(data, threshold: float) -> np.ndarray:
    validate_threshold(threshold)
    x = as_1d_float_array(data, name="data")
    exceedances = x[x > threshold] - threshold
    if exceedances.size == 0:
        raise ValueError("no exceedances found above the specified threshold")
    return exceedances


def fit_pot(data, threshold: float, method: str = "mle") -> GPDFit:
    if method != "mle":
        raise ValueError("only method='mle' is currently supported")
    x = as_1d_float_array(data, name="data")
    exceedances = extract_exceedances(x, threshold)
    xi_hat, loc_hat, beta_hat = genpareto.fit(exceedances, floc=0.0)
    if loc_hat != 0.0:
        raise RuntimeError("GPD fit returned nonzero location despite floc=0")
    return GPDFit(
        threshold=float(threshold),
        xi=float(xi_hat),
        beta=float(beta_hat),
        exceedance_fraction=float(exceedances.size / x.size),
        n_exceedances=int(exceedances.size),
        fit_method=method,
    )
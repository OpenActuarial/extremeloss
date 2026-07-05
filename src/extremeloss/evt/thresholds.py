from __future__ import annotations

import numpy as np

from ..results import ThresholdScan
from ..utils.validation import as_1d_float_array
from .pot import fit_pot


def mean_excess(data, thresholds) -> dict[str, np.ndarray]:
    x = as_1d_float_array(data, name="data")
    grid = as_1d_float_array(thresholds, name="thresholds")
    values = []
    counts = []
    for u in grid:
        exceedances = x[x > u] - u
        counts.append(int(exceedances.size))
        if exceedances.size == 0:
            values.append(np.nan)
        else:
            values.append(float(np.mean(exceedances)))
    return {
        "thresholds": grid,
        "mean_excess": np.asarray(values, dtype=float),
        "n_exceedances": np.asarray(counts, dtype=int),
    }


def threshold_diagnostic_table(data, thresholds) -> ThresholdScan:
    x = as_1d_float_array(data, name="data")
    grid = as_1d_float_array(thresholds, name="thresholds")
    me = []
    xi = []
    beta = []
    counts = []
    xi_se = []
    mod = []
    mod_se = []
    for u in grid:
        exceedances = x[x > u] - u
        counts.append(int(exceedances.size))
        if exceedances.size < 5:
            for acc in (me, xi, beta, xi_se, mod, mod_se):
                acc.append(np.nan)
            continue
        me.append(float(np.mean(exceedances)))
        fit = fit_pot(x, float(u))
        xi.append(float(fit.xi))
        beta.append(float(fit.beta))
        # modified scale beta* = beta - xi * u: constant in u above a valid
        # threshold, unlike raw beta; its variance follows by the delta
        # method with gradient (-u, 1) over (xi, beta)
        mod.append(float(fit.beta - fit.xi * u))
        if fit.covariance is not None:
            cov = np.asarray(fit.covariance, dtype=float)
            g = np.array([-float(u), 1.0])
            mod_var = float(g @ cov @ g)
            xi_se.append(float(np.sqrt(max(cov[0, 0], 0.0))))
            mod_se.append(float(np.sqrt(max(mod_var, 0.0))))
        else:
            xi_se.append(np.nan)
            mod_se.append(np.nan)
    return ThresholdScan(
        thresholds=grid,
        mean_excess=np.asarray(me, dtype=float),
        xi=np.asarray(xi, dtype=float),
        beta=np.asarray(beta, dtype=float),
        n_exceedances=np.asarray(counts, dtype=int),
        xi_se=np.asarray(xi_se, dtype=float),
        modified_scale=np.asarray(mod, dtype=float),
        modified_scale_se=np.asarray(mod_se, dtype=float),
    )

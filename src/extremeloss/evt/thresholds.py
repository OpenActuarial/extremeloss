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
    for u in grid:
        exceedances = x[x > u] - u
        counts.append(int(exceedances.size))
        if exceedances.size < 5:
            me.append(np.nan)
            xi.append(np.nan)
            beta.append(np.nan)
            continue
        me.append(float(np.mean(exceedances)))
        fit = fit_pot(x, float(u))
        xi.append(float(fit.xi))
        beta.append(float(fit.beta))
    return ThresholdScan(
        thresholds=grid,
        mean_excess=np.asarray(me, dtype=float),
        xi=np.asarray(xi, dtype=float),
        beta=np.asarray(beta, dtype=float),
        n_exceedances=np.asarray(counts, dtype=int),
    )
from __future__ import annotations

import numpy as np

from ..estimation.metrics import empirical_tvar, empirical_var, exceedance_curve
from ..utils.validation import as_1d_float_array


def var_tvar_diagnostic_table(losses, quantiles=(0.95, 0.99, 0.995)) -> dict[str, object]:
    arr = as_1d_float_array(losses, name="losses")
    rows = []
    for q in quantiles:
        var_q = empirical_var(arr, q)
        tvar_q = empirical_tvar(arr, q)
        rows.append(
            {
                "quantile": float(q),
                "var": float(var_q),
                "tvar": float(tvar_q),
                "tail_ratio": float(tvar_q / var_q) if var_q > 0.0 else np.nan,
            }
        )
    return {"n": int(arr.size), "rows": rows}


def extreme_loss_summary(
    losses,
    *,
    thresholds=None,
    quantiles=(0.95, 0.99, 0.995),
) -> dict[str, object]:
    arr = as_1d_float_array(losses, name="losses")
    out: dict[str, object] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "var_tvar": var_tvar_diagnostic_table(arr, quantiles=quantiles)["rows"],
    }
    if thresholds is not None:
        out["exceedance_curve"] = exceedance_curve(arr, thresholds)
    return out
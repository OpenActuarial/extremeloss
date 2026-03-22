from __future__ import annotations

import numpy as np

from ..utils.validation import as_1d_float_array, validate_q


def empirical_var(losses, q: float) -> float:
    validate_q(q)
    arr = as_1d_float_array(losses, name="losses")
    return float(np.quantile(arr, q))


def empirical_tvar(losses, q: float) -> float:
    validate_q(q)
    arr = as_1d_float_array(losses, name="losses")
    threshold = empirical_var(arr, q)
    tail = arr[arr >= threshold]
    if tail.size == 0:
        return threshold
    return float(np.mean(tail))


def exceedance_probability(losses, threshold: float) -> float:
    arr = as_1d_float_array(losses, name="losses")
    return float(np.mean(arr > threshold))


def exceedance_curve(losses, thresholds) -> dict[str, np.ndarray]:
    arr = as_1d_float_array(losses, name="losses")
    grid = as_1d_float_array(thresholds, name="thresholds")
    probs = np.array([np.mean(arr > u) for u in grid], dtype=float)
    return {"thresholds": grid, "probabilities": probs}


def survival_function(losses, grid) -> dict[str, np.ndarray]:
    return exceedance_curve(losses, grid)
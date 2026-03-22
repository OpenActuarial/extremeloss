from __future__ import annotations

import numpy as np

from ..utils.validation import as_1d_float_array


def _sorted_positive_tail(data) -> np.ndarray:
    x = as_1d_float_array(data, name="data")
    if np.any(x <= 0.0):
        raise ValueError("data must contain only positive values")
    return np.sort(x)


def hill_estimator(data, k: int) -> float:
    x = _sorted_positive_tail(data)
    n = x.size
    if k <= 0 or k >= n:
        raise ValueError("k must satisfy 1 <= k < len(data)")
    x_top = x[-k:]
    x_k1 = x[-k - 1]
    estimate = np.mean(np.log(x_top) - np.log(x_k1))
    return float(estimate)


def pickands_estimator(data, k: int) -> float:
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
    x = _sorted_positive_tail(data)
    n = x.size
    if k_grid is None:
        upper = max(2, n // 4)
        k_grid = np.arange(1, upper + 1, dtype=int)
    else:
        k_grid = np.asarray(k_grid, dtype=int)
    estimates = np.array([hill_estimator(x, int(k)) for k in k_grid], dtype=float)
    return {"k": k_grid, "hill": estimates}
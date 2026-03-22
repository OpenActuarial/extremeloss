from __future__ import annotations

import math

import numpy as np

from ..protocols import SupportsLosses, SupportsSample, SupportsSimulationResult


def as_1d_float_array(values, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array-like object")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def validate_q(q: float) -> None:
    if not (0.0 < q < 1.0):
        raise ValueError("q must be strictly between 0 and 1")


def validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")


def validate_positive(value: float, name: str = "value") -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_size(size: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")


def validate_weights(weights) -> np.ndarray:
    w = as_1d_float_array(weights, name="weights")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w


def validate_probabilities(probabilities, name: str = "probabilities") -> np.ndarray:
    arr = as_1d_float_array(probabilities, name=name)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr


def coerce_losses(data, size: int | None = None) -> np.ndarray:
    if isinstance(data, np.ndarray) or isinstance(data, (list, tuple)):
        return as_1d_float_array(data, name="losses")
    if isinstance(data, SupportsSimulationResult) or hasattr(data, "losses"):
        return as_1d_float_array(getattr(data, "losses"), name="losses")
    if isinstance(data, SupportsLosses) and hasattr(data, "losses"):
        return as_1d_float_array(data.losses, name="losses")
    if isinstance(data, SupportsSample) or hasattr(data, "sample"):
        if size is None:
            raise ValueError("size must be provided when data is a model-like object")
        validate_size(size)
        samples = data.sample(size=size)
        arr = as_1d_float_array(samples, name="sample(size)")
        if arr.size != size:
            raise ValueError(
                f"sample(size={size}) returned {arr.size} values instead of {size}"
            )
        return arr
    raise TypeError(
        "data must be a one-dimensional array-like object, expose losses, or implement sample(size)"
    )
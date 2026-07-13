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


def validate_q(q) -> None:
    """Validate scalar or array-like quantile level(s) in ``(0, 1)``."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.size == 0 or not np.all((q_arr > 0.0) & (q_arr < 1.0)):
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


def validate_gpd_params(
    threshold: float,
    xi: float,
    beta: float,
    exceedance_fraction: float,
    *,
    allow_zero_exceedance: bool = False,
) -> None:
    """Validate the parameter set shared by the GPD tail functions.

    Requires a finite threshold, finite shape ``xi``, a finite positive scale
    ``beta``, and an ``exceedance_fraction`` (the exceedance rate :math:`\\zeta_u`)
    in ``(0, 1]`` -- or ``[0, 1]`` when ``allow_zero_exceedance`` is set, which is
    meaningful for a tail *probability* (a zero exceedance rate gives probability
    zero) but not for an inverse-tail quantile like value-at-risk. This closes the
    gap where an unvalidated ``exceedance_fraction`` could yield a negative or
    above-one probability.
    """
    validate_threshold(threshold)
    for label, val in (("xi", xi), ("beta", beta), ("exceedance_fraction", exceedance_fraction)):
        if not math.isfinite(float(val)):
            raise ValueError(f"{label} must be finite")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    lo_ok = exceedance_fraction >= 0.0 if allow_zero_exceedance else exceedance_fraction > 0.0
    if not (lo_ok and exceedance_fraction <= 1.0):
        rng = "[0, 1]" if allow_zero_exceedance else "(0, 1]"
        raise ValueError(f"exceedance_fraction must lie in {rng}")


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
    # Concrete array-like data first. ``__array__`` covers pandas Series /
    # Index and any other numpy-convertible container; checking it before the
    # protocol branches matters because e.g. ``pandas.Series`` also exposes a
    # ``.sample`` method and would otherwise be misrouted to the model branch.
    if isinstance(data, (np.ndarray, list, tuple)) or hasattr(data, "__array__"):
        return as_1d_float_array(data, name="losses")
    if isinstance(data, SupportsSimulationResult) or isinstance(data, SupportsLosses) or hasattr(data, "losses"):
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

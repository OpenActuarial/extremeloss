from __future__ import annotations

import numpy as np

from .analytics.diagnostics import extreme_loss_summary
from .estimation.metrics import empirical_tvar, empirical_var, exceedance_probability
from .evt.pot import fit_pot
from .utils.validation import as_1d_float_array, coerce_losses, validate_q


_RISKSIM_VIEWS = {
    "losses": "losses",
    "gross": "gross_losses",
    "gross_losses": "gross_losses",
    "retained": "retained_losses",
    "retained_losses": "retained_losses",
    "ceded": "ceded_losses",
    "ceded_losses": "ceded_losses",
}


def sample_lossmodel(model, size: int) -> np.ndarray:
    """Sample losses from a lossmodels-style severity or aggregate model."""
    return coerce_losses(model, size=size)


def losses_from_risksim(result, *, view: str = "losses") -> np.ndarray:
    attr = _RISKSIM_VIEWS.get(view)
    if attr is None:
        raise ValueError(f"unknown view {view!r}")
    if not hasattr(result, attr):
        raise TypeError("result does not expose the requested risksim-style loss view")
    values = getattr(result, attr)
    if values is None:
        raise ValueError(f"requested view {view!r} is not available on this result")
    return as_1d_float_array(values, name=attr)


def fit_pot_from_lossmodel(model, *, size: int, threshold: float):
    losses = sample_lossmodel(model, size=size)
    return fit_pot(losses, threshold=threshold)


def tail_summary_from_risksim(result, *, view: str = "losses", thresholds=None, quantiles=(0.95, 0.99, 0.995)) -> dict[str, object]:
    losses = losses_from_risksim(result, view=view)
    return extreme_loss_summary(losses, thresholds=thresholds, quantiles=quantiles)


def component_tail_metrics(result, *, q: float = 0.99, threshold: float | None = None) -> dict[str, dict[str, float]]:
    validate_q(q)
    if not hasattr(result, "component_losses"):
        raise TypeError("result does not expose component_losses")
    values = np.asarray(result.component_losses, dtype=float)
    if values.ndim != 2:
        raise ValueError("component_losses must be a 2D array")
    names = getattr(result, "component_names", None)
    if names is None:
        names = [f"component_{i}" for i in range(values.shape[1])]
    out: dict[str, dict[str, float]] = {}
    for name, col in zip(names, values.T):
        metrics = {
            "var": float(empirical_var(col, q)),
            "tvar": float(empirical_tvar(col, q)),
        }
        if threshold is not None:
            metrics["exceedance_probability"] = float(exceedance_probability(col, threshold))
        out[str(name)] = metrics
    return out


def layer_tail_metrics(result, *, q: float = 0.99, threshold: float | None = None) -> dict[str, dict[str, float]]:
    validate_q(q)
    if not hasattr(result, "layer_losses"):
        raise TypeError("result does not expose layer_losses")
    values = np.asarray(result.layer_losses, dtype=float)
    if values.ndim != 2:
        raise ValueError("layer_losses must be a 2D array")
    names = getattr(result, "layer_names", None)
    if names is None:
        names = [f"layer_{i}" for i in range(values.shape[1])]
    out: dict[str, dict[str, float]] = {}
    for name, col in zip(names, values.T):
        metrics = {
            "var": float(empirical_var(col, q)),
            "tvar": float(empirical_tvar(col, q)),
        }
        if threshold is not None:
            metrics["exceedance_probability"] = float(exceedance_probability(col, threshold))
        out[str(name)] = metrics
    return out

from __future__ import annotations

import numpy as np

from .analytics.diagnostics import extreme_loss_summary
from .estimation.metrics import empirical_tvar, empirical_var, exceedance_probability
from .evt.pot import fit_pot
from .results import GPDTail
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
    """Extract a 1-D loss array from a risksim-style simulation result.

    Accepts any object exposing the requested attribute -- ``view`` may be
    ``"losses"``, ``"gross"`` / ``"gross_losses"``, ``"retained"`` /
    ``"retained_losses"``, or ``"ceded"`` / ``"ceded_losses"`` -- and
    returns it as a float array. Duck-typed: nothing risksim-specific is
    required beyond the attribute. Raises ``ValueError`` for an unknown
    view or one that is ``None`` on this result, ``TypeError`` if the
    attribute is absent.
    """
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
    """Sample a lossmodels-style severity and POT-fit its tail in one call.

    Draws ``size`` losses from ``model`` (anything
    :func:`sample_lossmodel` accepts) and returns
    ``fit_pot(losses, threshold=threshold)`` -- how a fitted severity's
    tail reads through the peaks-over-threshold lens.
    """
    losses = sample_lossmodel(model, size=size)
    return fit_pot(losses, threshold=threshold)


def tail_summary_from_risksim(result, *, view: str = "losses", thresholds=None, quantiles=(0.95, 0.99, 0.995)) -> dict[str, object]:
    """Tail summary of a risksim simulation result: extract the view, then summarize.

    ``extreme_loss_summary(losses_from_risksim(result, view=view), ...)``
    in one call: ``n`` / ``mean`` / ``std`` / ``min`` / ``max``, empirical
    VaR/TVaR at each quantile, and the exceedance curve when
    ``thresholds`` is given.
    """
    losses = losses_from_risksim(result, view=view)
    return extreme_loss_summary(losses, thresholds=thresholds, quantiles=quantiles)


def component_tail_metrics(result, *, q: float = 0.99, threshold: float | None = None) -> dict[str, dict[str, float]]:
    """Per-component empirical VaR/TVaR from a risksim result's component losses.

    Reads the ``(n_sims, n_components)`` array ``result.component_losses``
    (names from ``component_names`` when present) and returns, per
    component, the empirical VaR and TVaR at level ``q``, plus the
    exceedance probability of ``threshold`` when one is given. These are
    marginal, per-component metrics -- diversified totals belong to the
    portfolio-level summary.

    Returns
    -------
    dict of str -> dict of str -> float
        ``{name: {"var", "tvar"[, "exceedance_probability"]}}``.
    """
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
    for name, col in zip(names, values.T, strict=True):
        metrics = {
            "var": float(empirical_var(col, q)),
            "tvar": float(empirical_tvar(col, q)),
        }
        if threshold is not None:
            metrics["exceedance_probability"] = float(exceedance_probability(col, threshold))
        out[str(name)] = metrics
    return out


def layer_tail_metrics(result, *, q: float = 0.99, threshold: float | None = None) -> dict[str, dict[str, float]]:
    """Per-layer empirical VaR/TVaR from a risksim result's layer losses.

    The layer analogue of :func:`component_tail_metrics`: reads the
    ``(n_sims, n_layers)`` array ``result.layer_losses`` (names from
    ``layer_names`` when present) and returns, per layer, the empirical
    VaR and TVaR at level ``q``, plus the exceedance probability of
    ``threshold`` when one is given.

    Returns
    -------
    dict of str -> dict of str -> float
        ``{name: {"var", "tvar"[, "exceedance_probability"]}}``.
    """
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
    for name, col in zip(names, values.T, strict=True):
        metrics = {
            "var": float(empirical_var(col, q)),
            "tvar": float(empirical_tvar(col, q)),
        }
        if threshold is not None:
            metrics["exceedance_probability"] = float(exceedance_probability(col, threshold))
        out[str(name)] = metrics
    return out


def _require_spliced_severity():
    """Lazily import lossmodels.SplicedSeverity (optional dependency)."""
    try:
        from lossmodels.severity import SplicedSeverity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "this function builds a spliced severity and requires the "
            "'lossmodels' package. Install it with `pip install lossmodels`."
        ) from exc
    return SplicedSeverity


def splice_gpd_tail(body, fit, *, weight: float | None = None):
    """Splice an already-fitted GPD tail (a :class:`GPDFit`) onto ``body``.

    Returns a ``lossmodels.SplicedSeverity`` whose body is ``body`` (any fitted
    body severity) and whose tail is the conditional GPD of ``fit`` above its
    threshold. The mixing weight defaults to the body mass implied by the fit,
    ``1 - fit.exceedance_fraction`` (i.e. ``P(X <= threshold)``).
    """
    SplicedSeverity = _require_spliced_severity()
    tail = GPDTail.from_fit(fit)
    if weight is None:
        weight = float(1.0 - fit.exceedance_fraction)
    return SplicedSeverity(body=body, tail=tail, threshold=float(fit.threshold), weight=weight)


def fit_spliced_gpd(body, data, *, threshold: float, weight: float | None = None):
    """Fit a GPD tail above ``threshold`` (peaks-over-threshold) and splice it
    onto ``body``, returning a ``lossmodels.SplicedSeverity``.

    Parameters
    ----------
    body : severity model
        Any fitted body severity (e.g. a ``lossmodels`` ``Lognormal``).
    data : array-like
        Loss sample used to fit the tail and, by default, to set the body mass.
    threshold : float
        Peaks-over-threshold cutoff ``u``.
    weight : float, optional
        Body mass ``P(X <= u)``. Defaults to ``1 - exceedance_fraction`` from the
        POT fit (the empirical fraction at or below the threshold), consistent
        with the fitted exceedance rate.

    Requires the ``lossmodels`` package.
    """
    SplicedSeverity = _require_spliced_severity()
    losses = as_1d_float_array(data, name="data")
    fit = fit_pot(losses, threshold=threshold)
    tail = GPDTail.from_fit(fit)
    if weight is None:
        weight = float(1.0 - fit.exceedance_fraction)
    return SplicedSeverity(body=body, tail=tail, threshold=float(fit.threshold), weight=weight)

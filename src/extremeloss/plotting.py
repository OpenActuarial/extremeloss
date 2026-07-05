from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as _exc:  # pragma: no cover - exercised only without matplotlib
    raise ModuleNotFoundError(
        "extremeloss plotting requires matplotlib, which is an optional dependency. "
        "Install it with:  pip install \"extremeloss[plot]\""
    ) from _exc

import numpy as np

from .estimation.metrics import exceedance_curve
from .evt.tail_index import hill_curve
from .evt.thresholds import mean_excess
from .utils.validation import as_1d_float_array


def plot_exceedance_curve(losses, thresholds, ax=None):
    curve = exceedance_curve(losses, thresholds)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["thresholds"], curve["probabilities"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("P(X > u)")
    ax.set_title("Exceedance Curve")
    return ax


def plot_mean_excess(losses, thresholds, ax=None):
    curve = mean_excess(losses, thresholds)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["thresholds"], curve["mean_excess"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")
    ax.set_title("Mean Excess Plot")
    return ax


def plot_hill_curve(losses, k_grid=None, ax=None):
    arr = as_1d_float_array(losses, name="losses")
    curve = hill_curve(arr, k_grid=k_grid)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["k"], curve["hill"])
    ax.set_xlabel("k")
    ax.set_ylabel("Hill estimate")
    ax.set_title("Hill Plot")
    return ax


def _diag_panels(fit, data, rl, empirical_T, empirical_levels, pdf_x, pdf_y,
                 sample, title):
    from .evt.diagnostics import pp_points, qq_points

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pp = pp_points(fit, data)
    axes[0, 0].plot(pp["empirical"], pp["model"], "o", ms=3)
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set(title="Probability plot", xlabel="empirical",
                   ylabel="model")
    qq = qq_points(fit, data)
    axes[0, 1].plot(qq["theoretical"], qq["empirical"], "o", ms=3)
    lo = min(qq["theoretical"][0], qq["empirical"][0])
    hi = max(qq["theoretical"][-1], qq["empirical"][-1])
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0, 1].set(title="Quantile plot", xlabel="model", ylabel="empirical")
    axes[1, 0].plot(rl["return_period"], rl["return_level"], "-", lw=1.5)
    if not np.all(np.isnan(rl["se"])):
        axes[1, 0].fill_between(rl["return_period"], rl["ci_low"],
                                rl["ci_high"], alpha=0.25)
    axes[1, 0].plot(empirical_T, empirical_levels, "o", ms=3)
    axes[1, 0].set(xscale="log", title="Return level plot",
                   xlabel="return period", ylabel="return level")
    axes[1, 1].hist(sample, bins="auto", density=True, alpha=0.5)
    axes[1, 1].plot(pdf_x, pdf_y, lw=1.5)
    axes[1, 1].set(title="Density", xlabel="loss", ylabel="density")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_gpd_diagnostics(fit, data, observations_per_period: float = 1.0,
                         return_periods=None):
    """Coles-style four-panel diagnostics for a POT/GPD fit.

    Probability plot, quantile plot, return-level plot with the
    delta-method band and empirical points, and the fitted conditional
    density over the exceedance histogram. Requires matplotlib (the
    ``plot`` extra); the numerical content is available headlessly from
    :mod:`extremeloss.evt.diagnostics`. Returns the figure.
    """
    from scipy.stats import genpareto

    from .evt.gpd import gpd_return_level

    x = as_1d_float_array(data)
    exc = np.sort(x[x > fit.threshold] - fit.threshold)
    if return_periods is None:
        t_max = 2.0 * x.size / (observations_per_period
                                * max(fit.n_exceedances, 1)
                                * fit.exceedance_fraction)
        return_periods = np.geomspace(
            1.05 / (observations_per_period * fit.exceedance_fraction),
            max(t_max, 10.0), 60)
    rl = gpd_return_level(fit, return_periods,
                          observations_per_period=observations_per_period)
    n_total = int(round(fit.n_exceedances / fit.exceedance_fraction))
    k = np.arange(1, exc.size + 1)
    emp_T = 1.0 / (observations_per_period * (k[::-1] - 0.5) / n_total)
    emp_levels = fit.threshold + exc
    pdf_x = np.linspace(0.0, exc[-1], 200)
    pdf_y = genpareto.pdf(pdf_x, c=fit.xi, scale=fit.beta)
    return _diag_panels(fit, data, rl, emp_T, emp_levels,
                        fit.threshold + pdf_x, pdf_y, fit.threshold + exc,
                        "GPD fit diagnostics")


def plot_gev_diagnostics(fit, block_maxima, return_periods=None):
    """Coles-style four-panel diagnostics for a block-maxima GEV fit.

    Same panels as :func:`plot_gpd_diagnostics`; return periods are in
    blocks. Returns the figure.
    """
    from scipy.stats import genextreme

    from .evt.block_maxima import gev_return_level

    x = np.sort(as_1d_float_array(block_maxima))
    if return_periods is None:
        return_periods = np.geomspace(1.1, 4.0 * x.size, 60)
    rl = gev_return_level(fit, return_periods)
    k = np.arange(1, x.size + 1)
    emp_T = 1.0 / (1.0 - (k - 0.5) / x.size)
    pdf_x = np.linspace(x[0], x[-1], 200)
    pdf_y = genextreme.pdf(pdf_x, c=-fit.xi, loc=fit.loc, scale=fit.scale)
    return _diag_panels(fit, block_maxima, rl, emp_T, x, pdf_x, pdf_y, x,
                        "GEV fit diagnostics")

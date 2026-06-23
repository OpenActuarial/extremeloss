from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import genextreme, genpareto


def _scalarize(result, x):
    """Return a float for scalar input and a float ndarray for array input."""
    if np.ndim(x) == 0:
        return float(result)
    return np.asarray(result, dtype=float)


@dataclass(slots=True)
class TailEstimateResult:
    """Container for tail-estimation results."""

    estimate: float
    method: str
    stderr: float | None = None
    ci: tuple[float, float] | None = None
    n: int | None = None
    effective_n: float | None = None
    threshold: float | None = None
    quantile: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "estimate": float(self.estimate),
            "method": self.method,
        }
        if self.stderr is not None:
            out["stderr"] = float(self.stderr)
        if self.ci is not None:
            out["ci"] = (float(self.ci[0]), float(self.ci[1]))
        if self.n is not None:
            out["n"] = int(self.n)
        if self.effective_n is not None:
            out["effective_n"] = float(self.effective_n)
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        if self.quantile is not None:
            out["quantile"] = float(self.quantile)
        if self.diagnostics:
            out["diagnostics"] = dict(self.diagnostics)
        return out


@dataclass(slots=True)
class GPDFit:
    """Fitted generalized Pareto distribution above a threshold."""

    threshold: float
    xi: float
    beta: float
    exceedance_fraction: float
    n_exceedances: int
    fit_method: str = "mle"
    covariance: np.ndarray | None = None

    def tail_probability(self, x: float) -> float:
        from .evt.gpd import gpd_tail_probability

        return gpd_tail_probability(
            x=x,
            threshold=self.threshold,
            xi=self.xi,
            beta=self.beta,
            exceedance_fraction=self.exceedance_fraction,
        )

    def var(self, p: float) -> float:
        from .evt.gpd import gpd_var

        return gpd_var(
            p=p,
            threshold=self.threshold,
            xi=self.xi,
            beta=self.beta,
            exceedance_fraction=self.exceedance_fraction,
        )

    def tvar(self, p: float) -> float:
        from .evt.gpd import gpd_tvar

        return gpd_tvar(
            p=p,
            threshold=self.threshold,
            xi=self.xi,
            beta=self.beta,
            exceedance_fraction=self.exceedance_fraction,
        )

    def return_level(self, period: float) -> float:
        if period <= 1.0:
            raise ValueError("period must exceed 1.0")
        return self.var(1.0 - 1.0 / period)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "threshold": float(self.threshold),
            "xi": float(self.xi),
            "beta": float(self.beta),
            "exceedance_fraction": float(self.exceedance_fraction),
            "n_exceedances": int(self.n_exceedances),
            "fit_method": self.fit_method,
        }
        if self.covariance is not None:
            out["covariance"] = np.asarray(self.covariance, dtype=float)
        return out


@dataclass(slots=True)
class GEVFit:
    """Fitted generalized extreme value distribution for block maxima."""

    xi: float
    loc: float
    scale: float
    n_blocks: int
    block_size: int | None = None
    fit_method: str = "mle"
    covariance: np.ndarray | None = None

    def return_level(self, period: float) -> float:
        if period <= 1.0:
            raise ValueError("period must exceed 1.0")
        p = 1.0 - 1.0 / period
        return float(genextreme.ppf(p, c=-self.xi, loc=self.loc, scale=self.scale))

    def cdf(self, x: float) -> float:
        return float(genextreme.cdf(x, c=-self.xi, loc=self.loc, scale=self.scale))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "xi": float(self.xi),
            "loc": float(self.loc),
            "scale": float(self.scale),
            "n_blocks": int(self.n_blocks),
            "fit_method": self.fit_method,
        }
        if self.block_size is not None:
            out["block_size"] = int(self.block_size)
        if self.covariance is not None:
            out["covariance"] = np.asarray(self.covariance, dtype=float)
        return out


@dataclass(slots=True)
class BootstrapResult:
    """Bootstrap uncertainty summary for a scalar statistic."""

    estimate: float
    bootstrap_estimates: np.ndarray
    method: str = "percentile"
    ci: tuple[float, float] | None = None
    stderr: float | None = None
    alpha: float | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "estimate": float(self.estimate),
            "method": self.method,
            "n_bootstrap": int(self.bootstrap_estimates.size),
        }
        if self.stderr is not None:
            out["stderr"] = float(self.stderr)
        if self.ci is not None:
            out["ci"] = (float(self.ci[0]), float(self.ci[1]))
        if self.alpha is not None:
            out["alpha"] = float(self.alpha)
        return out


@dataclass(slots=True)
class ThresholdScan:
    """Threshold-diagnostic results across a grid of thresholds."""

    thresholds: np.ndarray
    mean_excess: np.ndarray
    xi: np.ndarray
    beta: np.ndarray
    n_exceedances: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "thresholds": np.asarray(self.thresholds, dtype=float),
            "mean_excess": np.asarray(self.mean_excess, dtype=float),
            "xi": np.asarray(self.xi, dtype=float),
            "beta": np.asarray(self.beta, dtype=float),
            "n_exceedances": np.asarray(self.n_exceedances, dtype=int),
        }

@dataclass(slots=True)
class GPDTail:
    """Conditional generalized-Pareto tail on ``[threshold, inf)``, for splicing.

    A thin distribution object wrapping ``scipy.stats.genpareto`` with shape
    ``xi``, scale ``beta`` and location ``threshold``. Unlike :class:`GPDFit`
    (which carries the exceedance rate and reports *unconditional* VaR/TVaR),
    this represents the tail *conditional* on ``X > threshold``: ``cdf(threshold)
    == 0`` and the density integrates to one over ``[threshold, inf)``. It
    exposes the ``cdf`` / ``pdf`` / ``sample`` / ``quantile`` / ``mean`` /
    ``variance`` interface that :class:`lossmodels.SplicedSeverity` consumes as a
    tail. Moments raise when they do not exist (``xi >= 1`` for the mean,
    ``xi >= 1/2`` for the variance), mirroring heavy-tailed severities.
    """

    threshold: float
    xi: float
    beta: float

    @classmethod
    def from_fit(cls, fit: "GPDFit") -> "GPDTail":
        """Build a conditional tail from a fitted :class:`GPDFit`."""
        return cls(threshold=float(fit.threshold), xi=float(fit.xi), beta=float(fit.beta))

    def _frozen(self):
        return genpareto(c=self.xi, loc=self.threshold, scale=self.beta)

    def pdf(self, x):
        return _scalarize(self._frozen().pdf(np.asarray(x, dtype=float)), x)

    def cdf(self, x):
        return _scalarize(self._frozen().cdf(np.asarray(x, dtype=float)), x)

    def quantile(self, p):
        return _scalarize(self._frozen().ppf(np.asarray(p, dtype=float)), p)

    def ppf(self, p):
        return self.quantile(p)

    def sample(self, size: int = 1) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return genpareto.rvs(c=self.xi, loc=self.threshold, scale=self.beta, size=size)

    def mean(self) -> float:
        if self.xi >= 1.0:
            raise ValueError("mean does not exist for xi >= 1.")
        return float(self.threshold + self.beta / (1.0 - self.xi))

    def variance(self) -> float:
        if self.xi >= 0.5:
            raise ValueError("variance does not exist for xi >= 1/2.")
        return float(self.beta ** 2 / ((1.0 - self.xi) ** 2 * (1.0 - 2.0 * self.xi)))

    def summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold), "xi": float(self.xi), "beta": float(self.beta)}

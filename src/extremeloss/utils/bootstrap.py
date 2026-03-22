from __future__ import annotations

import numpy as np

from ..estimation.metrics import empirical_tvar, empirical_var, exceedance_probability
from ..results import BootstrapResult
from .validation import as_1d_float_array, validate_alpha, validate_positive


def bootstrap_statistic(
    data,
    statistic,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    validate_positive(n_resamples, name="n_resamples")
    validate_alpha(alpha)
    x = as_1d_float_array(data, name="data")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    point = float(statistic(x))
    boot = np.empty(int(n_resamples), dtype=float)
    n = x.size
    for i in range(int(n_resamples)):
        sample = x[rng.integers(0, n, size=n)]
        boot[i] = float(statistic(sample))
    ci = tuple(np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())
    stderr = float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0
    return BootstrapResult(
        estimate=point,
        bootstrap_estimates=boot,
        method="percentile",
        ci=(float(ci[0]), float(ci[1])),
        stderr=stderr,
        alpha=float(alpha),
    )


def bootstrap_tail_probability(losses, threshold: float, **kwargs) -> BootstrapResult:
    return bootstrap_statistic(losses, lambda x: exceedance_probability(x, threshold), **kwargs)


def bootstrap_var(losses, q: float, **kwargs) -> BootstrapResult:
    return bootstrap_statistic(losses, lambda x: empirical_var(x, q), **kwargs)


def bootstrap_tvar(losses, q: float, **kwargs) -> BootstrapResult:
    return bootstrap_statistic(losses, lambda x: empirical_tvar(x, q), **kwargs)

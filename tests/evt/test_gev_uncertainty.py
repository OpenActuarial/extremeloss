"""GEV parameter uncertainty and return-level intervals."""
import numpy as np
import pytest
from scipy.stats import genextreme

import extremeloss as el
from extremeloss.evt.block_maxima import fit_gev


@pytest.fixture(scope="module")
def gev_fit():
    rng = np.random.default_rng(6)
    xi_t, loc_t, scale_t = 0.15, 100.0, 20.0
    maxima = genextreme.rvs(c=-xi_t, loc=loc_t, scale=scale_t, size=600,
                            random_state=rng)
    return fit_gev(maxima), (xi_t, loc_t, scale_t)


def test_covariance_populated_truth_within_bands(gev_fit):
    fit, (xi_t, loc_t, scale_t) = gev_fit
    assert fit.covariance is not None and fit.covariance.shape == (3, 3)
    se = fit.se
    assert se is not None and np.all(se > 0)
    assert abs(fit.xi - xi_t) < 3 * se[0]
    assert abs(fit.loc - loc_t) < 3 * se[1]
    assert abs(fit.scale - scale_t) < 3 * se[2]


def test_return_level_matches_method_and_inverts_cdf(gev_fit):
    fit, _ = gev_fit
    periods = np.array([10.0, 50.0, 200.0])
    out = el.gev_return_level(fit, periods)
    for T, r in zip(periods, out["return_level"], strict=True):
        assert r == pytest.approx(fit.return_level(T), rel=1e-9)
        assert fit.cdf(r) == pytest.approx(1.0 - 1.0 / T, rel=1e-9)
    assert np.all(np.diff(out["return_level"]) > 0)
    assert np.all(out["ci_low"] < out["return_level"])
    assert np.all(out["return_level"] < out["ci_high"])
    assert np.all(np.diff(out["se"]) > 0)


def test_xi_zero_branch_and_no_covariance():
    from extremeloss.results import GEVFit

    fit = GEVFit(xi=0.0, loc=50.0, scale=10.0, n_blocks=100)
    out = el.gev_return_level(fit, 100.0)
    y = -np.log(1.0 - 1.0 / 100.0)
    assert out["return_level"][0] == pytest.approx(50.0 - 10.0 * np.log(y),
                                                   rel=1e-12)
    assert np.isnan(out["se"][0])


def test_guards(gev_fit):
    fit, _ = gev_fit
    with pytest.raises(ValueError, match="exceed 1"):
        el.gev_return_level(fit, 1.0)
    with pytest.raises(ValueError, match="confidence_level"):
        el.gev_return_level(fit, 50.0, confidence_level=0.0)


def test_return_level_ci_coverage():
    """Empirical coverage at nominal 0.95: same honest-Wald story as the
    GPD test -- expect modest undercoverage from the skewed sampling
    distribution of a long-horizon quantile."""
    rng = np.random.default_rng(1)
    xi_t, loc_t, scale_t = 0.1, 100.0, 20.0
    T = 100.0
    y = -np.log(1.0 - 1.0 / T)
    r_true = loc_t + (scale_t / xi_t) * (y**(-xi_t) - 1.0)
    reps, hits, usable = 120, 0, 0
    for _ in range(reps):
        maxima = genextreme.rvs(c=-xi_t, loc=loc_t, scale=scale_t, size=300,
                                random_state=rng)
        fit = fit_gev(maxima)
        if fit.covariance is None:
            continue
        usable += 1
        out = el.gev_return_level(fit, T, confidence_level=0.95)
        hits += out["ci_low"][0] <= r_true <= out["ci_high"][0]
    assert usable >= reps * 0.9
    cov = hits / usable
    assert 0.85 <= cov <= 0.99, f"coverage {cov:.3f}"

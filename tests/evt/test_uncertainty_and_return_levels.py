"""GPD parameter uncertainty, return levels with CIs, threshold-scan bands."""
import numpy as np
import pytest
from scipy.stats import genpareto

import extremeloss as el


@pytest.fixture(scope="module")
def pot_fit():
    rng = np.random.default_rng(42)
    xi_true, beta_true, u = 0.2, 1000.0, 5000.0
    body = rng.uniform(0.0, u, 20_000)
    n_exc = 5_000
    tail = u + genpareto.rvs(c=xi_true, scale=beta_true, size=n_exc, random_state=rng)
    data = np.concatenate([body, tail])
    return el.fit_pot(data, threshold=u), xi_true, beta_true


def test_covariance_populated_and_matches_asymptotics(pot_fit):
    fit, xi_true, beta_true = pot_fit
    assert fit.covariance is not None and fit.covariance.shape == (2, 2)
    n = fit.n_exceedances
    xi, beta = fit.xi, fit.beta
    # Fisher information for the GPD: avar(xi) = (1+xi)^2 / n,
    # avar(beta) = 2 beta^2 (1+xi) / n, acov = -beta (1+xi) / n
    np.testing.assert_allclose(fit.covariance[0, 0], (1 + xi) ** 2 / n, rtol=0.15)
    np.testing.assert_allclose(
        fit.covariance[1, 1], 2 * beta**2 * (1 + xi) / n, rtol=0.15
    )
    np.testing.assert_allclose(fit.covariance[0, 1], -beta * (1 + xi) / n, rtol=0.25)
    se = fit.se
    assert se is not None and np.all(se > 0)
    np.testing.assert_allclose(se, np.sqrt(np.diag(fit.covariance)), rtol=1e-12)
    # truth inside 3-sigma
    assert abs(xi - xi_true) < 3 * se[0]
    assert abs(beta - beta_true) < 3 * se[1]


def test_return_level_consistent_with_method_and_inverts_tail(pot_fit):
    fit, _, _ = pot_fit
    obs_per_period = 1000.0
    periods = np.array([10.0, 50.0, 200.0])
    out = el.gpd_return_level(fit, periods, observations_per_period=obs_per_period)
    # exact agreement with the existing single-point method
    for T, r in zip(periods, out["return_level"]):
        assert r == pytest.approx(fit.return_level(T * obs_per_period), rel=1e-9)
        # and the level inverts the unconditional tail exactly
        assert fit.tail_probability(r) * T * obs_per_period == pytest.approx(
            1.0, rel=1e-9
        )
    # monotone in T, CIs bracket with positive width
    assert np.all(np.diff(out["return_level"]) > 0)
    assert np.all(out["ci_low"] < out["return_level"])
    assert np.all(out["return_level"] < out["ci_high"])
    assert np.all(np.diff(out["se"]) > 0)  # longer horizons are less certain


def test_return_level_xi_zero_branch():
    fit = el.GPDFit(
        threshold=100.0, xi=0.0, beta=50.0,
        exceedance_fraction=0.05, n_exceedances=500,
    )
    out = el.gpd_return_level(fit, 100.0, observations_per_period=100.0)
    expected = 100.0 + 50.0 * np.log(100.0 * 100.0 * 0.05)
    assert out["return_level"][0] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(out["se"][0])  # no covariance -> nan bands, not garbage


def test_return_level_guards(pot_fit):
    fit, _, _ = pot_fit
    with pytest.raises(ValueError, match="fewer than one expected exceedance"):
        el.gpd_return_level(fit, 1.0, observations_per_period=1.0)
    with pytest.raises(ValueError, match="positive"):
        el.gpd_return_level(fit, -5.0)
    with pytest.raises(ValueError, match="confidence_level"):
        el.gpd_return_level(fit, 50.0, observations_per_period=1000.0,
                            confidence_level=1.5)


def test_threshold_scan_bands_flat_on_true_gpd():
    rng = np.random.default_rng(7)
    xi_true, beta_true = 0.15, 2000.0
    data = genpareto.rvs(c=xi_true, scale=beta_true, size=30_000, random_state=rng)
    grid = np.quantile(data, [0.5, 0.6, 0.7, 0.8, 0.9])
    scan = el.threshold_diagnostic_table(data, grid)
    assert scan.xi_se is not None and scan.modified_scale is not None
    assert np.all(np.isfinite(scan.xi_se))
    # on true-GPD data the modified scale is flat: every estimate within
    # 3 se of the whole-sample fit's beta* (= beta_true here, u anchored at 0)
    resid = np.abs(scan.modified_scale - beta_true) / scan.modified_scale_se
    assert np.all(resid < 3.0)
    d = scan.to_dict()
    assert set(d) >= {"xi_se", "modified_scale", "modified_scale_se"}


def test_gpd_mean_excess_closed_form_and_guards(pot_fit):
    fit, _, _ = pot_fit
    u, xi, beta = fit.threshold, fit.xi, fit.beta
    assert fit.mean_excess(u) == pytest.approx(beta / (1 - xi), rel=1e-12)
    d = u + 3000.0
    assert fit.mean_excess(d) == pytest.approx(
        (beta + xi * 3000.0) / (1 - xi), rel=1e-12
    )
    with pytest.raises(ValueError, match="at or above the threshold"):
        fit.mean_excess(u - 1.0)
    heavy = el.GPDFit(threshold=0.0, xi=1.2, beta=1.0,
                      exceedance_fraction=1.0, n_exceedances=100)
    assert np.isinf(heavy.mean_excess(0.0))


def test_sf_alias(pot_fit):
    fit, _, _ = pot_fit
    x = fit.threshold + 2500.0
    assert fit.sf(x) == fit.tail_probability(x)


def test_thin_threshold_rows_are_nan_across_new_columns():
    rng = np.random.default_rng(4)
    data = rng.exponential(1000.0, 500)
    grid = np.array([500.0, float(data.max()) + 1.0])
    scan = el.threshold_diagnostic_table(data, grid)
    assert scan.n_exceedances[1] == 0
    for col in (scan.xi, scan.xi_se, scan.modified_scale, scan.modified_scale_se):
        assert np.isnan(col[1])
    assert np.isfinite(scan.modified_scale[0])


def test_scalar_return_period_gives_length_one_arrays(pot_fit):
    fit, _, _ = pot_fit
    out = el.gpd_return_level(fit, 50.0, observations_per_period=1000.0)
    for key in ("return_period", "return_level", "se", "ci_low", "ci_high"):
        assert np.shape(out[key]) == (1,)

"""Regression: the tail-pricing worked-example page numbers stay true."""
import numpy as np
import pytest

import lossmodels as lm

import extremeloss as el


@pytest.fixture(scope="module")
def page_run():
    rng = np.random.default_rng(2026)
    claims = rng.lognormal(mean=9.4, sigma=1.15, size=2_400)
    best = lm.fit_lognormal(claims)
    u = float(np.quantile(claims, 0.90))
    fit = el.fit_pot(claims, threshold=u)
    return claims, best, u, fit


def test_tail_page_model_contest(page_run):
    claims, best, _, _ = page_run
    tab = lm.compare_fits({"lognormal": best, "gamma": lm.fit_gamma(claims),
                           "weibull": lm.fit_weibull(claims)}, claims)
    assert round(tab.loc["lognormal", "aic"], 3) == 52_458.394
    assert round(tab.loc["gamma", "aic"], 3) == 52_930.563
    assert round(tab.loc["lognormal", "ks"], 3) == 0.009
    assert round(tab.loc["gamma", "ad"], 3) == 36.741
    assert round(best.mu, 4) == 9.3652
    assert round(best.sigma, 4) == 1.1547


def test_tail_page_uncertainty_and_ilf(page_run):
    claims, best, _, _ = page_run
    unc = lm.fit_uncertainty(best, claims)
    summ = unc.summary()
    assert round(summ.loc["mu", "se"], 4) == 0.0236
    assert round(summ.loc["sigma", "se"], 4) == 0.0167
    ilf = lm.increased_limits_table(best, limits=[250_000, 500_000, 1_000_000],
                                    base_limit=250_000, uncertainty=unc)
    assert round(ilf.loc[500_000, "ilf"], 4) == 1.0182
    assert round(ilf.loc[500_000, "ilf_se"], 4) == 0.0023
    assert round(ilf.loc[1_000_000, "ci_high"], 4) == 1.0288


def test_tail_page_scan_fit_and_return_levels(page_run):
    claims, _, u, fit = page_run
    grid = np.quantile(claims, [0.85, 0.90, 0.93, 0.95])
    scan = el.threshold_diagnostic_table(claims, grid)
    np.testing.assert_allclose(scan.xi.round(3), [0.365, 0.450, 0.403, 0.380])
    np.testing.assert_allclose(scan.xi_se.round(3), [0.070, 0.097, 0.113, 0.133])
    assert round(u, 0) == 51_810
    assert round(fit.xi, 4) == 0.4502
    assert round(fit.beta, 0) == 29_152
    rl = el.gpd_return_level(fit, [10, 50], observations_per_period=2_400.0)
    assert round(rl["return_level"][0], 0) == 2_140_226
    assert round(rl["se"][0], 0) == 1_051_790
    assert round(rl["return_level"][1], 0) == 4_430_985


def test_tail_page_pooling_both_ways(page_run):
    rm = pytest.importorskip("ratingmodels")
    claims, best, _, fit = page_run
    body = rm.pooling_charge_from_severity(best, pooling_point=250_000.0,
                                           expected_frequency=0.9)
    tail = rm.pooling_charge_from_severity(fit, pooling_point=250_000.0,
                                           expected_frequency=0.9)
    assert round(body["pooling_charge"], 2) == 474.28
    assert round(tail["pooling_charge"], 2) == 862.02
    assert round(body["mean_excess"], 0) == 132_302
    assert round(tail["mean_excess"], 0) == 215_323

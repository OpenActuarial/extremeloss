"""Seeded parameter-recovery checks for the tail estimators."""
import numpy as np
import pytest

from extremeloss import fit_gpd, fit_pot, hill_estimator, threshold_diagnostic_table


def _pareto_type1(alpha, n, seed):
    u = np.random.default_rng(seed).uniform(size=n)
    return 100.0 * (1 - u) ** (-1 / alpha)


@pytest.mark.parametrize("alpha", [2.5, 4.0])
def test_hill_estimates_xi_on_exact_pareto(alpha):
    data = _pareto_type1(alpha, 20_000, seed=1)
    assert hill_estimator(data, k=1000) == pytest.approx(1 / alpha, abs=0.05)


def _synthetic(seed=2):
    from scipy.stats import genpareto
    rng = np.random.default_rng(seed)
    body = rng.uniform(0.0, 1000.0, size=5000)
    excesses = genpareto.rvs(c=0.2, scale=500.0, size=5000, random_state=rng)
    return np.concatenate([body, 1000.0 + excesses])


def test_fit_pot_recovers_the_generating_parameters():
    fit = fit_pot(_synthetic(), threshold=1000.0)
    assert fit.n_exceedances == 5000
    assert fit.exceedance_fraction == pytest.approx(0.5, rel=1e-12)
    assert fit.xi == pytest.approx(0.2, abs=0.04)
    assert fit.beta == pytest.approx(500.0, rel=0.10)


def test_fit_gpd_on_excesses_equals_fit_pot():
    data = _synthetic()
    pot = fit_pot(data, threshold=1000.0)
    direct = fit_gpd(data[data > 1000.0] - 1000.0)
    assert direct.xi == pytest.approx(pot.xi, rel=1e-8)
    assert direct.beta == pytest.approx(pot.beta, rel=1e-8)


def test_threshold_scan_is_consistent_with_fit_pot():
    data = _synthetic()
    scan = threshold_diagnostic_table(data, [1000.0])
    pot = fit_pot(data, threshold=1000.0)
    assert scan.xi[0] == pytest.approx(pot.xi, rel=1e-12)
    assert scan.beta[0] == pytest.approx(pot.beta, rel=1e-12)
    assert scan.n_exceedances[0] == pot.n_exceedances

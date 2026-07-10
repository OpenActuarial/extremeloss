"""fit_gpd fits pure excesses (exceedance_fraction == 1); its risk measures
must be usable and equal the plain GPD quantities."""
import pytest
from scipy.stats import genpareto

from extremeloss.evt.gpd import fit_gpd, gpd_tvar, gpd_var


def test_full_distribution_var_matches_scipy_ppf():
    for xi in (-0.2, 0.0, 0.4):
        for p in (0.9, 0.99):
            v = gpd_var(p, threshold=0.0, xi=xi, beta=50.0, exceedance_fraction=1.0)
            assert v == pytest.approx(genpareto.ppf(p, c=xi, scale=50.0), rel=1e-9)


def test_fit_gpd_result_supports_risk_measures():
    exc = genpareto.rvs(c=0.25, scale=40.0, size=4000, random_state=7)
    fit = fit_gpd(exc)
    assert fit.exceedance_fraction == 1.0
    v = fit.var(0.99)
    assert v == pytest.approx(genpareto.ppf(0.99, c=fit.xi, scale=fit.beta), rel=1e-9)
    assert fit.tvar(0.99) >= v
    assert fit.return_level(100.0) == pytest.approx(fit.var(0.99), rel=1e-12)


def test_exceedance_fraction_above_one_still_rejected():
    with pytest.raises(ValueError):
        gpd_var(0.99, 0.0, 0.2, 50.0, exceedance_fraction=1.5)


def test_tvar_matches_average_quantile_integral():
    from scipy.integrate import quad
    xi, beta, zeta, p = 0.3, 50.0, 0.4, 0.99
    es, _ = quad(lambda s: gpd_var(s, 100.0, xi, beta, zeta), p, 1.0 - 1e-12, limit=400)
    assert gpd_tvar(p, 100.0, xi, beta, zeta) == pytest.approx(es / (1.0 - p), rel=1e-5)

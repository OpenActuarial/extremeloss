"""Fitted-model diagnostics: QQ/PP geometry, honest bootstrap p-values."""
import numpy as np
import pytest
from scipy.stats import genextreme, genpareto

import extremeloss as el
from extremeloss.evt.block_maxima import fit_gev


@pytest.fixture(scope="module")
def good_gpd():
    rng = np.random.default_rng(2)
    u = 1000.0
    data = np.concatenate([
        rng.uniform(0, u, 8000),
        u + genpareto.rvs(c=0.2, scale=300.0, size=900, random_state=rng),
    ])
    return el.fit_pot(data, threshold=u), data


def test_qq_pp_geometry_on_true_model(good_gpd):
    fit, data = good_gpd
    qq = el.qq_points(fit, data)
    assert qq["n"] == fit.n_exceedances
    # both axes are sorted, so rank correlation is 1 by construction and
    # Pearson is dominated by the wildly variable top order statistics of
    # a heavy tail; the meaningful geometry check is the BODY hugging the
    # 45-degree line
    n = qq["n"]
    body = slice(n // 4, 3 * n // 4)
    rel_dev = np.abs(qq["empirical"][body] - qq["theoretical"][body]) / qq[
        "theoretical"][body]
    assert np.max(rel_dev) < 0.10
    assert np.corrcoef(qq["theoretical"], qq["empirical"])[0, 1] > 0.95
    assert qq["empirical"][0] >= fit.threshold  # original loss units
    pp = el.pp_points(fit, data)
    assert np.all((0 <= pp["model"]) & (pp["model"] <= 1))
    assert np.max(np.abs(pp["model"] - pp["empirical"])) < 0.05


def test_gof_accepts_true_model(good_gpd):
    fit, data = good_gpd
    out = el.parametric_bootstrap_gof(fit, data, n_boot=149, rng=1)
    # under the true model the p-value is Uniform(0,1): a fixed p > 0.05
    # check fails 5% of seeds by construction, so demand only non-tiny
    assert out["ks_pvalue"] > 0.01
    assert out["ad_pvalue"] > 0.01
    assert out["n"] == fit.n_exceedances and out["n_boot"] == 149


def test_gof_pvalues_are_calibrated_not_anticonservative():
    """Mean p-value over independent true-model datasets should sit near
    0.5, not near 0 -- the cheap guard that the refit-per-replicate
    bootstrap is actually calibrated."""
    rng = np.random.default_rng(11)
    ps = []
    for _ in range(5):
        exc = genpareto.rvs(c=0.2, scale=300.0, size=600, random_state=rng)
        data = np.concatenate([rng.uniform(0, 1000.0, 3000), 1000.0 + exc])
        fit = el.fit_pot(data, threshold=1000.0)
        ps.append(el.parametric_bootstrap_gof(fit, data, n_boot=49,
                                              rng=rng)["ks_pvalue"])
    assert np.mean(ps) > 0.15, ps


def test_gof_rejects_planted_misfit():
    rng = np.random.default_rng(3)
    u = 1000.0
    # bimodal exceedances: nothing a single GPD can describe
    exc = np.concatenate([rng.normal(200.0, 10.0, 700),
                          rng.normal(900.0, 10.0, 700)])
    data = np.concatenate([rng.uniform(0, u, 5000), u + np.abs(exc)])
    fit = el.fit_pot(data, threshold=u)
    out = el.parametric_bootstrap_gof(fit, data, n_boot=199, rng=2)
    assert out["ks_pvalue"] < 0.05
    assert out["ad_pvalue"] < 0.05


def test_gev_dispatch_and_guards():
    rng = np.random.default_rng(4)
    maxima = genextreme.rvs(c=-0.1, loc=50.0, scale=8.0, size=250,
                            random_state=rng)
    fit = fit_gev(maxima)
    qq = el.qq_points(fit, maxima)
    assert np.corrcoef(qq["theoretical"], qq["empirical"])[0, 1] > 0.99
    out = el.parametric_bootstrap_gof(fit, maxima, n_boot=99, rng=5)
    assert out["ks_pvalue"] > 0.05
    with pytest.raises(TypeError, match="GPDFit or GEVFit"):
        el.qq_points(object(), maxima)
    with pytest.raises(ValueError, match="n_boot"):
        el.parametric_bootstrap_gof(fit, maxima, n_boot=5)


def test_diagnostic_plots_render():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    rng = np.random.default_rng(6)
    u = 500.0
    data = np.concatenate([
        rng.uniform(0, u, 3000),
        u + genpareto.rvs(c=0.15, scale=200.0, size=600, random_state=rng),
    ])
    fit = el.fit_pot(data, threshold=u)
    from extremeloss.plotting import plot_gev_diagnostics, plot_gpd_diagnostics

    fig = plot_gpd_diagnostics(fit, data, observations_per_period=720.0)
    assert len(fig.axes) == 4
    maxima = genextreme.rvs(c=-0.1, loc=50.0, scale=8.0, size=200,
                            random_state=rng)
    fig2 = plot_gev_diagnostics(fit_gev(maxima), maxima)
    assert len(fig2.axes) == 4
    mpl.pyplot.close("all")

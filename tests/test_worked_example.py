"""Regression: the docs worked-example page numbers stay true."""
import numpy as np
import pytest

import lossmodels as lm
import risksim as rs
from extremeloss import fit_pot, splice_gpd_tail

rmod = pytest.importorskip("ratingmodels")
ap = pytest.importorskip("actuarialpy")


def test_worked_example_page_numbers():
    rng = np.random.default_rng(20260702)
    losses = lm.Burr(2.2, 20_000, 1.6).sample(2500, rng=rng)

    body = lm.fit_lognormal(losses)
    assert round(body.mu, 3) == 9.204 and round(body.sigma, 3) == 0.945

    u = float(np.quantile(losses, 0.95))
    fit = fit_pot(losses, threshold=u)
    assert round(fit.threshold) == 38_886
    assert round(fit.xi, 3) == 0.217
    assert round(fit.beta) == 15_281
    assert fit.n_exceedances == 125

    sev = splice_gpd_tail(body, fit)
    counts = np.array([242, 166, 153, 164, 195, 163, 162, 176])
    crm = lm.CollectiveRiskModel(lm.fit_negbinomial(counts), sev)
    assert crm.mean() == pytest.approx(2_475_636, rel=1e-4)

    port = rs.Portfolio([rs.PortfolioItem("commercial_block", crm)])
    treaty = rs.AggregateLayer(attachment=3_200_000, limit=1_500_000, name="agg_stop_loss")
    res = port.simulate(100_000, contract=treaty, rng=7)

    # Structural invariants: these hold for *any* valid draw, so they are immune
    # to lossless upstream numerical changes. (A fit landing at a different but
    # likelihood-equivalent parameter set re-seeds the simulation's draw
    # sequence, which shifts tail order statistics without changing correctness.)
    assert np.allclose(res.gross_losses, res.ceded_losses + res.retained_losses)
    assert res.ceded_losses.max() <= treaty.limit + 1e-6
    assert rs.metrics.tvar(res.gross_losses, 0.99) >= rs.metrics.var(res.gross_losses, 0.99)
    # Retained loss is capped at the attachment, so its 99% TVaR is exact.
    assert rs.metrics.tvar(res.retained_losses, 0.99) == pytest.approx(3_200_000.0, rel=1e-12)
    # Convergence: the simulation must reproduce the analytical aggregate mean
    # (a stable correctness check that does not depend on the exact draw).
    assert res.gross_losses.mean() == pytest.approx(crm.mean(), rel=1e-2)

    # Documented worked-example figures (seed=7). Tolerances are set from the
    # measured Monte Carlo variability of these estimators at 100k paths --
    # ~0.22% relative standard deviation for the 99% VaR/TVaR and ~2% for the
    # mean ceded -- so a re-seeded simulation stays green while a gross
    # regression (a broken splice, mis-scaled treaty) is still caught.
    assert rs.metrics.var(res.gross_losses, 0.99) == pytest.approx(3_511_586, rel=1.5e-2)
    assert rs.metrics.tvar(res.gross_losses, 0.99) == pytest.approx(3_688_820, rel=1.5e-2)
    assert res.ceded_losses.mean() == pytest.approx(9_303, rel=1e-1)

    lc = crm.mean() / 12_500.0
    ret = rmod.RetentionLoad(fixed_expense=22.0, variable_expense_ratio=0.09,
                             profit_margin=0.03, lae_ratio=0.05)
    pe = rmod.PricingEvaluation(loss_cost=lc, current_rate=255.0, retention=ret,
                                exposure=12_500.0, persistency=0.90)
    assert pe.premium_for_margin(0.03) == pytest.approx(261.31, abs=0.005)
    assert pe.at(0.0).margin_rate == pytest.approx(2.0966, abs=2e-3)  # dollars per unit

    uw = ap.UnderwritingSummary.from_per_exposure(
        revenue_per_exposure={"premium": 261.31},
        loss_per_exposure={"expected_losses": 198.05 * 1.05},
        expense_per_exposure=0.09 * 261.31 + 22.0,
        exposure=12_500.0,
    )
    assert uw.combined_ratio == pytest.approx(0.97, abs=2e-4)
    assert uw.gain_per_exposure == pytest.approx(7.84, abs=0.01)

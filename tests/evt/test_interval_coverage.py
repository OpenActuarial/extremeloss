"""Empirical coverage of the shipped confidence intervals.

The gold-standard validation: simulate from known truth, count how often
the interval contains it. Delta-method (Wald) intervals for return levels
are known to run modestly below nominal at moderate sample sizes -- the
sampling distribution of a return level is right-skewed and a symmetric
interval clips its upper reach (Coles recommends profile likelihood where
the last percent matters). The point of this test is that the number is
*measured and stated*, not assumed: ~0.91-0.92 observed at n_exc ~ 400
for nominal 0.95.
"""
import numpy as np
import pytest
from scipy.stats import genpareto

import extremeloss as el

XI, BETA, U, ZETA = 0.2, 1000.0, 5000.0, 0.10
T, M = 50.0, 500.0
R_TRUE = U + (BETA / XI) * ((T * M * ZETA) ** XI - 1.0)


def test_return_level_and_parameter_coverage():
    rng = np.random.default_rng(0)
    reps, n_total = 150, 4000
    rl_hits = xi_hits = usable = 0
    for _ in range(reps):
        n_exc = rng.binomial(n_total, ZETA)
        data = np.concatenate([
            rng.uniform(0.0, U, n_total - n_exc),
            U + genpareto.rvs(c=XI, scale=BETA, size=n_exc, random_state=rng),
        ])
        fit = el.fit_pot(data, threshold=U)
        if fit.covariance is None:
            continue
        usable += 1
        out = el.gpd_return_level(fit, T, observations_per_period=M,
                                  confidence_level=0.95)
        rl_hits += out["ci_low"][0] <= R_TRUE <= out["ci_high"][0]
        xi_hits += abs(fit.xi - XI) <= 1.959964 * fit.se[0]
    assert usable >= reps * 0.95  # covariance rarely degenerate here
    rl_cov = rl_hits / usable
    xi_cov = xi_hits / usable
    # return level: nominal 0.95, observed ~0.91 -- honest Wald behavior
    assert 0.85 <= rl_cov <= 0.99, f"return-level coverage {rl_cov:.3f}"
    # xi itself is closer to nominal (near-quadratic log-likelihood)
    assert 0.88 <= xi_cov <= 0.99, f"xi coverage {xi_cov:.3f}"

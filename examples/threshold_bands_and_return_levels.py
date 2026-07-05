"""Threshold selection with error bands, then return levels with CIs.

    losses -> threshold_diagnostic_table (xi and modified scale, WITH
    standard errors) -> pick the flat region -> fit_pot (covariance now
    populated) -> gpd_return_level with delta-method intervals
    -> closed-form mean excess for the pooling seam

Run with:  python examples/threshold_bands_and_return_levels.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto

import extremeloss as el


def main() -> None:
    rng = np.random.default_rng(7)
    body = rng.gamma(2.0, 8_000.0, 18_000)
    tail = 60_000.0 + genpareto.rvs(c=0.22, scale=25_000.0, size=2_000,
                                    random_state=rng)
    losses = np.concatenate([body, tail])

    # ----- where does GPD behavior start ---------------------------------- #
    grid = np.quantile(losses, [0.80, 0.85, 0.90, 0.93, 0.95, 0.97])
    scan = el.threshold_diagnostic_table(losses, grid)
    print("=== Threshold scan: flat xi and modified scale = valid region ===")
    print(pd.DataFrame({
        "threshold": scan.thresholds.round(0),
        "n_exceed": scan.n_exceedances,
        "xi": scan.xi.round(3), "xi_se": scan.xi_se.round(3),
        "mod_scale": scan.modified_scale.round(0),
        "mod_scale_se": scan.modified_scale_se.round(0),
    }).to_string(index=False))

    # ----- fit above the chosen threshold --------------------------------- #
    u = float(np.quantile(losses, 0.93))
    fit = el.fit_pot(losses, threshold=u)
    print(f"\n=== GPD above u = {u:,.0f} ===")
    print(f"xi   : {fit.xi:.3f}  (se {fit.se[0]:.3f})")
    print(f"beta : {fit.beta:,.0f}  (se {fit.se[1]:,.0f})")

    # ----- the loss exceeded once per T years ------------------------------ #
    out = el.gpd_return_level(fit, [10, 50, 200],
                              observations_per_period=2_000.0)
    print("\n=== Return levels (2,000 claims per year) ===")
    print(pd.DataFrame(out).round(0).to_string(index=False))

    # ----- the tail as a pooling input ------------------------------------- #
    d = u + 50_000.0
    print(f"\nmean excess at {d:,.0f}: {fit.mean_excess(d):,.0f}")
    print("(sf + mean_excess is the protocol"
          " ratingmodels.pooling_charge_from_severity consumes)")


if __name__ == "__main__":
    main()

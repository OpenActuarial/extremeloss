"""Did the tail fit succeed: numbers first, pictures if you have them.

    fit_pot -> qq_points / pp_points (headless diagnostics)
    -> parametric_bootstrap_gof (refit-per-replicate p-values)
    -> plot_gpd_diagnostics (the Coles four-panel, needs the plot extra)

Two datasets: one where the GPD is true and one where it cannot be --
the same exhibit should bless the first and convict the second.

Run with:  python examples/fit_diagnostics.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

import extremeloss as el


def main() -> None:
    rng = np.random.default_rng(7)
    u = 10_000.0

    good = np.concatenate([
        rng.uniform(0, u, 12_000),
        u + genpareto.rvs(c=0.2, scale=4_000.0, size=1_200, random_state=rng),
    ])
    bad = np.concatenate([
        rng.uniform(0, u, 12_000),
        u + np.abs(np.concatenate([rng.normal(2_000, 150, 600),
                                   rng.normal(12_000, 150, 600)])),
    ])

    for label, data in (("true GPD tail", good), ("bimodal tail", bad)):
        fit = el.fit_pot(data, threshold=u)
        qq = el.qq_points(fit, data)
        body = slice(qq["n"] // 4, 3 * qq["n"] // 4)
        dev = np.max(np.abs(qq["empirical"][body] - qq["theoretical"][body])
                     / qq["theoretical"][body])
        gof = el.parametric_bootstrap_gof(fit, data, n_boot=199, rng=1)
        print(f"=== {label} ===")
        print(f"xi = {fit.xi:.3f}, beta = {fit.beta:,.0f}, "
              f"n_exceedances = {fit.n_exceedances}")
        print(f"QQ body max deviation : {dev:.1%}")
        print(f"KS = {gof['ks']:.4f}  (bootstrap p = {gof['ks_pvalue']:.3f})")
        print(f"AD = {gof['ad']:.3f}  (bootstrap p = {gof['ad_pvalue']:.3f})")
        print()

    try:
        import matplotlib

        matplotlib.use("Agg")
        from extremeloss.plotting import plot_gpd_diagnostics

        fig = plot_gpd_diagnostics(el.fit_pot(good, threshold=u), good,
                                   observations_per_period=1_200.0)
        fig.savefig("gpd_diagnostics.png", dpi=110)
        print("wrote gpd_diagnostics.png (the Coles four-panel)")
    except ModuleNotFoundError:
        print("matplotlib not installed; numerical diagnostics above are"
              " the headless equivalent")


if __name__ == "__main__":
    main()

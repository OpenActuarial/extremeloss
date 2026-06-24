# extremeloss

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Extreme value theory, tail-risk estimation, and rare-event diagnostics.

---

## Overview

`extremeloss` focuses on the part of a loss distribution that is hardest to
estimate well: the **far tail**. It provides peaks-over-threshold and block-maxima
workflows, generalized-Pareto and generalized-extreme-value fits, tail-index
estimators, tail risk measures (VaR/TVaR) with uncertainty, importance-sampling and
bootstrap machinery, and threshold diagnostics — all returning rich result objects
you can summarize, plot, and feed back into the rest of a modeling pipeline.

It is built to sit alongside `lossmodels` (loss distributions and aggregate models)
and `risksim` (portfolio and contract simulation): it can fit a tail directly from
a `lossmodels` object, splice a fitted tail onto a body distribution, or analyze a
`risksim` simulation result. Its only hard dependencies are `numpy` and `scipy`.

## Highlights

- **Peaks-over-threshold (POT) / GPD** — extract exceedances, choose a threshold,
  fit a generalized Pareto distribution, and read off tail probabilities, VaR,
  TVaR, and return levels.
- **Block maxima / GEV** — block a series, fit a generalized extreme value
  distribution, and compute return levels.
- **Tail-index estimators** — Hill and Pickands estimators, with a Hill curve.
- **Threshold diagnostics** — mean-excess analysis and a threshold-stability scan.
- **Tail risk with uncertainty** — empirical and model-based VaR/TVaR/tail
  probabilities returned as estimates with confidence intervals.
- **Variance reduction** — importance-sampling estimators with weight diagnostics
  and effective sample size.
- **Bootstrap** — resampling uncertainty for tail probabilities, VaR, TVaR, and
  arbitrary scalar statistics.
- **Ecosystem integration** — fit tails from `lossmodels` objects, splice GPD tails
  onto bodies, and summarize `risksim` simulations.

## Installation

```bash
pip install extremeloss
```

From source:

```bash
pip install -e .
```

Requires Python `>=3.10` with `numpy` and `scipy`. The integration helpers use
`lossmodels` / `risksim` objects when present, and the optional plotting helpers
require `matplotlib`.

## Package structure

```text
extremeloss/
├── evt/             # GPD, POT, block-maxima/GEV, tail-index, thresholds
├── estimation/      # empirical, conditional-MC, and importance-sampling estimators
├── analytics/       # return periods/levels, summaries, diagnostics
├── utils/           # bootstrap and validation helpers
├── integration.py   # lossmodels / risksim interop and splicing
├── results.py       # GPDFit, GEVFit, GPDTail, TailEstimateResult, BootstrapResult, ThresholdScan
├── protocols.py     # SupportsSample / SupportsLosses / SupportsSimulationResult
└── plotting.py      # optional matplotlib diagnostics
```

## Quick start

```python
import numpy as np
from extremeloss import fit_pot

losses = np.random.default_rng(0).pareto(2.5, 50_000) * 1000  # heavy-tailed sample

fit = fit_pot(losses, threshold=np.quantile(losses, 0.95))   # -> GPDFit
print("shape xi      :", fit.xi)
print("scale beta    :", fit.beta)
print("99.5% VaR     :", fit.var(0.995))
print("99.5% TVaR    :", fit.tvar(0.995))
print("P(loss > 50k) :", fit.tail_probability(50_000))
print("100-obs return level:", fit.return_level(100))
print(fit.summary())
```

## Peaks-over-threshold and the GPD

The POT workflow: pick a threshold (guided by diagnostics below), keep the
exceedances, and fit a generalized Pareto distribution to them.

```python
from extremeloss import extract_exceedances, fit_gpd, fit_pot, gpd_var, gpd_tvar

excesses = extract_exceedances(losses, threshold=10_000)   # losses above the threshold
fit = fit_gpd(excesses, threshold=10_000)                  # fit to excesses, or:
fit = fit_pot(losses, threshold=10_000)                    # fit directly from the data

# functional forms are available when you already have the parameters
v = gpd_var(0.995, threshold=10_000, xi=fit.xi, beta=fit.beta,
            exceedance_fraction=fit.exceedance_fraction)
```

A `GPDFit` carries `threshold`, `xi`, `beta`, `exceedance_fraction`, and
`n_exceedances`, and exposes `var(p)`, `tvar(p)`, `tail_probability(x)`,
`return_level(period)`, and `summary()`.

## Block maxima and the GEV

```python
from extremeloss import make_blocks, fit_block_maxima, fit_gev, block_return_level

maxima = make_blocks(losses, block_size=250)        # block maxima
gev = fit_block_maxima(losses, block_size=250)      # or fit_gev(maxima)
print(gev.xi, gev.loc, gev.scale)
print("100-block return level:", gev.return_level(100))
```

A `GEVFit` carries `xi`, `loc`, `scale`, and `n_blocks`, and exposes
`return_level(period)`, `cdf(x)`, and `summary()`.

## Threshold diagnostics

Choosing the POT threshold is the crux of a good tail fit. Mean-excess and
threshold-stability scans help:

```python
from extremeloss import mean_excess, threshold_diagnostic_table

grid = np.quantile(losses, [0.90, 0.95, 0.975, 0.99])
me = mean_excess(losses, grid)                       # mean excess at each threshold
scan = threshold_diagnostic_table(losses, grid)      # -> ThresholdScan
print(scan.thresholds, scan.xi, scan.beta, scan.n_exceedances)
```

A `ThresholdScan` exposes `thresholds`, `mean_excess`, `xi`, `beta`,
`n_exceedances`, and `to_dict()`. Look for the threshold above which the shape
`xi` and the mean excess stabilize.

## Tail-index estimators

```python
from extremeloss import hill_estimator, hill_curve, pickands_estimator

hill_estimator(losses, k=500)     # Hill tail-index estimate using the top k order statistics
pickands_estimator(losses, k=500)
curve = hill_curve(losses)        # Hill estimate across a grid of k (for a Hill plot)
```

## Return periods and levels

```python
from extremeloss import return_period, return_level, block_return_level

return_period(0.01)             # expected waiting time for a 1% exceedance
return_level(100, fit)          # the level exceeded once per 100 observations (GPDFit)
block_return_level(100, gev)    # the level exceeded once per 100 blocks (GEVFit)
```

## Tail risk estimation with uncertainty

Empirical estimators are one-liners; the model-based estimators return a
`TailEstimateResult` with a confidence interval and standard error.

```python
from extremeloss import empirical_var, empirical_tvar, estimate_var, estimate_tvar, estimate_tail_probability

empirical_var(losses, 0.99)
empirical_tvar(losses, 0.99)

res = estimate_var(losses, 0.99)          # -> TailEstimateResult
print(res.estimate, res.ci, res.stderr)
print(res.summary())

estimate_tvar(losses, 0.99)
estimate_tail_probability(losses, threshold=100_000)
```

Conditional Monte Carlo summaries are available when you already have per-scenario
conditional probabilities or tail expectations (`estimate_tail_probability_cmc`,
`estimate_tvar_cmc`).

## Importance sampling

When the event of interest is rare, importance sampling estimates it far more
efficiently than crude Monte Carlo. Supply the losses drawn from a proposal
distribution and their importance weights:

```python
from extremeloss import (
    estimate_var_is, estimate_tvar_is, estimate_tail_probability_is,
    importance_sampling_diagnostics, effective_sample_size, stabilize_weights,
)

# losses and weights come from your proposal sampler
res = estimate_tail_probability_is(losses, weights, threshold=1_000_000)
print(res.estimate, res.ci, res.effective_n)

estimate_var_is(losses, weights, 0.999)
estimate_tvar_is(losses, weights, 0.999)

print(importance_sampling_diagnostics(weights))   # weight quality metrics
print(effective_sample_size(weights))
w = stabilize_weights(weights, clip_quantile=0.999)  # tame extreme weights
```

`log_importance_weights(log_target_density, log_proposal_density)` builds normalized
weights directly from log-densities, and `estimate_mean_is` /
`estimate_exceedance_curve_is` / `estimate_var_tvar_is` cover means, full exceedance
curves, and joint VaR/TVaR.

## Bootstrap uncertainty

```python
from extremeloss import bootstrap_var, bootstrap_tvar, bootstrap_tail_probability, bootstrap_statistic

bv = bootstrap_var(losses, 0.99, n_resamples=1000)   # -> BootstrapResult
print(bv.estimate, bv.ci, bv.stderr)

bootstrap_tvar(losses, 0.99)
bootstrap_tail_probability(losses, threshold=100_000)

# wrap any scalar statistic
bootstrap_statistic(losses, np.median, n_resamples=1000)
```

A `BootstrapResult` carries `estimate`, `bootstrap_estimates`, `ci`, `stderr`, and
`summary()`.

## The GPD tail object and splicing

`GPDTail` turns a fitted tail into a standalone, sampleable severity supported above
the threshold (`sample`, `cdf`, `quantile`, `mean`, `variance`). Because its
`cdf(threshold) = 0`, it satisfies the splicing contract used by
`lossmodels.SplicedSeverity`, letting you attach an EVT tail to a body distribution:

```python
from extremeloss import fit_pot, GPDTail, splice_gpd_tail, fit_spliced_gpd
from lossmodels import Gamma, SplicedSeverity

fit = fit_pot(losses, threshold=10_000)
tail = GPDTail.from_fit(fit)             # a severity for the tail above 10k

# splice an EVT tail onto a lossmodels body, two equivalent ways:
spliced = splice_gpd_tail(Gamma(2.0, 4_000), fit)            # via extremeloss helper
spliced = fit_spliced_gpd(Gamma(2.0, 4_000), losses, threshold=10_000)

# or assemble it yourself with lossmodels:
body = Gamma(2.0, 4_000)
u = tail.threshold
spliced = SplicedSeverity(body=body, tail=tail, threshold=u, weight=body.cdf(u))
```

The spliced object is a full `lossmodels` severity, so it drops straight back into a
collective-risk model or a `risksim` portfolio.

## Ecosystem integration

`extremeloss` reads directly from `lossmodels` and `risksim` objects:

| Helper | Purpose |
| --- | --- |
| `sample_lossmodel(model, size)` | draw a sample from any `lossmodels` model |
| `fit_pot_from_lossmodel(model, size, threshold)` | sample a model and fit a GPD tail in one step |
| `losses_from_risksim(result, view)` | pull a loss vector (gross / ceded / retained) from a `risksim` `SimulationResult` |
| `tail_summary_from_risksim(result, ...)` | a tail summary of a `risksim` simulation |
| `component_tail_metrics(result, q, ...)` | per-component tail metrics from a simulation |
| `layer_tail_metrics(result, q, ...)` | per-layer tail metrics from a simulation |

```python
from extremeloss import fit_pot_from_lossmodel, losses_from_risksim, tail_summary_from_risksim
from lossmodels import ParetoII

fit = fit_pot_from_lossmodel(ParetoII(2.5, 1000), size=40_000, threshold=1_500)

# after running a risksim Portfolio.simulate(...) -> result:
# net_losses = losses_from_risksim(result, view="retained")
# summary = tail_summary_from_risksim(result, quantiles=(0.95, 0.99, 0.995))
```

## Plotting (optional)

With `matplotlib` installed, the `extremeloss.plotting` module offers diagnostic
plots — `plot_mean_excess(losses, thresholds)`, `plot_hill_curve(losses)`, and
`plot_exceedance_curve(losses, thresholds)` (each accepts an optional `ax`).

## The ActuarialPy ecosystem

`extremeloss` is the tail-modeling layer of a small family of actuarial packages
that interoperate through the `.sample()` / `.mean()` interface:

- **`lossmodels`** — frequency / severity distributions, aggregate (collective-risk)
  loss models, coverage modifications, and model fitting; the source of body
  distributions for splicing and of models to fit tails from.
- **`risksim`** — portfolio loss simulation and aggregate reinsurance; its
  simulation results feed the integration helpers above.
- **`actuarialpy`** — deterministic, experience-and-data analysis (summaries,
  triangles, trend, credibility) on tabular data.

## Testing

```bash
pytest -q
```

## License

MIT License
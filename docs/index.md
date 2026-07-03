# extremeloss documentation

`extremeloss` is a Python library for **extreme-loss estimation**, **extreme value modeling**, and **tail-risk diagnostics**.

It is intended to sit alongside:

- `lossmodels` for loss distributions and aggregate-loss modeling
- `risksim` for contract and portfolio simulation

The package focuses on the hardest part of the distribution to estimate well: the **far tail**.

## What is included

### Rare-event estimation

- empirical exceedance probabilities, VaR, and TVaR
- conditional Monte Carlo summaries from precomputed conditional expectations or probabilities
- importance-sampling estimators for means, tail probabilities, exceedance curves, VaR, and TVaR
- effective sample size and weight diagnostics

### Extreme value theory workflows

- peaks-over-threshold / GPD fitting
- block-maxima / GEV fitting
- Hill and Pickands tail-index estimators
- mean-excess and threshold-stability diagnostics

### Tail-risk analytics

- extreme-loss summary tables
- return periods and return levels
- bootstrap uncertainty estimation for tail statistics
- integration helpers for `lossmodels`-style models and `risksim`-style simulation results

## Documentation map

- [Getting started](guides/getting-started.md)
- [Package overview](guides/package-overview.md)
- [Design and ecosystem fit](guides/design.md)
- [Integration guide](guides/integration.md)
- [Examples overview](examples/README.md)
- [API reference](api/reference.md)

## Recommended reading order

1. Start with [Getting started](guides/getting-started.md)
2. Review [Package overview](guides/package-overview.md)
3. Read [Integration guide](guides/integration.md) if you plan to use `lossmodels` or `risksim`
4. Browse the runnable scripts in [Examples overview](examples/README.md)
5. Use [API reference](api/reference.md) as needed

## First example

```python
import numpy as np
from extremeloss import estimate_tail_probability, fit_pot

rng = np.random.default_rng(123)
losses = rng.lognormal(mean=2.0, sigma=0.9, size=20_000)

result = estimate_tail_probability(losses, threshold=100.0)
print(result.summary())

fit = fit_pot(losses, threshold=50.0)
print(fit.summary())
print(fit.var(0.995))
```

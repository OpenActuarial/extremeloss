# extremeloss

`extremeloss` is a Python library for **extreme-loss estimation**, **extreme value modeling**, and **tail-risk diagnostics**.

It is designed to sit alongside:

- [`lossmodels`](https://github.com/michaelabryant/lossmodels) for actuarial loss distributions and aggregate modeling
- [`risksim`](https://github.com/michaelabryant/risksim) for portfolio and contract-level simulation

The package focuses on the part of the loss distribution that is hardest to estimate well: the **far tail**.

## Scope

`extremeloss` currently provides an MVP focused on three areas:

1. **Rare-event / tail estimation**
  - empirical exceedance probability estimation
  - empirical VaR and TVaR estimation
  - importance-sampling utilities for weighted tail estimation

2. **Extreme value theory (EVT)**
  - peaks-over-threshold (POT) workflows
  - generalized Pareto distribution (GPD) fitting
  - Hill and Pickands tail-index estimators
  - threshold diagnostics and mean-excess analysis

3. **Tail-risk analytics**
  - return periods and return levels
  - summary tables for tail quantities
  - lightweight plotting helpers for tail diagnostics

## Why this library exists

Naive Monte Carlo works well for central parts of the distribution but becomes inefficient in very small-probability regions. `extremeloss` provides a place for:

- extreme-tail estimators
- EVT-based tail extrapolation
- threshold diagnostics
- extreme-region risk reporting

That gives it a distinct role relative to `lossmodels` and `risksim` instead of duplicating generic risk-measure functionality.

## Installation

Editable install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e .[dev]
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

## Quick start

### Empirical tail probability

```python
import numpy as np
from extremeloss import estimate_tail_probability

rng = np.random.default_rng(123)
losses = rng.lognormal(mean=2.0, sigma=0.9, size=50_000)

result = estimate_tail_probability(losses, threshold=50.0)
print(result.summary())
```

### Empirical VaR and TVaR

```python
import numpy as np
from extremeloss import estimate_var, estimate_tvar

rng = np.random.default_rng(123)
losses = rng.gamma(shape=2.0, scale=20.0, size=25_000)

var_99 = estimate_var(losses, q=0.99)
tvar_99 = estimate_tvar(losses, q=0.99)

print("VaR(0.99):", var_99.estimate)
print("TVaR(0.99):", tvar_99.estimate)
```

### POT / GPD workflow

```python
import numpy as np
from extremeloss import fit_pot

rng = np.random.default_rng(123)
losses = rng.lognormal(mean=2.0, sigma=1.0, size=100_000)

fit = fit_pot(losses, threshold=40.0)
print(fit.summary())
print("Tail probability above 100:", fit.tail_probability(100.0))
print("VaR(0.995):", fit.var(0.995))
print("Return level for period 250:", fit.return_level(250.0))
```

### Tail diagnostics and plotting

```python
import numpy as np
from extremeloss import hill_curve, mean_excess
from extremeloss.plotting import plot_hill_curve, plot_mean_excess

rng = np.random.default_rng(123)
losses = rng.pareto(a=3.0, size=10_000) + 1.0

curve = hill_curve(losses)
plot_hill_curve(curve)

thresholds = np.quantile(losses, np.linspace(0.80, 0.98, 10))
scan = mean_excess(losses, thresholds)
plot_mean_excess(scan)
```

## Main API

### Estimation

- `estimate_tail_probability`
- `estimate_var`
- `estimate_tvar`
- `estimate_tail_probability_is`
- `estimate_var_is`
- `estimate_tvar_is`
- `exceedance_probability`
- `effective_sample_size`

### EVT

- `extract_exceedances`
- `fit_gpd`
- `fit_pot`
- `gpd_tail_probability`
- `gpd_var`
- `gpd_tvar`
- `hill_estimator`
- `pickands_estimator`
- `hill_curve`
- `mean_excess`
- `threshold_diagnostic_table`

### Analytics

- `return_period`
- `return_level`
- `extreme_loss_summary`

### Result objects

- `TailEstimateResult`
- `GPDFit`
- `ThresholdScan`

## Package layout

```text
extremeloss/
pyproject.toml
examples/
src/extremeloss/
    analytics/
    __init__.py
    diagnostics.py
    return_periods.py
                evt/
    __init__.py
    gpd.py
    pot.py
    tail_index.py
    thresholds.py
    protocols.py
    utils/
        validation.py
```

## Design principles

### Array-first API

Most functions work directly on one-dimensional arrays of losses.

### Compatible with simulation workflows

Where appropriate, the package can also work with objects exposing a `sample(size)` method. That keeps it compatible with the style used in `risksim`.

### Lightweight result containers

Estimators and EVT fits return objects that carry both values and metadata, rather than raw scalars only.

### Focus on the far tail

`extremeloss` is meant to specialize in extreme-region estimation and diagnostics, not to replace general-purpose simulation or loss-distribution libraries.

## Documentation

Documentation is included in the repository under `docs/`.

Suggested reading order:

1. `docs/guides/getting-started.md`
2. `docs/guides/package-overview.md`
3. `docs/guides/design.md`
4. `docs/api/reference.md`
5. `docs/examples/README.md`

Main entry point:

- `docs/index.md`

## Examples

Example scripts are included in `examples/`:

- `empirical_tail_analysis.py`
- `pot_gpd_workflow.py`
- `importance_sampling_demo.py`
- `diagnostic_plots.py`

Generated example outputs are stored in `examples/output/`.

## Testing

Run the test suite with:

```bash
pytest -q
```

Install development dependencies first if needed:

```bash
pip install -e .[dev]
```

## Current status

This repository is currently an **alpha-stage scaffold / MVP**. The current version establishes:

- package structure
- public API shape
- result objects
- documentation and examples
- test coverage for the existing functionality

Natural next steps include:

- conditional Monte Carlo methods
- richer importance-sampling estimators
- block-maxima / GEV workflows
- bootstrap uncertainty estimation
- deeper integration helpers for `lossmodels` and `risksim`

## License

MIT
# extremeloss

`extremeloss` is a Python library for **extreme-loss estimation**, **extreme value modeling**, and **tail-risk diagnostics**.

It is designed to sit alongside:

- [`lossmodels`](https://github.com/actuarialpy/lossmodels) for actuarial loss distributions and aggregate modeling
- [`risksim`](https://github.com/actuarialpy/risksim) for portfolio and contract-level simulation

The package focuses on the part of the loss distribution that is hardest to estimate well: the **far tail**.

## Scope

`extremeloss` currently covers five main areas.

### 1. Rare-event and tail estimation

- empirical exceedance probability estimation
- empirical VaR and TVaR estimation
- conditional Monte Carlo summaries from precomputed conditional probabilities or tail expectations
- importance-sampling estimators for means, tail probabilities, exceedance curves, VaR, and TVaR
- effective sample size and weight diagnostics

### 2. Extreme value theory workflows

- peaks-over-threshold (POT) workflows
- generalized Pareto distribution (GPD) fitting
- block-maxima / generalized extreme value (GEV) workflows
- Hill and Pickands tail-index estimators
- threshold diagnostics and mean-excess analysis

### 3. Tail-risk analytics

- return periods and return levels
- summary tables for tail quantities
- exceedance-frequency views

### 4. Uncertainty estimation

- bootstrap uncertainty estimation for tail probabilities, VaR, and TVaR
- reusable bootstrap wrapper for scalar statistics

### 5. Ecosystem integration helpers

- duck-typed helpers for `lossmodels`-style objects exposing `sample(size)`
- duck-typed helpers for `risksim`-style result objects exposing `losses`, `gross_losses`, `retained_losses`, or `ceded_losses`
- component and layer tail summaries for simulation outputs

## Why this library exists

Naive Monte Carlo works well in the center of a distribution but becomes inefficient when estimating very small tail probabilities or very high quantiles. `extremeloss` gives your ecosystem a clear third layer:

- `lossmodels` defines or samples loss models
- `risksim` simulates portfolios and contract structures
- `extremeloss` handles the far tail, EVT extrapolation, and tail diagnostics

That keeps the package distinct rather than turning it into another generic risk-measures library.

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

### Block-maxima / GEV workflow

```python
import numpy as np
from extremeloss import fit_block_maxima

rng = np.random.default_rng(123)
losses = rng.pareto(a=3.0, size=36_500) * 20.0

fit = fit_block_maxima(losses, block_size=365)
print(fit.summary())
print("Return level for period 20:", fit.return_level(20.0))
```

### Bootstrap uncertainty estimation

```python
import numpy as np
from extremeloss import bootstrap_tail_probability

rng = np.random.default_rng(123)
losses = rng.lognormal(mean=2.0, sigma=0.9, size=8_000)

boot = bootstrap_tail_probability(losses, threshold=80.0, n_resamples=250, random_state=1)
print(boot.summary())
```

### Risksim-style integration

```python
from dataclasses import dataclass
import numpy as np
from extremeloss import tail_summary_from_risksim

@dataclass
class SimpleResult:
    losses: np.ndarray
    gross_losses: np.ndarray
    retained_losses: np.ndarray
    ceded_losses: np.ndarray

rng = np.random.default_rng(123)
losses = rng.gamma(shape=2.0, scale=30.0, size=25_000)
result = SimpleResult(
    losses=losses,
    gross_losses=losses,
    retained_losses=0.9 * losses,
    ceded_losses=0.1 * losses,
)

print(tail_summary_from_risksim(result, view="retained", thresholds=[20, 40, 60]))
```

## Main API

### Estimation

- `estimate_tail_probability`
- `estimate_var`
- `estimate_tvar`
- `estimate_var_tvar`
- `estimate_tail_probability_cmc`
- `estimate_tvar_cmc`
- `estimate_mean_is`
- `estimate_tail_probability_is`
- `estimate_exceedance_curve_is`
- `estimate_var_is`
- `estimate_tvar_is`
- `estimate_var_tvar_is`
- `effective_sample_size`
- `importance_sampling_diagnostics`
- `log_importance_weights`
- `stabilize_weights`

### EVT

- `extract_exceedances`
- `fit_gpd`
- `fit_pot`
- `gpd_tail_probability`
- `gpd_var`
- `gpd_tvar`
- `make_blocks`
- `fit_gev`
- `fit_block_maxima`
- `block_return_level`
- `hill_estimator`
- `pickands_estimator`
- `hill_curve`
- `mean_excess`
- `threshold_diagnostic_table`

### Analytics

- `return_period`
- `return_level`
- `exceedance_frequency`
- `extreme_loss_summary`
- `var_tvar_diagnostic_table`

### Bootstrap utilities

- `bootstrap_statistic`
- `bootstrap_tail_probability`
- `bootstrap_var`
- `bootstrap_tvar`

### Integration helpers

- `sample_lossmodel`
- `fit_pot_from_lossmodel`
- `losses_from_risksim`
- `tail_summary_from_risksim`
- `component_tail_metrics`
- `layer_tail_metrics`

### Result objects

- `TailEstimateResult`
- `GPDFit`
- `GEVFit`
- `BootstrapResult`
- `ThresholdScan`

## Package layout

```text
extremeloss/
├── README.md
├── pyproject.toml
├── docs/
├── examples/
├── tests/
└── src/extremeloss/
    ├── __init__.py
    ├── analytics/
    │   ├── __init__.py
    │   ├── diagnostics.py
    │   └── return_periods.py
    ├── estimation/
    │   ├── __init__.py
    │   ├── conditional_mc.py
    │   ├── importance_sampling.py
    │   ├── metrics.py
    │   └── rare_event.py
    ├── evt/
    │   ├── __init__.py
    │   ├── block_maxima.py
    │   ├── gpd.py
    │   ├── pot.py
    │   ├── tail_index.py
    │   └── thresholds.py
    ├── integration.py
    ├── plotting.py
    ├── protocols.py
    ├── results.py
    └── utils/
        ├── __init__.py
        ├── bootstrap.py
        └── validation.py
```

## Design principles

### Array-first API

Most functions work directly on one-dimensional arrays of losses.

### Duck-typed interoperability

The package stays standalone, but convenience helpers accept objects that behave like your other packages:

- `sample(size)` for `lossmodels`-style models
- `losses` / `gross_losses` / `retained_losses` / `ceded_losses` for `risksim`-style results

### Lightweight result containers

Estimators and fitted models return structured objects rather than raw scalars only.

### Focus on the far tail

`extremeloss` is meant to specialize in extreme-region estimation and diagnostics, not replace general-purpose simulation or loss-distribution libraries.

## Documentation

Repository markdown docs are included under `docs/`.

Suggested reading order:

1. `docs/guides/getting-started.md`
2. `docs/guides/package-overview.md`
3. `docs/guides/design.md`
4. `docs/guides/integration.md`
5. `docs/examples/README.md`
6. `docs/api/reference.md`

Main entry point:

- `docs/index.md`

## Examples

Runnable scripts are included in `examples/`.

### Core workflows

- `empirical_tail_analysis.py`
- `pot_gpd_workflow.py`
- `importance_sampling_demo.py`
- `diagnostic_plots.py`

### Extended workflows

- `conditional_mc_demo.py`
- `block_maxima_gev_workflow.py`
- `bootstrap_uncertainty_demo.py`
- `integration_helpers_demo.py`

Run them from the repository root:

```bash
python examples/empirical_tail_analysis.py
python examples/pot_gpd_workflow.py
python examples/importance_sampling_demo.py
python examples/conditional_mc_demo.py
python examples/block_maxima_gev_workflow.py
python examples/bootstrap_uncertainty_demo.py
python examples/integration_helpers_demo.py
python examples/diagnostic_plots.py
```

The scripts add `src/` to `sys.path`, so they can run directly from the repo without requiring a prior editable install.

Generated plot images are written to `examples/output/`.

## Testing

Run the full suite with:

```bash
pytest -q
```

Install development dependencies first if needed:

```bash
pip install -e .[dev]
```

## Troubleshooting editable installs

If you add new modules or result classes and Python still imports an older installed copy, clear caches and reinstall:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
pip uninstall -y extremeloss
pip install -e .
```

To verify the active import path:

```bash
python -c "import extremeloss; import extremeloss.results as r; print(extremeloss.__file__); print(r.__file__)"
```

## Roadmap ideas

Natural next areas to add include:

- more specialized conditional Monte Carlo methods for aggregate-loss models
- more advanced importance-sampling strategies for compound losses
- richer EVT diagnostics and threshold-selection tools
- multivariate extremes and tail dependence
- direct optional adapters for installed `lossmodels` and `risksim`
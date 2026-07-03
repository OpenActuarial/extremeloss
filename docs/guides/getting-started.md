# Getting started

## Installation

Editable install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e .[dev]
```

Run the test suite:

```bash
pytest -q
```

## Running examples

From the repository root:

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

The example scripts add `src/` to `sys.path`, so they run directly from the repo without requiring a prior editable install.

## Package layout

```text
src/extremeloss/
├── analytics/
├── estimation/
├── evt/
├── utils/
├── integration.py
├── plotting.py
├── protocols.py
└── results.py
```

## Core concepts

### Array-first API

Most functions accept a one-dimensional array of losses:

```python
import numpy as np
from extremeloss import estimate_var

losses = np.array([5.0, 7.0, 9.0, 15.0, 40.0])
result = estimate_var(losses, q=0.95)
print(result.estimate)
```

### Model-like inputs

Integration helpers also work with objects that implement `sample(size)`:

```python
import numpy as np
from extremeloss import fit_pot_from_lossmodel

class LognormalModel:
    def __init__(self, mean: float, sigma: float, seed: int = 123):
        self.mean = mean
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int) -> np.ndarray:
        return self.rng.lognormal(self.mean, self.sigma, size=size)

model = LognormalModel(mean=2.0, sigma=0.7)
fit = fit_pot_from_lossmodel(model, size=50_000, threshold=40.0)
print(fit.summary())
```

### Risksim-style result views

Some helpers operate on objects exposing `losses`, `gross_losses`, `retained_losses`, or `ceded_losses` attributes.

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
losses = rng.gamma(shape=2.0, scale=30.0, size=10_000)
result = SimpleResult(losses=losses, gross_losses=losses, retained_losses=0.9 * losses, ceded_losses=0.1 * losses)

print(tail_summary_from_risksim(result, view="retained"))
```

## Typical workflows

### 1. Empirical tail analysis

- estimate exceedance probabilities
- estimate empirical VaR and TVaR
- summarize tail behavior over several thresholds

### 2. EVT extrapolation

- choose a threshold or block size
- fit POT/GPD or block-maxima/GEV
- compute extreme-region VaR, TVaR, or return levels

### 3. Simulation-based tail analysis

- generate or collect losses from another package
- use importance sampling or conditional Monte Carlo summaries
- bootstrap uncertainty around tail measures

## Troubleshooting

### ImportError after updating the package

If a newly added class or function cannot be imported, clear cached bytecode and reinstall in editable mode:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
pip uninstall -y extremeloss
pip install -e .
```

Then verify the active install path:

```bash
python -c "import extremeloss; print(extremeloss.__file__)"
```

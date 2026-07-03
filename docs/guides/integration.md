# Integration guide

`extremeloss` is designed to work with your other packages without hard-coding imports from them.

## Using `sample(size)` models

Any object that implements `sample(size)` can be used with the integration helpers.

```python
import numpy as np
from extremeloss import sample_lossmodel, fit_pot_from_lossmodel

class ToySeverity:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int) -> np.ndarray:
        return self.rng.lognormal(mean=2.0, sigma=0.8, size=size)

model = ToySeverity()
losses = sample_lossmodel(model, size=100_000)
fit = fit_pot_from_lossmodel(model, size=100_000, threshold=50.0)
```

## Using risksim-like simulation results

A risksim-like result can expose several views.

Accepted view names:

- `"losses"`
- `"gross"` or `"gross_losses"`
- `"retained"` or `"retained_losses"`
- `"ceded"` or `"ceded_losses"`

```python
from dataclasses import dataclass
import numpy as np
from extremeloss import losses_from_risksim, tail_summary_from_risksim

@dataclass
class SimpleResult:
    losses: np.ndarray
    gross_losses: np.ndarray
    retained_losses: np.ndarray
    ceded_losses: np.ndarray

rng = np.random.default_rng(123)
losses = rng.gamma(shape=2.5, scale=20.0, size=25_000)
result = SimpleResult(
    losses=losses,
    gross_losses=losses,
    retained_losses=0.85 * losses,
    ceded_losses=0.15 * losses,
)

retained = losses_from_risksim(result, view="retained")
summary = tail_summary_from_risksim(result, view="ceded", thresholds=[20, 40, 60])
```

## Component and layer metrics

If your result object exposes `component_losses` or `layer_losses` as 2D arrays, the helpers can summarize each column separately.

```python
from dataclasses import dataclass
import numpy as np
from extremeloss import component_tail_metrics, layer_tail_metrics

@dataclass
class PortfolioResult:
    losses: np.ndarray
    component_losses: np.ndarray
    component_names: list[str]
    layer_losses: np.ndarray
    layer_names: list[str]

rng = np.random.default_rng(123)
component_losses = rng.gamma(shape=2.0, scale=10.0, size=(10_000, 3))
layer_losses = rng.gamma(shape=1.5, scale=12.0, size=(10_000, 2))
result = PortfolioResult(
    losses=component_losses.sum(axis=1),
    component_losses=component_losses,
    component_names=["freq", "sev", "cat"],
    layer_losses=layer_losses,
    layer_names=["working", "cat"],
)

print(component_tail_metrics(result, q=0.99, threshold=50.0))
print(layer_tail_metrics(result, q=0.99, threshold=40.0))
```

## Practical recommendation

Keep `extremeloss` array-first internally. Use the integration layer for convenience, but avoid building the core estimators around hard assumptions about one upstream package.

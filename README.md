# extremeloss

Extreme value analysis for insurance losses: POT/GPD, GEV, and tail risk with uncertainty.

[![CI](https://github.com/OpenActuarial/extremeloss/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/extremeloss/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/extremeloss)](https://pypi.org/project/extremeloss/)
[![Python](https://img.shields.io/pypi/pyversions/extremeloss)](https://pypi.org/project/extremeloss/)

## Overview

`extremeloss` estimates the part of the distribution the data barely shows
you: peaks-over-threshold fitting of the generalized Pareto distribution,
block-maxima fitting of the GEV with uncertainty quantification, threshold
selection diagnostics, tail-index estimators, return periods and levels, and
body–tail splicing that plugs a fitted tail onto any `lossmodels` severity.

Fits report parameter uncertainty rather than point estimates alone, and the
tail measures (VaR, TVaR, tail probabilities, return levels) quote
unconditional, ground-up results.

## Installation

```bash
pip install extremeloss
```

Diagnostic plots require the plotting extra:

```bash
pip install "extremeloss[plot]"
```

Requires Python 3.10 or newer.

## Quick start

```python
import numpy as np
from extremeloss import fit_pot

losses = np.random.default_rng(0).pareto(2.5, 50_000) * 1000  # heavy-tailed sample

fit = fit_pot(losses, threshold=np.quantile(losses, 0.95))    # -> GPD fit
print("shape xi      :", fit.xi)
print("scale beta    :", fit.beta)
print("99.5% VaR     :", fit.var(0.995))
print("99.5% TVaR    :", fit.tvar(0.995))
print("P(loss > 50k) :", fit.tail_probability(50_000))
print(fit.summary())
```

## What's inside

- **Peaks over threshold** — GPD fitting with parameter uncertainty and
  functional forms when parameters are already known.
- **Block maxima** — GEV fitting, diagnostics, and uncertainty
  quantification.
- **Threshold selection** — mean-excess and stability diagnostics.
- **Tail indices** — Hill-type and related estimators.
- **Tail measures** — VaR, TVaR, tail probabilities, return periods and
  levels, importance sampling for rare events.
- **Integration** — body–tail splicing onto `lossmodels` severities;
  optional plotting via the `plot` extra.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/extremeloss.html](https://openactuarial.org/extremeloss.html)**.

## The OpenActuarial ecosystem

`extremeloss` is one of seven packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| [actuarialpy](https://github.com/OpenActuarial/actuarialpy) | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| [projectionmodels](https://github.com/OpenActuarial/projectionmodels) | Claim, premium, and expense projection over a renewal horizon |
| [ratingmodels](https://github.com/OpenActuarial/ratingmodels) | Manual and experience rating, credibility, indication, GLM relativities |
| [lossmodels](https://github.com/OpenActuarial/lossmodels) | Severity and frequency fitting, aggregate loss distributions |
| **[extremeloss](https://github.com/OpenActuarial/extremeloss)** | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| [risksim](https://github.com/OpenActuarial/risksim) | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/extremeloss
cd extremeloss
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).

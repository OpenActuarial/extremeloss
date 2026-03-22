# extremeloss

`extremeloss` is a Python library for extreme-loss estimation, peaks-over-threshold modeling, and tail-risk diagnostics.

It is designed to complement:

- `lossmodels` for actuarial loss distributions and aggregate models
- `risksim` for portfolio and contract-level simulation

## Highlights

- Empirical tail probability, VaR, and TVaR estimation
- Importance-sampling utilities for weighted rare-event estimation
- Generalized Pareto (GPD) fitting for peaks-over-threshold workflows
- Hill and Pickands tail-index estimators
- Threshold diagnostics and return-period utilities
- Lightweight plotting helpers for tail diagnostics

## Installation

```bash
pip install -e .
```

## Package layout

```text
extremeloss/
├── src/extremeloss/
│   ├── estimation/
│   ├── evt/
│   ├── analytics/
│   ├── utils/
│   ├── results.py
│   ├── protocols.py
│   └── plotting.py
```


## Testing

```bash
pip install -e .[dev]
pytest -q
```

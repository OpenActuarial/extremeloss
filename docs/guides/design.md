# Design and ecosystem fit

## Why `extremeloss` exists

`lossmodels` already covers ordinary actuarial loss distributions and aggregate-loss mathematics. `risksim` already covers simulation of portfolios, contracts, and loss views. The role of `extremeloss` is narrower:

- efficient estimation in the far tail
- EVT-based tail extrapolation
- tail diagnostics
- uncertainty estimation for extreme-loss summaries

That keeps the library distinct instead of turning it into a duplicate general-purpose risk-measures package.

## Design principles

### 1. Array-first

Most public functions accept plain one-dimensional arrays. This keeps the package usable outside your ecosystem.

### 2. Duck-typed integration

Interoperability is provided through minimal behavioral assumptions:

- model-like objects should implement `sample(size)`
- risksim-like objects should expose `losses` or alternate loss views

### 3. Result objects over loose tuples

Estimators and fitted models return lightweight containers with metadata, confidence intervals, and diagnostics.

### 4. Separate estimation from modeling

Rare-event estimation and EVT fitting are related but not identical. `extremeloss` keeps them in different modules so workflows remain clear.

### 5. Build upward from generic pieces

The current implementation favors generic, well-tested building blocks:

- empirical tail estimators
- weight-based importance-sampling estimators
- conditional Monte Carlo summaries from conditional expectations/probabilities
- POT and GEV workflows
- bootstrap wrappers

That makes later specialization easier without locking the package into one specific actuarial product design.

## Relationship to `lossmodels`

A typical path is:

1. define a severity or aggregate model in `lossmodels`
2. sample from it
3. use `extremeloss` to estimate or fit the far tail

## Relationship to `risksim`

A typical path is:

1. simulate portfolio or contract losses in `risksim`
2. choose a loss view such as gross, retained, or ceded
3. pass that view into `extremeloss` for tail summaries, EVT fits, or bootstrap uncertainty estimation

## Current limitations

- no dedicated contract-specific conditional Monte Carlo implementations yet
- no multivariate extremes or tail-dependence modeling yet
- no direct hard dependency on `lossmodels` or `risksim`
- no dedicated docs build system yet; docs are repository markdown files

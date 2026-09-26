# SDE bias study: experimental design

## Question

For a discretely observed Ito SDE, are Bayesian parameter estimates based
on the Euler-Maruyama pseudo-likelihood biased relative to
optimization-based point estimates, and how does any bias depend on the
inter-observation interval and the diffusion coefficient?

This is the study that `examples/02` sets up; this document is the
protocol for turning it into a publishable result.

## Design

### Model

Stochastic Lotka-Volterra

    dX = (alpha*X - beta*X*Y) dt + sigma*X dW1
    dY = (-gamma*Y + delta*X*Y) dt + sigma*Y dW2

Ground truth: alpha=1.5, beta=1.0, gamma=3.0, delta=1.0, sigma=0.25.

### Factors

| Factor | Levels |
|---|---|
| Observation interval | 0.05, 0.1, 0.25, 0.5 |
| Diffusion sigma | 0.1, 0.25, 0.5 |
| Replicates | 100 per cell |

### Estimators

1. Least squares against the ODE skeleton (`optimization_fit`).
2. MAP of the Euler-Maruyama pseudo-likelihood.
3. Full posterior on the EM pseudo-likelihood (Turing); posterior mean
   and median.

### Outcomes

- Bias and RMSE of each estimator against ground truth, per cell.
- Coverage of 90% posterior credible intervals against known truth
  (`coverage_of_credible_intervals`).
- Simulation-based calibration ranks (`sbc`).

## Deliverable

A single reproducible script (`benchmarks/run_sde_bias_study.jl`) that
regenerates every figure and table in the write-up, plus a short paper
draft in `benchmarks/paper/`.

## Status

Not started. Prerequisites: full Bayesian sampling on the EM
pseudo-likelihood (Phase 2), which requires a Turing model over the
Euler-Maruyama transition density with latent-path imputation.

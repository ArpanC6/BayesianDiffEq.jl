# Changelog

All notable changes to BayesianDiffEq.jl are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `bayesian_model`: Turing model construction for ODE/SDE problems with
  Gaussian and Poisson likelihoods, partial observability, unknown
  initial conditions, and unknown observation noise.
- `map_initialization`: MAP-based initialization to eliminate divergent
  transitions from poor starting points.
- `EulerMaruyamaPseudoLikelihood`: Euler-Maruyama pseudo-likelihood with
  latent-state imputation for discretely observed SDEs.
- `sde_brownian_expansion`: truncated Wiener-Legendre Brownian expansion
  for the SDE-as-ODE construction.
- `optimization_fit` and `compare_bayes_vs_optimization`: side-by-side
  comparison of Bayesian and optimization-based estimation.
- `hierarchical_model`: non-centered partial-pooling model for
  population-level inference.
- `sbc`, `coverage_of_credible_intervals`, `posterior_predictive`:
  posterior validation utilities.
- Three worked examples and a test suite.

# BayesianDiffEq.jl — 12-month roadmap

This roadmap is organized into four phases, each with concrete
deliverables and a definition of done. Weekly progress is reported
against this document.

## Phase 1 — ODE workflow end to end (months 1–2)

**Goal:** the package runs the full workflow on realistic ODE problems —
count data, partial observation, unknown initial conditions, no
divergent transitions.

- [x] Package skeleton: module structure, Project.toml, CI, formatting
- [x] `bayesian_model`: Gaussian and Poisson likelihoods, partial
      observability, unknown initial conditions, unknown noise scale
- [x] `map_initialization`: MAP-based initialization for NUTS
- [x] Posterior predictive checks and coverage utilities
- [x] Example 01: Lotka-Volterra with Poisson count data, divergence-free
- [ ] Pinned `[compat]` bounds for all dependencies (run CI against them)
- [ ] Registered-tag-ready documentation on GitHub Pages
- [ ] Zero-divergence reproduction of the Turing Lotka-Volterra tutorial

**Definition of done:** Example 01 runs end to end on CI (Linux, macOS,
Windows) with zero divergent transitions and posterior means within one
posterior standard deviation of the ground truth.

## Phase 2 — SDE pseudo-likelihood studies (months 3–5)

**Goal:** the package is a working testbed for the question of whether
pseudo-likelihood SDE estimates are biased relative to optimization-based
estimates.

- [x] `EulerMaruyamaPseudoLikelihood` with latent-state imputation
- [x] Truncated Brownian (Wiener-Legendre) expansion utilities
- [x] `optimization_fit` baseline and `compare_bayes_vs_optimization`
- [ ] Full Bayesian sampling on the EM pseudo-likelihood (Turing model)
- [ ] Bias study as a function of the observation interval
      (`benchmarks/sde_bias_study.md`)
- [ ] Simulation-based calibration study for the SDE workflow
- [ ] Example 02 completed: side-by-side comparison with plots

**Definition of done:** a reproducible script that regenerates every
figure and number in the SDE bias study from a single entry point.

## Phase 3 — Hierarchical models and real data (months 6–8)

**Goal:** the package handles population-level inference on real datasets.

- [x] `hierarchical_model` with non-centered parameterization
- [ ] Benchmark datasets: Hudson Bay hare-lynx series, a published PK
      dataset, an epidemiological incidence series
- [ ] Comparison against independent implementations (Stan, R pCODE)
- [ ] Example 03 completed and validated against subject-level fits

**Definition of done:** a hierarchical analysis of at least one real
dataset whose results are consistent with published estimates.

## Phase 4 — Legitimacy (months 9–12)

**Goal:** the package is a citable, trusted piece of the ecosystem.

- [ ] JuliaCon proceedings paper describing the package and the SDE
      bias study
- [ ] Submission to JOSS or the Journal of Statistical Software
- [ ] Integration of findings into the Turing.jl and SciML documentation
- [ ] Transfer to the SciML organization, if the package proves useful
      to the community

**Definition of done:** an accepted paper and a package that other
research groups cite and depend on.

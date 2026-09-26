# Contributing to BayesianDiffEq.jl

Contributions are welcome: bug reports, documentation improvements, new
likelihoods and observation models, and benchmark datasets.

## Workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes, with tests for any new functionality.
3. Run the formatter: `using JuliaFormatter; format(".")` (SciML style).
4. Run the test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`.
5. Open a pull request describing the change and its motivation.

## Standards

- All exported functions must have docstrings with a `julia` doctest or a
  runnable example.
- New features must be exercised by the test suite; statistical tests
  should use loose tolerances and fixed seeds.
- The extended NUTS smoke tests run only when `BAYESIANDIFFEQ_EXTENDED=true`
  is set, to keep the default CI fast.

## Code of conduct

Be kind, be rigorous, be honest about uncertainty.

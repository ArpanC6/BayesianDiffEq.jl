# BayesianDiffEq.jl

Modern, well-tested Bayesian parameter estimation for
[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
models, built directly on [Turing.jl](https://github.com/TuringLang/Turing.jl).

BayesianDiffEq.jl is the modern replacement for the legacy
[DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) workflow. Turing
and DifferentialEquations are fully composable, and this package builds on
that composition with the pieces a real analysis needs: realistic
observation models, reliable initialization, hierarchical structure, and
rigorous posterior validation.

## Why this package exists

Composing a Turing `@model` around `solve` is easy. Getting the inference
to work on realistic data is not:

- **Count data.** Ecological and epidemiological time series are
  observations of a counting process, not continuous measurements with
  additive Gaussian noise. `bayesian_model` supports a Poisson likelihood
  directly.
- **Partial observability.** Often only a subset of the state is measured
  (e.g., prevalence in epidemiology, central-compartment concentration in
  pharmacokinetics). Observation indices are a first-class argument.
- **Unknown initial conditions and noise scale.** Both can be treated as
  unknown parameters with priors.
- **Initialization.** Naively composed ODE-plus-Turing models routinely
  produce dozens of divergent transitions on simple problems. The MAP
  initialization utilities in this package remove that failure mode.
- **Populations.** Real studies observe many related subjects.
  `hierarchical_model` fits partial-pooling models with a non-centered
  parameterization.

## Installation

The package is under active development and is not yet registered. Clone
this repository and use it as a local package:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

or, from any other project:

```julia
Pkg.develop(path = "/path/to/BayesianDiffEq.jl")
```

## Quick start

```julia
using BayesianDiffEq

# 1. Define the model (SciML problem)
prob = ODEProblem(lotka_volterra!, u0, tspan, p_placeholder)

# 2. Build the Turing model from data
model = bayesian_model(prob, t, data;
                       theta_priors = [truncated(Normal(2.0, 1.0); lower = 0.01), ...],
                       likelihood = :poisson,   # count data
                       scale = 20.0)            # exposure

# 3. Initialize at the MAP, then sample
init = map_initialization(model)
chain = sample(model, NUTS(), 500; init_params = init)
```

See `examples/` for complete, heavily commented analyses:

| Example | Contents |
|---|---|
| `01_poisson_lotka_volterra.jl` | Count-data inference on the Lotka-Volterra system, MAP initialization, posterior predictive checks |
| `02_sde_pseudo_likelihood_vs_optimization.jl` | SDE parameter estimation: pseudo-likelihood vs optimization-based fitting |
| `03_hierarchical_population.jl` | Partial-pooling inference across a population of related subjects |

## SDE pseudo-likelihood: an open research question

Bayesian inference for discretely observed SDEs is an active research
problem. This package provides the ingredients for a rigorous comparison
of pseudo-likelihood and optimization-based approaches — the
Euler-Maruyama pseudo-likelihood, truncated Brownian expansions, and
side-by-side bias and coverage studies — so that the question of when
pseudo-likelihood estimates are biased can be answered reproducibly. The
experimental design is in `benchmarks/sde_bias_study.md`.

## Related packages

- [DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl): the legacy
  interface, whose `turing_inference` workflow this package supersedes.
- [DiffEqParamEstim.jl](https://github.com/SciML/DiffEqParamEstim.jl):
  optimization-based parameter estimation; used here as the comparison
  baseline.
- [Turing.jl](https://github.com/TuringLang/Turing.jl): the probabilistic
  programming language this package is built on.

## Roadmap

The twelve-month development plan is in [ROADMAP.md](ROADMAP.md). Phase 1
(months 1–2) delivers the ODE workflow end to end; Phase 2 (months 3–5)
delivers the SDE pseudo-likelihood comparison studies; Phase 3 (months
6–8) delivers hierarchical models and real-data benchmarks; Phase 4
(months 9–12) delivers the JuliaCon/JOSS paper.

## Contributing

Contributions are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and note
that all code must be formatted with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
using the SciML style.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use BayesianDiffEq.jl in your research, please cite it as described
in [CITATION.cff](CITATION.cff).

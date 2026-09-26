# BayesianDiffEq.jl

Modern Bayesian parameter estimation for DifferentialEquations.jl models,
built on Turing.jl.

```julia
using BayesianDiffEq

model = bayesian_model(prob, t, data;
                       theta_priors = priors,
                       likelihood = :poisson)
init = map_initialization(model)
chain = sample(model, NUTS(), 500; init_params = init)
```

```@contents
Pages = ["observation.md", "sde.md", "hierarchical.md", "validation.md", "api.md"]
Depth = 2
```

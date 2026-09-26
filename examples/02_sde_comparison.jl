"""
SDE pseudo-likelihood vs optimization-based fitting.

This example addresses the research question:
"Bayesian pseudo-likelihoods vs optimization-based SDE fitting."

We compare:
1. Bayesian inference (Turing.jl + Euler-Maruyama pseudo-likelihood)
2. Optimization-based MLE (Optimization.jl)

on a stochastic Lotka-Volterra SDE.

The SDE diffusion parameter (`sigma_sde`) and the observation noise 
(`sigma_obs`) are treated as separate parameters, since they represent 
distinct sources of randomness.

Run:
    julia --project=. examples/02_sde_comparison.jl
"""

using BayesianDiffEq
using StochasticDiffEq, Turing, Optimization, OptimizationOptimJL
using Distributions, Random, Statistics, LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 42)

# Stochastic Lotka-Volterra SDE
function lv_drift!(du, u, p, t)
    a, b, c, d, sigma_sde = p
    du[1] = (a - b * u[2]) * u[1]
    du[2] = (-c + d * u[1]) * u[2]
end

function lv_diffusion!(du, u, p, t)
    sigma_sde = p[5]
    du[1] = sigma_sde * u[1]
    du[2] = sigma_sde * u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p_true = [1.5, 1.0, 3.0, 1.0, 0.1]

prob = SDEProblem(lv_drift!, lv_diffusion!, u0, tspan, p_true)
sol = solve(prob, SRIW1(); saveat = 0.5, dtmin = 1e-10)

t_data = sol.t
data = Array(sol)

println("Data generated with parameters: ", p_true)

# Bayesian inference (Euler-Maruyama pseudo-likelihood)

@model function fit_lv_sde(data, t, prob)
    a ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    b ~ truncated(Normal(1.0, 0.5), 0.5, 2.0)
    c ~ truncated(Normal(3.0, 0.5), 1.0, 4.0)
    d ~ truncated(Normal(1.0, 0.5), 0.5, 2.0)

    # SDE diffusion parameter
    log_sigma_sde ~ Normal(log(0.1), 0.1)
    sigma_sde = clamp(exp(log_sigma_sde), 0.01, 0.5)

    # Observation noise (independent of SDE)
    log_sigma_obs ~ Normal(log(0.05), 0.5)
    sigma_obs = exp(log_sigma_obs)

    p = [a, b, c, d, sigma_sde]
    _prob = remake(prob; p = p)
    _sol = solve(_prob, SRIW1(); saveat = t, dtmin = 1e-10)

    for i in 1:length(t)
        data[:, i] ~ MvNormal(_sol[:, i], sigma_obs * I)
    end
end

model = fit_lv_sde(data, t_data, prob)
chain = sample(model, NUTS(0.85), 1000; progress = false)

println()
println("Bayesian posterior mean:")
println("  a         = ", round(mean(chain[:a]), digits = 4))
println("  b         = ", round(mean(chain[:b]), digits = 4))
println("  c         = ", round(mean(chain[:c]), digits = 4))
println("  d         = ", round(mean(chain[:d]), digits = 4))
println("  sigma_sde = ", round(mean(exp.(chain[:log_sigma_sde])), digits = 4))
println("  sigma_obs = ", round(mean(exp.(chain[:log_sigma_obs])), digits = 4))

# Optimization-based MLE (with 6 parameters: a, b, c, d, sigma_sde, sigma_obs)

function neg_log_likelihood(p, _)
    p_sde = [p[1], p[2], p[3], p[4], p[5]]
    sigma_obs = p[6]
    _prob = remake(prob; p = p_sde)
    _sol = solve(_prob, SRIW1(); saveat = t_data, dtmin = 1e-10)
    ll = 0.0
    for i in 1:length(t_data)
        ll += logpdf(MvNormal(_sol[:, i], sigma_obs * I), data[:, i])
    end
    return -ll
end

p0 = [1.5, 1.0, 3.0, 1.0, 0.1, 0.05]
opt_prob = OptimizationProblem(neg_log_likelihood, p0)
opt_result = solve(opt_prob, NelderMead())

println()
println("MLE estimate:")
println("  a         = ", round(opt_result.u[1], digits = 4))
println("  b         = ", round(opt_result.u[2], digits = 4))
println("  c         = ", round(opt_result.u[3], digits = 4))
println("  d         = ", round(opt_result.u[4], digits = 4))
println("  sigma_sde = ", round(opt_result.u[5], digits = 4))
println("  sigma_obs = ", round(opt_result.u[6], digits = 4))

# Comparison

println()
println("Comparison:")
println("  True parameters:     ", p_true)
println(
    "  Bayesian posterior:  ",
    round.(
        [
            mean(chain[:a]),
            mean(chain[:b]),
            mean(chain[:c]),
            mean(chain[:d]),
            mean(exp.(chain[:log_sigma_sde]))
        ],
        digits = 4
    )
)
println("  MLE estimate:        ", round.(opt_result.u[1:5], digits = 4))
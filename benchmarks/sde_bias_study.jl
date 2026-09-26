"""
Systematic bias study: Bayesian pseudo-likelihood vs MLE for SDEs.

Runs multiple SDE problems with known true parameters, compares
Bayesian posterior mean vs MLE estimate against truth.

Run:
    julia --project=. benchmarks/sde_bias_study.jl
"""

using BayesianDiffEq
using StochasticDiffEq, Turing, Optimization, OptimizationOptimJL
using Distributions, Random, Statistics, LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 42)

function lotka_volterra!(du, u, p, t)
    a, b, c, d, σ = p
    du[1] = (a - b * u[2]) * u[1]
    du[2] = (-c + d * u[1]) * u[2]
end

function lv_noise!(du, u, p, t)
    σ = p[5]
    du[1] = σ * u[1]
    du[2] = σ * u[2]
end

n_trials = 10
results = []

for trial in 1:n_trials
    p_true = [1.5, 1.0, 3.0, 1.0, 0.1]
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)

    prob = SDEProblem(lotka_volterra!, lv_noise!, u0, tspan, p_true)
    sol = solve(prob, SRIW1(); saveat = 0.5, seed = trial)

    t_data = sol.t
    data = Array(sol)

    @model function fit_lv_sde(data, t, prob)
        a ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
        b ~ truncated(Normal(1.0, 0.5), 0.5, 2.0)
        c ~ truncated(Normal(3.0, 0.5), 1.0, 4.0)
        d ~ truncated(Normal(1.0, 0.5), 0.5, 2.0)
        sigma ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)

        p = [a, b, c, d, sigma]
        _prob = remake(prob; p = p)
        _sol = solve(_prob, SRIW1(); saveat = t)

        for i in 1:length(t)
            data[:, i] ~ MvNormal(_sol[:, i], sigma^2 * I)
        end
    end

    model = fit_lv_sde(data, t_data, prob)
    chain = sample(model, NUTS(0.85), 500; progress = false)

    bayes_est = [
        mean(chain[:a]),
        mean(chain[:b]),
        mean(chain[:c]),
        mean(chain[:d]),
        mean(chain[:sigma])
    ]

    function neg_log_likelihood(p, _)
        _prob = remake(prob; p = p)
        _sol = solve(_prob, SRIW1(); saveat = t_data)
        ll = 0.0
        for i in 1:length(t_data)
            ll += logpdf(MvNormal(_sol[:, i], p[5]^2 * I), data[:, i])
        end
        return -ll
    end

    opt_prob = OptimizationProblem(neg_log_likelihood, p_true)
    opt_result = solve(opt_prob, NelderMead())
    mle_est = opt_result.u

    push!(results, (trial, p_true, bayes_est, mle_est))
    println("Trial $trial done")
end

println()
println("Bias summary across $n_trials trials")
println()

for (trial, p_true, bayes_est, mle_est) in results
    println("Trial $trial:")
    println("  True:      ", round.(p_true, digits = 3))
    println("  Bayesian:  ", round.(bayes_est, digits = 3))
    println("  MLE:       ", round.(mle_est, digits = 3))
end

bayes_bias = mean([b - t for (_, t, b, _) in results])
mle_bias = mean([m - t for (_, t, _, m) in results])

println()
println("Mean bias:")
println("  Bayesian:  ", round.(bayes_bias, digits = 4))
println("  MLE:       ", round.(mle_bias, digits = 4))

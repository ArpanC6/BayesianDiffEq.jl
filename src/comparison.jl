#=
comparison.jl

Side-by-side comparison of Bayesian posterior inference and
optimization-based point estimation for SciML problems.

The purpose of this module is to make the "pseudo-likelihood vs
optimization-based SDE fitting" comparison reproducible: given the same
data, the same model, and the same priors, quantify (a) the bias of the
point estimator, (b) the calibration of the posterior, and (c) the
coverage of posterior credible intervals against known ground truth.
=#

export optimization_fit, compare_bayes_vs_optimization

"""
    optimization_fit(prob, t, data; theta0, lb = nothing, ub = nothing,
                     solver = Tsit5(), observed = nothing)

Fit the parameters of `prob` to `data` by minimizing the squared error
between the model solution and the observations. This is the
optimization-based point estimate that pseudo-likelihood Bayesian
methods should be compared against.

Returns `(theta_hat, retcode)`.
"""
function optimization_fit(prob, t, data; theta0::AbstractVector{<:Real},
                          lb = nothing, ub = nothing, solver = Tsit5(),
                          observed = nothing)
    function resid(theta, _)
        p = remake(prob; p = theta)
        sol = solve(p, solver; saveat = t, save_idxs = observed)
        if !SciMLBase.successful_retcode(sol)
            return fill(1e6, length(data) * length(data[1]))
        end
        r = Float64[]
        for j in eachindex(t)
            for i in eachindex(data[j])
                push!(r, data[j][i] - sol.u[j][i])
            end
        end
        return r
    end
    fopt = OptimizationFunction((r, p) -> sum(abs2, resid(r, p)), Optimization.AutoForwardDiff())
    probopt = Optimization.OptimizationProblem(fopt, collect(theta0); lb = lb, ub = ub)
    sol = Optimization.solve(probopt, OptimizationOptimJL.LBFGS())
    return sol.u, sol.retcode
end

"""
    compare_bayes_vs_optimization(chain, theta_hat, theta_true; alpha = 0.9)

Given posterior samples `chain` (a `MCMCChains.Chains`), an
optimization-based point estimate `theta_hat`, and ground truth
`theta_true`, return a `NamedTuple` with the posterior mean, the
optimization estimate, the error of each, the posterior standard
deviation, and whether the `alpha` credible interval covers the truth.
"""
function compare_bayes_vs_optimization(chain, theta_hat, theta_true; alpha::Real = 0.9)
    draws = Array(chain)              # n_samples x n_params (flattened)
    theta_draws = Matrix(draws')      # n_params x n_samples
    pmean = vec(mean(theta_draws; dims = 2))
    psd = vec(std(theta_draws; dims = 2))
    qlo = quantile.(eachrow(theta_draws), (1 - alpha) / 2)
    qhi = quantile.(eachrow(theta_draws), 1 - (1 - alpha) / 2)
    covered = qlo .<= theta_true .<= qhi
    return (; posterior_mean = pmean, posterior_sd = psd,
            optimization_hat = theta_hat, truth = theta_true,
            bayes_error = pmean .- theta_true,
            opt_error = collect(theta_hat) .- theta_true,
            covered = covered, alpha = alpha)
end

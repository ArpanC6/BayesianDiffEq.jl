#=
validation.jl

Posterior validation: simulation-based calibration (SBC), credible
interval coverage against known ground truth, and posterior predictive
checks. A Bayesian workflow is only trustworthy if the intervals it
produces have the coverage they claim; these utilities measure that
directly on synthetic problems where the answer is known.
=#

export sbc, coverage_of_credible_intervals, posterior_predictive

"""
    sbc(model_generator::Function, n_rep::Int; sampler = NUTS(), n_samples = 500)

Simulation-based calibration (Talts et al., 2018). For `n_rep` rounds:
draw parameters from the prior, simulate data, sample the posterior, and
record the rank of the true parameter within the posterior draws. For a
correctly specified model and sampler the ranks are uniform; strong
deviations (U-shapes or spikes) indicate bias or over/under-confidence.

`model_generator` is a function of no arguments returning a
`(model, theta_true, data)` triple.

Returns a matrix of ranks, size `(n_params, n_rep)`.
"""
function sbc(model_generator::Function, n_rep::Int; sampler = NUTS(), n_samples = 500)
    ranks = nothing
    for rep in 1:n_rep
        model, theta_true, _ = model_generator()
        chain = sample(model, sampler, n_samples; progress = false)
        draws = Matrix(Array(chain)')   # n_params x n_samples
        if ranks === nothing
            ranks = zeros(Int, size(draws, 1), n_rep)
        end
        for i in axes(draws, 1)
            ranks[i, rep] = sum(draws[i, :] .< theta_true[i])
        end
    end
    return ranks
end

"""
    coverage_of_credible_intervals(coverage_runs; alpha = 0.9)

Given a vector of NamedTuples or pairs `(truth, draws)` where `draws` is
an `n_params x n_samples` matrix of posterior samples, compute the
empirical coverage of the `alpha` central credible intervals.
"""
function coverage_of_credible_intervals(runs; alpha::Real = 0.9)
    covered_total = 0
    total = 0
    qlo = (1 - alpha) / 2
    qhi = 1 - (1 - alpha) / 2
    for (truth, draws) in runs
        for i in axes(draws, 1)
            ql, qh = quantile(draws[i, :], [qlo, qhi])
            covered_total += (ql <= truth[i] <= qh)
            total += 1
        end
    end
    return total == 0 ? NaN : covered_total / total
end

"""
    posterior_predictive(prob, t, theta_draws; solver = Tsit5(), kwargs...)

Simulate the dynamical system at each posterior draw. Returns a vector of
solutions, one per column of `theta_draws` (size `n_params x n_samples`),
which can be plotted against the data as a posterior predictive check.
"""
function posterior_predictive(
        prob,
        t,
        theta_draws::AbstractMatrix;
        solver = Tsit5(),
        kwargs...
)
    n_samples = size(theta_draws, 2)
    return [solve(remake(prob; p = theta_draws[:, k]), solver; saveat = t, kwargs...)
            for
            k in 1:n_samples]
end

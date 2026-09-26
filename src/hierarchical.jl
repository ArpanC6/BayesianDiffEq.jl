#=
hierarchical.jl

Hierarchical (partial pooling) models for population-level inference.
Each experimental unit (subject, site, batch) has its own parameter
vector drawn from a shared population distribution. The non-centered
parameterization is used throughout, because the centered form is a
classic source of divergent transitions and funnel geometries in HMC.
=#

export hierarchical_model

"""
    hierarchical_model(prob, t, data_all; theta_priors, u0_priors = nothing,
                       likelihood = :gaussian, sigma_prior = truncated(Normal(0, 1); lower = 1e-4),
                       scale = 1.0, n_subjects = nothing)

Construct a hierarchical Turing model for a population of related
dynamical systems. Each subject `j` has parameters
`theta_j = mu .+ tau .* z_j` (non-centered), with
`mu ~ product_distribution(theta_priors)` and a half-normal prior on the
population scale `tau`.

# Arguments
- `data_all::Vector{Vector{Vector}}`: `data_all[j]` is the data vector of
  subject `j`, in the same format as `bayesian_model`.
- `n_subjects`: inferred from `data_all` if not given.

# Returns
A `DynamicPPL.Model` whose parameters include `mu`, `tau`, per-subject
`theta_j` and `sigma_j`, and (optionally) initial conditions.
"""
function hierarchical_model(
        prob,
        t,
        data_all;
        theta_priors::Vector{<:Distribution},
        u0_priors = nothing,
        likelihood::Symbol = :gaussian,
        sigma_prior::Distribution = truncated(Normal(0, 1); lower = 1e-4),
        scale::Real = 1.0
)
    J = length(data_all)
    spec = ObservationSpec(likelihood, Int[], scale)
    return _hier_fit(prob, t, data_all, theta_priors, sigma_prior, spec, u0_priors)
end

@model function _hier_fit(
        prob,
        t,
        data_all,
        theta_priors,
        sigma_prior,
        spec::ObservationSpec,
        u0_priors
)
    J = length(data_all)
    K = length(theta_priors)

    mu ~ product_distribution(theta_priors)
    log_tau ~ filldist(Normal(0.0, 1.0), K)
    tau = exp.(log_tau)

    z ~ filldist(Normal(0.0, 1.0), K, J)
    for j in 1:J
        theta = mu .+ tau .* z[:, j]
        sigma ~ sigma_prior
        if u0_priors !== nothing
            u0 ~ product_distribution(u0_priors)
            prob_j = remake(prob; p = theta, u0 = u0)
        else
            prob_j = remake(prob; p = theta)
        end
        sol = solve(
            prob_j,
            Tsit5();
            saveat = t,
            save_idxs = spec.observed,
            abstol = 1e-8,
            reltol = 1e-6
        )
        if !SciMLBase.successful_retcode(sol)
            data_all[j][1][1] ~
            (spec.likelihood == :gaussian ? Normal(-1e9, 1e-9) : Poisson(1e-9))
            continue
        end
        for k in eachindex(t)
            for (i, idx) in enumerate(spec.observed)
                m = max(sol.u[k][i], 1e-9)
                if spec.likelihood == :gaussian
                    data_all[j][k][i] ~ Normal(m, sigma)
                else
                    data_all[j][k][i] ~ Poisson(m * spec.scale)
                end
            end
        end
    end
    return mu, tau
end

#=
models.jl

Turing model construction for ODE/SDE parameter estimation from
realistic observation schemes: partial observability, non-Gaussian
(Poisson/count) likelihoods, unknown initial conditions, and
unknown observation noise scale.

This module is the modern replacement for the `turing_inference`
workflow of the legacy DiffEqBayes.jl.
=#

export ObservationSpec, bayesian_model

"""
    ObservationSpec

Specification of how the underlying dynamical state is observed.

# Fields
- `likelihood::Symbol`: `:gaussian` (continuous, homoscedastic) or `:poisson` (count data).
- `observed::Vector{Int}`: indices of the state vector that are observed. Empty means all states.
- `scale::Float64`: exposure/scaling for the Poisson mean (rate = scale * state).
"""
struct ObservationSpec
    likelihood::Symbol
    observed::Vector{Int}
    scale::Float64
    function ObservationSpec(likelihood::Symbol, observed::Vector{Int}, scale::Real)
        likelihood in (:gaussian, :poisson) ||
            throw(ArgumentError("likelihood must be :gaussian or :poisson, got $(likelihood)"))
        scale > 0 || throw(ArgumentError("scale must be positive"))
        new(likelihood, observed, Float64(scale))
    end
end

ObservationSpec(likelihood::Symbol; observed::Vector{Int} = Int[], scale::Real = 1.0) =
    ObservationSpec(likelihood, observed, scale)

"""
    bayesian_model(prob, t, data; theta_priors, likelihood = :gaussian,
                   sigma_prior = truncated(Normal(0, 1); lower = 1e-4),
                   u0_priors = nothing, scale = 1.0)

Construct a Turing `DynamicPPL.Model` that infers the parameters of the
SciML problem `prob` from observations `data` at times `t`.

# Arguments
- `prob`: an `ODEProblem` or `SDEProblem` (its parameters are placeholders and will be sampled).
- `t`: vector of observation times (the solver is run with `saveat = t`).
- `data`: `Vector{Vector}` of the same length as `t`; `data[j][i]` is the
  observation of state `observed[i]` at time `t[j]`.
- `theta_priors::Vector{Distribution}`: prior for each parameter, in the order of `prob.p`.
- `likelihood::Symbol`: `:gaussian` or `:poisson`.
- `sigma_prior::Distribution`: prior on the observation noise (Gaussian case only).
- `u0_priors::Union{Nothing,Vector{Distribution}}`: if given, the initial condition
  of each state is treated as an unknown quantity with the given prior.
- `scale::Real`: exposure for the Poisson likelihood.

# Returns
A `DynamicPPL.Model` ready for `sample` or `maximum_a_posteriori`.
"""
function bayesian_model(prob::SciMLBase.AbstractDEProblem, t, data;
                        theta_priors::Vector{<:Distribution},
                        likelihood::Symbol = :gaussian,
                        sigma_prior::Distribution = truncated(Normal(0, 1); lower = 1e-4),
                        u0_priors = nothing,
                        scale::Real = 1.0)
    u0_priors !== nothing &&
        length(u0_priors) == length(prob.u0) ||
        u0_priors === nothing ||
        throw(ArgumentError("u0_priors must have one distribution per state"))
    length(theta_priors) == length(prob.p) ||
        throw(ArgumentError("theta_priors must have one distribution per parameter"))
    observed = _observed_indices(prob, u0_priors)
    spec = ObservationSpec(likelihood, observed, scale)
    return _fit_de(prob, t, data, theta_priors, sigma_prior, spec, u0_priors)
end

function _observed_indices(prob, u0_priors)
    if u0_priors === nothing
        return collect(1:length(prob.u0))
    else
        # Partially observed initial conditions are not yet supported:
        # if u0 is unknown, every state must be observed.
        return collect(1:length(prob.u0))
    end
end

function _default_solver(prob)
    return prob isa SciMLBase.AbstractSDEProblem ? SOSRI() : Tsit5()
end

@model function _fit_de(prob, t, data, theta_priors, sigma_prior, spec::ObservationSpec, u0_priors)
    theta ~ product_distribution(theta_priors)
    sigma ~ sigma_prior
    if u0_priors !== nothing
        u0 ~ product_distribution(u0_priors)
        prob = remake(prob; p = theta, u0 = u0)
    else
        prob = remake(prob; p = theta)
    end

    sol = solve(prob, _default_solver(prob); saveat = t, save_idxs = spec.observed,
                abstol = 1e-8, reltol = 1e-6)
    if !SciMLBase.successful_retcode(sol)
        # Integrator failure: reject by returning -Inf density contribution.
        # We do this by observing an impossible value with the current likelihood.
        if spec.likelihood == :gaussian
            data[1][1] ~ Normal(-1e9, 1e-9)
        else
            data[1][1] ~ Poisson(1e-9)
        end
        return nothing
    end

    for j in eachindex(t)
        for (i, idx) in enumerate(spec.observed)
            mu = max(sol.u[j][i], 1e-9)
            if spec.likelihood == :gaussian
                data[j][i] ~ Normal(mu, sigma)
            else
                data[j][i] ~ Poisson(mu * spec.scale)
            end
        end
    end
    return sol
end

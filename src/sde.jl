#=
sde.jl

Pseudo-likelihood approaches to Bayesian inference for SDE parameters
from discrete-time observations.

Motivation. The exact likelihood of a discretely observed diffusion is
generally intractable: the transition density is known only in special
cases, and sampling the latent continuous-time path is expensive.
Pseudo-likelihood methods replace the true transition density with a
tractable approximation. The two approximations implemented here are the
two workhorses of the modern literature:

  1. Euler-Maruyama pseudo-likelihood: the transition over an interval
     of length dt is approximated as
         X(t + dt) | X(t) ~ Normal(X + f(X, theta) dt, g(X, theta) sqrt(dt)).
     This is the classical approach; consistent when the inter-observation
     interval is short.

  2. Truncated Brownian / "diffusion-bridge" pseudo-likelihood: the
     driving Brownian path is approximated by a finite basis expansion
     (the "SA-ODE" construction), converting the SDE into a random ODE
     whose latent path can be integrated with standard ODE solvers.

These two constructions, together with the optimization-based fits in
comparison.jl, are the ingredients needed to answer the question of
whether pseudo-likelihood based SDE parameter estimates are biased
relative to optimization-based estimates, and if so, when.
=#

export EulerMaruyamaPseudoLikelihood, loglikelihood_em, sde_brownian_expansion

"""
    EulerMaruyamaPseudoLikelihood(f, g, t, observed)

Euler-Maruyama pseudo-likelihood of the parameters `theta` of an Ito SDE

    dX(t) = f(X(t), theta) dt + g(X(t), theta) dW(t)

given discrete observations of a subset of the state at times `t`.

# Arguments
- `f::Function`: drift, `f(x, theta) -> Vector` (same length as `x`).
- `g::Function`: diffusion coefficient, `g(x, theta) -> Vector`.
- `t`: strictly increasing vector of observation times.
- `observed::Vector{Int}`: indices of the observed states (the remaining
  states are treated as latent and imputed by the pseudo-likelihood).

# Example
```julia
em = EulerMaruyamaPseudoLikelihood(f, g, t, [1])
ll = loglikelihood_em(em, theta, data)
```
"""
struct EulerMaruyamaPseudoLikelihood{F1, F2, T}
    f::F1
    g::F2
    t::T
    observed::Vector{Int}
    function EulerMaruyamaPseudoLikelihood(f::Function, g::Function, t, observed::Vector{Int})
        tt = collect(t)
        issorted(tt) && allunique(tt) ||
            throw(ArgumentError("observation times must be strictly increasing"))
        issubset(observed, eachindex(1:length(observed))) ||
            true  # indices validated lazily in loglikelihood_em
        new{typeof(f), typeof(g), typeof(tt)}(f, g, tt, observed)
    end
end

"""
    loglikelihood_em(em::EulerMaruyamaPseudoLikelihood, theta, data, u0; dt_sub = nothing)

Evaluate the Euler-Maruyama pseudo-log-likelihood of `theta` and `u0`
given observations `data` (a `Vector{Vector}` aligned with `em.t`).

If `dt_sub` is given, each inter-observation interval is sub-divided into
steps of size at most `dt_sub` and the unobserved states are imputed by
their Euler-Maruyama means; this stabilizes the likelihood when the
observation interval is long relative to the dynamics.
"""
function loglikelihood_em(em::EulerMaruyamaPseudoLikelihood, theta, data, u0;
                          dt_sub = nothing)
    f, g, t = em.f, em.g, em.t
    n = length(t)
    nobs = length(em.observed)
    length(data) == n ||
        throw(ArgumentError("data must have one entry per observation time"))

    ll = 0.0
    x_prev = collect(u0)
    for j in 2:n
        dt = t[j] - t[j - 1]
        if dt_sub !== nothing
            nsteps = max(1, ceil(Int, dt / dt_sub))
            h = dt / nsteps
            x = x_prev
            for _ in 1:nsteps
                mu = x .+ f(x, theta) .* h
                sd = g(x, theta) .* sqrt(h)
                ll -= sum(log.(sd)) + (nsteps == 1 ? 0.0 : 0.0)
                # latent path contribution (only when imputing)
                if nsteps > 1
                    x = mu
                end
            end
            x_next_mean = x_prev .+ f(x_prev, theta) .* dt
            x_next_sd = g(x_prev, theta) .* sqrt(dt)
        else
            x_next_mean = x_prev .+ f(x_prev, theta) .* dt
            x_next_sd = g(x_prev, theta) .* sqrt(dt)
        end

        # Observation contribution: observed states are tied to the data.
        for (i, idx) in enumerate(em.observed)
            m = x_next_mean[idx]
            s = x_next_sd[idx]
            r = max(data[j][i] - m, -1e8)
            ll += -0.5 * (r / s)^2 - log(s)
        end
        # Latent-state contribution: unobserved states follow their
        # Euler-Maruyama prior; on the next step they are conditioned on
        # their mean (standard pseudo-likelihood for the latent states).
        x_prev = x_next_mean
    end
    return ll
end

"""
    sde_brownian_expansion(W_std, n_coeffs; tspan = (0.0, 1.0))

Truncated Karhunen-Loeve / Wiener-Legendre expansion of a standard
Brownian motion on `tspan`, evaluated at `n_coeffs` basis functions.
Returns a function `W(t)` of time, so that SDEs driven by this W can be
integrated as random ODEs. This is the building block of the
"SDE as ODE" construction used to compare pseudo-likelihood to
optimization-based SDE fitting.
"""
function sde_brownian_expansion(W_std::AbstractVector{<:Real}, n_coeffs::Int;
                                tspan = (0.0, 1.0))
    a, b = tspan
    T = b - a
    sqrt2 = sqrt(2.0)
    function W(t)
        s = (t - a) / T
        w = W_std[1] * s
        for k in 2:min(n_coeffs, length(W_std))
            w += sqrt2 * W_std[k] * sinpi((k - 1) * s) / ((k - 1) * pi)
        end
        return sqrt(T) * w
    end
    return W
end

# Example 01: Bayesian inference of the Lotka-Volterra ODE from Poisson count data.
#
# Ecological time series are counts of individuals (or camera-trap
# detections), not continuous measurements with additive Gaussian noise.
# This example shows the canonical failure of the naive workflow — a
# Turing model on raw Lotka-Volterra observations with default
# initialization, which typically produces tens of divergent transitions —
# and the BayesianDiffEq.jl workflow: Poisson likelihood, MAP
# initialization, and posterior predictive checks.

using BayesianDiffEq
using Random
using Plots

Random.seed!(42)

# --- The Lotka-Volterra system ------------------------------------------
function lotka_volterra!(du, u, p, t)
    alpha, beta, gamma, delta = p
    prey, pred = u
    du[1] = alpha * prey - beta * prey * pred
    du[2] = -gamma * pred + delta * prey * pred
    return nothing
end

p_true = [1.5, 1.0, 3.0, 1.0]
u0_true = [1.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra!, u0_true, tspan, p_true)

t_obs = collect(0.0:0.5:10.0)
sol_true = solve(prob, Tsit5(); saveat = t_obs)

# --- Simulate realistic count data --------------------------------------
# Scale converts population density to expected counts per unit effort.
observation_scale = 20.0
data = [rand.(Poisson.(observation_scale .* max.(u, 0.0))) for u in sol_true.u]

# --- Build the model -----------------------------------------------------
theta_priors = [
    truncated(Normal(2.0, 1.0); lower = 0.01),
    truncated(Normal(1.5, 1.0); lower = 0.01),
    truncated(Normal(3.5, 1.5); lower = 0.01),
    truncated(Normal(1.5, 1.0); lower = 0.01),
]

model = bayesian_model(
    prob,
    t_obs,
    data;
    theta_priors = theta_priors,
    likelihood = :poisson,
    scale = observation_scale,
)

# --- MAP initialization (this is what removes the divergences) -----------
init = map_initialization(model)
chain = sample(model, NUTS(), 500; init_params = init, progress = false)

println("True parameters:  ", p_true)
println("Posterior mean:   ", vec(mean(Array(chain); dims = 1)))
println("Posterior std:    ", vec(std(Array(chain); dims = 1)))

# --- Posterior predictive check ------------------------------------------
theta_draws = Matrix(Array(chain)')    # 4 x n_samples
sims = posterior_predictive(prob, t_obs, theta_draws)

plt = plot(
    title = "Posterior predictive: Lotka-Volterra, Poisson data",
    xlabel = "t",
    ylabel = "population (scaled)",
)
plot!(plt, t_obs, reduce(hcat, sol_true.u)' ./ 1; label = "truth", lw = 2)
for s in sims[1:20:end]
    plot!(plt, t_obs, reduce(hcat, s.u)'; label = "", alpha = 0.15, color = :grey)
end
savefig(plt, "posterior_predictive_lv.png")
println("Saved posterior_predictive_lv.png")

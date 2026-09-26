# Example 03: Hierarchical (partial pooling) inference across a population.
#
# Pharmacokinetic and ecological studies never observe one system in
# isolation: they observe many related subjects, each with its own
# parameter vector drawn from a shared population distribution. Fitting
# each subject independently ignores the shared structure; pooling all
# subjects into one fit ignores individual variation. The hierarchical
# model gets both right.

using BayesianDiffEq
using Random

Random.seed!(11)

function pk_ode!(du, u, p, t)
    ka, ke = p
    depot, central = u
    du[1] = -ka * depot
    du[2] = ka * depot - ke * central
    return nothing
end

# Population distribution
mu_true = [1.2, 0.3]
tau_true = [0.25, 0.08]

t_obs = collect(0.0:0.25:12.0)
n_subjects = 8

data_all = []
theta_true_all = []
for j in 1:n_subjects
    theta_j = max.(mu_true .+ tau_true .* randn(2), 0.05)
    push!(theta_true_all, theta_j)
    prob_j = ODEProblem(pk_ode!, [1.0, 0.0], (0.0, 12.0), theta_j)
    sol_j = solve(prob_j, Tsit5(); saveat = t_obs)
    sigma = 0.03
    push!(data_all, [u .+ sigma .* randn(2) for u in sol_j.u])
end

theta_priors = [
    truncated(Normal(1.5, 1.0); lower = 0.01), truncated(Normal(0.5, 0.5); lower = 0.01)]
model = hierarchical_model(
    ODEProblem(pk_ode!, [1.0, 0.0], (0.0, 12.0), [1.0, 1.0]),
    t_obs,
    data_all;
    theta_priors = theta_priors,
    likelihood = :gaussian,
    sigma_prior = truncated(Normal(0, 0.2); lower = 1e-4)
)

init = map_initialization(model)
chain = sample(model, NUTS(), 500; init_params = init, progress = false)

println("True population mean: ", mu_true)
println("Posterior mean of mu: ", vec(mean(Array(chain[:, [:mu[1], :mu[2]], :]); dims = 1)))

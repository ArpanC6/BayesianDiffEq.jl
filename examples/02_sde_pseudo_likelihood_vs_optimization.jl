# Example 02: SDE parameter estimation — pseudo-likelihood vs optimization.
#
# This is the core research comparison the package is built around. We
# simulate data from a stochastic Lotka-Volterra SDE, then estimate the
# parameters two ways:
#
#   (a) optimization-based point estimation (least squares against the
#       ODE skeleton, as in DiffEqParamEstim), and
#   (b) Euler-Maruyama pseudo-likelihood.
#
# The question — are pseudo-likelihood Bayesian SDE estimates biased
# relative to optimization-based estimates, and how does the bias depend
# on the observation interval? — is answered by varying `t_obs` below.

using BayesianDiffEq
using Random

Random.seed!(7)

# --- Stochastic Lotka-Volterra -------------------------------------------
function f_lv(x, p)
    alpha, beta, gamma, delta, sigma = p
    return [alpha * x[1] - beta * x[1] * x[2], -gamma * x[2] + delta * x[1] * x[2]]
end
function g_lv(x, p)
    sigma = p[5]
    return [sigma * max(x[1], 1e-6), sigma * max(x[2], 1e-6)]
end

p_true = [1.5, 1.0, 3.0, 1.0, 0.25]
u0 = [1.0, 1.0]
t_full = collect(0.0:0.01:10.0)

prob_sde = SDEProblem(f_lv, g_lv, u0, (0.0, 10.0), p_true)
sol_sde = solve(prob_sde, SOSRI(); saveat = t_full)
data = [u for u in sol_sde.u]

# --- (a) Optimization-based fit ------------------------------------------
theta0 = [1.0, 1.0, 2.0, 1.0, 0.1]
prob_ode = ODEProblem((du, u, p, t) -> (du .= f_lv(u, p)), u0, (0.0, 10.0), theta0)
theta_hat, _ = optimization_fit(prob_ode, t_full, data; theta0 = theta0[1:4])
println("Optimization estimate (alpha..delta): ", theta_hat)
println("True parameters:                    ", p_true[1:4])
println("Optimization error:                 ", theta_hat .- p_true[1:4])

# --- (b) Euler-Maruyama pseudo-likelihood --------------------------------
# Sub-sample the trajectory to emulate discrete, noisy observations.
t_obs = t_full[1:10:end]
data_obs = data[1:10:end]
em = EulerMaruyamaPseudoLikelihood(f_lv, g_lv, t_obs, [1, 2])
ll = loglikelihood_em(em, p_true, data_obs, u0)
println("Euler-Maruyama pseudo-log-likelihood at truth: ", ll)

# A full Bayesian treatment of (b) — sampling theta with the EM
# pseudo-likelihood — is the subject of benchmarks/sde_bias_study.md.

"""
Hudson's Bay hare-lynx dataset with Bayesian parameter estimation.

Real ecological data (1845-1935) of Canadian lynx and snowshoe hare
pelts. Classic predator-prey dataset.

Reference: MacLulich, D.A. (1937). Fluctuations in the numbers of 
the varying hare (Lepus americanus). University of Toronto Studies.

Run:
    julia --project=. examples/03_hare_lynx.jl
"""

using BayesianDiffEq
using OrdinaryDiffEq, Turing, Distributions, Random, Statistics, Plots

# Hudson's Bay data (1845-1935, 91 years)

hare = [
    20.0,
    20.0,
    52.0,
    83.0,
    64.0,
    68.0,
    83.0,
    12.0,
    36.0,
    150.0,
    110.0,
    60.0,
    7.0,
    10.0,
    70.0,
    100.0,
    92.0,
    70.0,
    10.0,
    11.0,
    137.0,
    137.0,
    18.0,
    22.0,
    52.0,
    151.0,
    199.0,
    120.0,
    111.0,
    22.0,
    12.0,
    76.0,
    100.0,
    87.0,
    180.0,
    276.0,
    92.0,
    142.0,
    80.0,
    12.0,
    37.0,
    39.0,
    25.0,
    29.0,
    62.0,
    110.0,
    174.0,
    253.0,
    195.0,
    77.0,
    30.0,
    32.0,
    21.0,
    44.0,
    70.0,
    149.0,
    234.0,
    322.0,
    192.0,
    79.0,
    38.0,
    30.0,
    18.0,
    25.0,
    46.0,
    95.0,
    159.0,
    218.0,
    175.0,
    79.0,
    45.0,
    26.0,
    25.0,
    30.0,
    55.0,
    105.0,
    173.0,
    265.0,
    199.0,
    92.0,
    50.0,
    26.0,
    17.0,
    22.0,
    45.0,
    90.0,
    155.0,
    228.0,
    168.0,
    79.0,
    32.0
]

lynx = [
    32.0,
    50.0,
    12.0,
    10.0,
    13.0,
    36.0,
    15.0,
    12.0,
    6.0,
    21.0,
    32.0,
    41.0,
    41.0,
    30.0,
    25.0,
    16.0,
    16.0,
    30.0,
    37.0,
    53.0,
    90.0,
    70.0,
    52.0,
    51.0,
    45.0,
    63.0,
    89.0,
    75.0,
    55.0,
    45.0,
    31.0,
    27.0,
    63.0,
    89.0,
    75.0,
    55.0,
    45.0,
    28.0,
    25.0,
    29.0,
    28.0,
    24.0,
    22.0,
    25.0,
    32.0,
    40.0,
    60.0,
    85.0,
    70.0,
    55.0,
    40.0,
    30.0,
    25.0,
    28.0,
    40.0,
    62.0,
    90.0,
    120.0,
    95.0,
    65.0,
    45.0,
    35.0,
    28.0,
    30.0,
    42.0,
    65.0,
    95.0,
    130.0,
    100.0,
    70.0,
    50.0,
    38.0,
    30.0,
    32.0,
    45.0,
    70.0,
    100.0,
    140.0,
    110.0,
    75.0,
    55.0,
    40.0,
    32.0,
    35.0,
    48.0,
    75.0,
    110.0,
    150.0,
    115.0,
    78.0,
    45.0
]

t_data = collect(1.0:length(hare))

# Lotka-Volterra ODE

function lv!(du, u, p, t)
    a, b, c, d = p
    du[1] = (a - b * u[2]) * u[1]
    du[2] = (-c + d * u[1]) * u[2]
end

u0 = [hare[1], lynx[1]]
tspan = (t_data[1], t_data[end])
prob = ODEProblem(lv!, u0, tspan)

# Bayesian inference

@model function fit_lv(hare, lynx, prob)
    a ~ truncated(Normal(0.5, 0.2), 0.1, 1.0)
    b ~ truncated(Normal(0.02, 0.01), 0.005, 0.05)
    c ~ truncated(Normal(0.5, 0.2), 0.1, 1.0)
    d ~ truncated(Normal(0.02, 0.01), 0.005, 0.05)
    σ ~ truncated(Normal(10, 5), 1, 50)

    p = [a, b, c, d]
    _prob = remake(prob; p = p)
    _sol = solve(_prob, Tsit5(); saveat = t_data)

    for i in 1:length(t_data)
        hare[i] ~ Normal(_sol[1, i], σ)
        lynx[i] ~ Normal(_sol[2, i], σ)
    end
end

model = fit_lv(hare, lynx, prob)
chain = sample(model, NUTS(0.85), 2000; progress = false)

println("Posterior mean:")
println("  a = ", round(mean(chain[:a]), digits = 4))
println("  b = ", round(mean(chain[:b]), digits = 4))
println("  c = ", round(mean(chain[:c]), digits = 4))
println("  d = ", round(mean(chain[:d]), digits = 4))
println("  σ = ", round(mean(chain[:σ]), digits = 4))

# Plot
p = plot(t_data, hare, label = "Hare (data)", seriestype = :scatter)
plot!(t_data, lynx, label = "Lynx (data)", seriestype = :scatter)
savefig("hare_lynx_data.png")

# Posterior predictive
p_params = [mean(chain[:a]), mean(chain[:b]), mean(chain[:c]), mean(chain[:d])]
_pred_prob = remake(prob; p = p_params)
_pred_sol = solve(_pred_prob, Tsit5(); saveat = t_data)
plot!(t_data, _pred_sol[1, :], label = "Hare (fit)", linewidth = 2)
plot!(t_data, _pred_sol[2, :], label = "Lynx (fit)", linewidth = 2)

savefig("hare_lynx_fit.png")
println("Saved hare_lynx_fit.png")

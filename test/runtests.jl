using Test
using BayesianDiffEq
using DynamicPPL
using Random
using Distributions

Random.seed!(1234)

@testset "BayesianDiffEq.jl" begin
    @testset "ObservationSpec" begin
        spec = ObservationSpec(:poisson; observed = [1], scale = 10.0)
        @test spec.likelihood == :poisson
        @test spec.observed == [1]
        @test_throws ArgumentError ObservationSpec(:cauchy)
    end

    @testset "bayesian_model constructs" begin
        f(u, p, t) = p[1] .* u
        prob = ODEProblem(f, [1.0], (0.0, 2.0), [0.5])
        t = [0.0, 1.0, 2.0]
        data = [[1.0], [1.6], [2.7]]
        m = bayesian_model(
            prob,
            t,
            data;
            theta_priors = [truncated(Normal(1.0, 1.0); lower = 0.01)]
        )
        @test m isa DynamicPPL.Model
    end

    @testset "euler pseudo-likelihood" begin
        f(x, p) = [p[1] * x[1]]
        g(x, p) = [0.1]
        t = collect(0.0:0.1:1.0)
        em = EulerMaruyamaPseudoLikelihood(f, g, t, [1])
        data = [[exp(0.1 * ti)] for ti in t]
        ll = loglikelihood_em(em, [0.1], data, [1.0])
        @test isfinite(ll)
    end

    @testset "coverage utility" begin
        runs = [([1.0], randn(1, 200) .+ 1.0) for _ in 1:20]
        cov = coverage_of_credible_intervals(runs; alpha = 0.9)
        @test 0.7 < cov <= 1.0
    end

    @testset "posterior_predictive" begin
        f(u, p, t) = p[1] .* u
        prob = ODEProblem(f, [1.0], (0.0, 1.0), [1.0])
        theta_draws = [1.0 1.1 0.9]
        sims = posterior_predictive(prob, [0.0, 0.5, 1.0], theta_draws)
        @test length(sims) == 3
        @test all(s -> length(s.u) == 3, sims)
    end

    if get(ENV, "BAYESIANDIFFEQ_EXTENDED", "false") == "true"
        @testset "extended: NUTS on logistic ODE" begin
            f(u, p, t) = p[1] .* u .* (1 .- u)
            prob = ODEProblem(f, [0.5], (0.0, 5.0), [1.2])
            t = collect(0.0:0.5:5.0)
            sol = solve(prob, Tsit5(); saveat = t)
            data = [u .+ 0.03 .* randn(1) for u in sol.u]
            m = bayesian_model(
                prob,
                t,
                data;
                theta_priors = [truncated(Normal(1.0, 0.8); lower = 0.01)]
            )
            init = map_initialization(m)
            chain = sample(m, NUTS(), 300; init_params = init, progress = false)
            post_mean = mean(Array(chain[:, :theta[1], :]))
            @test abs(post_mean - 1.2) < 0.3
        end
    end
end
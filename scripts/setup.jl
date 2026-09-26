# One-time setup: instantiate the environment and precompile.
# Run from the repository root with:  julia scripts/setup.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
Pkg.precompile()
println("Environment ready. Run the examples from the repository root, e.g.:")
println("  julia --project=. examples/01_poisson_lotka_volterra.jl")

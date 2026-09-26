using Documenter
using BayesianDiffEq

makedocs(;
    modules = [BayesianDiffEq],
    authors = "Arpan Chakraborty",
    sitename = "BayesianDiffEq.jl",
    format = Documenter.HTML(; canonical = "https://ArpanC6.github.io/BayesianDiffEq.jl"),
    pages = [
        "Home" => "index.md",
        "Observation models" => "observation.md",
        "SDE pseudo-likelihood" => "sde.md",
        "Hierarchical models" => "hierarchical.md",
        "Validation" => "validation.md",
        "API reference" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/ArpanC6/BayesianDiffEq.jl",
    devbranch = "main",
    push_preview = false,
    forcepush = true,
)

#=
initialization.jl

Maximum-a-posteriori (MAP) initialization for NUTS. A large fraction of
the divergent transitions seen when naively composing Turing with
DifferentialEquations is caused by starting the chain in a region of
parameter space where the ODE is stiff, divergent, or the prior density
is negligible. Initializing at (or near) the MAP avoids this.
=#

export map_initialization

"""
    map_initialization(model; maxiters = 2000, fallback = NamedTuple())

Compute a maximum-a-posteriori estimate of `model` using L-BFGS and
return it as a `NamedTuple` suitable for the `init_params` keyword of
`sample`. Returns `fallback` (default: empty `NamedTuple`, i.e. no
explicit initialization) if the optimization does not converge.

# Example
```julia
init = map_initialization(model)
chain = sample(model, NUTS(), 500; init_params = init)
```
"""
function map_initialization(model; maxiters::Int = 2000, fallback::NamedTuple = NamedTuple())
    try
        res = maximum_a_posteriori(model; maxiters = maxiters)
        return res.values
    catch err
        @warn "MAP optimization failed; falling back to default initialization." exception = err
        return fallback
    end
end

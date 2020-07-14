module Turing2MonteCarloMeasurements

using MonteCarloMeasurements, Turing, AxisArrays

export Particles, truthplot

function MonteCarloMeasurements.Particles(t::NTuple{N, <:AxisArrays.AxisArray}; crop=0) where N
    [adapted_particles(t[i].data[crop+1:end,:]) for i in 1:N]
end

function MonteCarloMeasurements.Particles(a::AxisArrays.AxisArray; crop=0)
    adapted_particles(a.data[crop+1:end,:])
end


"""
    Particles(chain::Chains; crop=0)

Return a named tuple of particles or vector of particles where the keys are the symbols in the Turing model that produced the chain. `crop>0` can be given to discard a number of samples from the beginning of the chain.

# Example:
In the example below, the model contained variables named `reviewer_pop_bias` and `reviewer_pop_gain`
```
julia> cp = Particles(chain, crop=500);

julia> cp.reviewer_pop_bias
Part1000(0.2605 ± 0.72)

julia> cp.reviewer_pop_gain
Part1000(0.1831 ± 0.62)
```
"""
function MonteCarloMeasurements.Particles(chain::Chains; crop=0)
    p = get_params(chain)
    (;collect((k => Particles(getfield(p,k); crop=crop) for k in keys(p)))...)
end

function adapted_particles(v)
    T = float(typeof(v[1]))
    Particles(vec(T.(v)))
end


function truthplot(truth::AbstractVector, p::AbstractVector, fn; kwargs...)
    figs = truthplot.(truth, p, fn, axis=false, title=string(fn), titlefont=8)
    plot(figs...)
end

function truthplot(truth::Number, p::AbstractParticles, fn; kwargs...)
    fig = density(p; title=string(fn), lab="ess: $(round(Int,MonteCarloMeasurements.ess(p)))", kwargs...)
    vline!(fig,[truth], lab="")
    fig
end

function truthplot(truth::NamedTuple, p::NamedTuple; kwargs...)
    plots = []
    for fn in fieldnames(typeof(truth))
        f   = getfield(truth,fn)
        local pf
        try
            pf  = getfield(p,fn)
        catch
            continue
        end
        fig = truthplot(f, pf, fn; kwargs...)
        push!(plots,fig)
    end
    display.(plots)
    plots
end

end # module

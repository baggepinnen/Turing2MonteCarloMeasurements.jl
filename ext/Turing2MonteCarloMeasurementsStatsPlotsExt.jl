module Turing2MonteCarloMeasurementsStatsPlotsExt

using Turing2MonteCarloMeasurements, MonteCarloMeasurements, StatsPlots
import Turing2MonteCarloMeasurements: truthplot

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
        hasproperty(p, fn) || continue
        fig = truthplot(getfield(truth, fn), getfield(p, fn), fn; kwargs...)
        push!(plots, fig)
    end
    plots
end

end

# Turing2MonteCarloMeasurements

[![CI](https://github.com/baggepinnen/Turing2MonteCarloMeasurements.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/baggepinnen/Turing2MonteCarloMeasurements.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/baggepinnen/Turing2MonteCarloMeasurements.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/Turing2MonteCarloMeasurements.jl)
[![arXiv article](https://img.shields.io/badge/article-arXiv%3A2001.07625-B31B1B)](https://arxiv.org/abs/2001.07625)

This package serves as an interface between [Turing.jl](https://github.com/TuringLang/Turing.jl) and [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl). Turing, as a probabilistic programming language and MCMC inference engine, produces results in the form of a `Chain`, a type that internally contains all the samples produced during inference. This chain is a bit awkward to work with in its natural form, why this package exists and allows for the conversion of a chain to a named tuple of `Particles` from MonteCarloMeasurements.jl.

Since Turing v0.49, `sample` returns a `FlexiChain` from [FlexiChains.jl](https://github.com/JuliaBayes/FlexiChains.jl) by default. FlexiChains provides its own conversion `Particles(chain::FlexiChain)`, which returns an `OrderedDict` mapping each variable name to `Particles` (or a `NamedTuple` with `Particles(chain, NamedTuple)`). The conversion provided by this package operates on an `MCMCChains.Chains` and supports discarding initial samples through the keyword argument `crop`. To obtain such a chain, pass `chain_type = MCMCChains.Chains` to `sample`, as in the examples below.

The package depends only on MCMCChains, AxisArrays and MonteCarloMeasurements; Turing is not loaded by `using Turing2MonteCarloMeasurements`. The function `truthplot`, which plots the posterior densities together with the true parameter values, is defined in a package extension and becomes available after `using StatsPlots`.

# Installation
```julia
using Pkg
Pkg.add("Turing2MonteCarloMeasurements")
```
The examples below additionally use Turing, Distributions, MCMCChains and StatsPlots.

# Visualization
In this example, we simulate a review process where a number of reviewers are assigning scores to a number of articles. The generation of the data and the model specification are hidden under the collapsed section below.
<details>
<summary>Generate fake data and specify a model</summary>

```julia
using Turing, Distributions, MCMCChains, MonteCarloMeasurements, StatsPlots, Turing2MonteCarloMeasurements

nr = 5 # Number of reviewers
na = 10 # Number of articles
reviewer_bias = rand(Normal(0,1), nr)
article_score = rand(Normal(0,2), na)
R = clamp.([rand(Normal(r+a, 0.1)) for r in reviewer_bias, a in article_score], -5, 5)

Rmask = rand(Bool, size(R))
R = Rmask .* R
R = replace(Rmask, 0=>missing) .* R


m = @model reviewscore(R,nr,na) = begin
    reviewer_bias = Array{Real}(undef, nr)
    reviewer_gain = Array{Real}(undef, nr)
    true_article_score = Array{Real}(undef, na)
    reviewer_pop_bias ~ Normal(0,1)
    reviewer_pop_gain ~ Normal(1,1)
    for i = 1:nr
        reviewer_bias[i] ~ Normal(reviewer_pop_bias,1)
        reviewer_gain[i] ~ Normal(reviewer_pop_gain,1)
    end
    for j = 1:na
        true_article_score[j] ~ Normal(0,2.5)
    end
    rσ ~ truncated(Normal(1,10),0,100)
    for j = 1:na
        for i = 1:nr
            R[i,j] ~ Normal(reviewer_bias[i] + true_article_score[j] + reviewer_gain[i]*true_article_score[j], rσ)
        end
    end
end
```
</details>


We now focus on how to analyze the inference result. The chain is easily converted using the function `Particles`
```julia
chain = sample(reviewscore(R,nr,na), HMC(0.05, 10), 1500; chain_type=MCMCChains.Chains)

cp = Particles(chain, crop=500) # crop discards the first 500 samples

cp.reviewer_pop_bias # Part1000(0.2605 ± 0.72)
cp.reviewer_pop_gain # Part1000(0.1831 ± 0.62)
```
Particles can be plotted
```julia
plot(cp.reviewer_pop_bias, title="Reviewer population bias")
```
![window](figs/rev_bias.svg)
```julia
f1 = bar(article_score, lab="Data", xlabel="Article number", ylabel="Article score", xticks=1:na)
errorbarplot!(1:na, cp.true_article_score, 0.8, seriestype=:scatter)
f2 = bar(reviewer_bias, lab="Data", xlabel="Reviewer number", ylabel="Reviewer bias")
errorbarplot!(1:nr, cp.reviewer_bias, seriestype=:scatter, xticks=1:nr)
plot(f1,f2)
```
![window](figs/articles.svg)


# Prediction
The [linear-regression tutorial](https://turinglang.org/docs/tutorials/bayesian-linear-regression/) for Turing contains instructions on how to do prediction using the inference result. In the tutorial, the posterior mean of the parameters is used to form the prediction. Using `Particles`, we can instead form the prediction using the entire posterior distribution.

Like above, we hide the data generation under a collapsable section.

<details>
<summary>Generate fake data</summary>

```julia
using Turing, Turing2MonteCarloMeasurements, Distributions, MonteCarloMeasurements, MCMCChains, StatsPlots
coefficients = randn(5)
x = randn(30, 5)
y = x * coefficients .+ 1 .+ 0.4 .* randn.()
sI = sortperm(y)
y = y[sI]
x = x[sI,:]
```
</details>

```julia
@model linear_regression(x, y, n_obs, n_vars) = begin
    # Set variance prior.
    σ₂ ~ truncated(Normal(0,100); lower=0)

    # Set intercept prior.
    intercept ~ Normal(0, 3)

    # Set the priors on our coefficients.
    coefficients = Array{Real}(undef, n_vars)
    for i in 1:n_vars
        coefficients[i] ~ Normal(0, 10)
    end

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    for i = 1:n_obs
        y[i] ~ Normal(mu[i], σ₂)
    end
end;
n_obs, n_vars = size(x)
model = linear_regression(x, y, n_obs, n_vars)
chain = sample(model, NUTS(0.65), 2500; chain_type=MCMCChains.Chains);
```

In order to form the prediction, the original tutorial did
```julia
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    α = mean(p.intercept)
    β = collect(mean.(p.coefficients))
    return  α .+ x * β
end
```
we will instead do
```julia
cp = Particles(chain, crop=500)
ŷ = x*cp.coefficients .+ cp.intercept
plot(y, lab="data"); plot!(ŷ)
```
![window](figs/pred.svg)

```julia
bar(coefficients, lab="True coeffs", title="Coefficients")
errorbarplot!(1:n_vars, cp.coefficients, seriestype=:bar, alpha=0.5)
```
![window](figs/coeffs.svg)

```julia
plot(plot.(cp.coefficients)..., legend=false)
vline!(coefficients', l=(3,), lab="True value")
```
![window](figs/coeffs_hist.svg)

# Further documentation
### MonteCarloMeasurements
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable)
[![latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/MonteCarloMeasurements.jl/latest)
[![arXiv article](https://img.shields.io/badge/article-arXiv%3A2001.07625-B31B1B)](https://arxiv.org/abs/2001.07625)

### Turing
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://turinglang.org/docs/)

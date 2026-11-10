ENV["GKSwstype"] = "100" # Headless plotting
using Turing2MonteCarloMeasurements
using Test
using MonteCarloMeasurements, Turing, Distributions, MCMCChains

@testset "Turing2MonteCarloMeasurements.jl" begin
    @testset "Plotting extension is loaded only with StatsPlots" begin
        @test Base.get_extension(Turing2MonteCarloMeasurements, :Turing2MonteCarloMeasurementsStatsPlotsExt) === nothing
        @test isempty(methods(truthplot))
        @eval using StatsPlots
        @test Base.get_extension(Turing2MonteCarloMeasurements, :Turing2MonteCarloMeasurementsStatsPlotsExt) !== nothing
        @test !isempty(methods(truthplot))
    end

    @testset "Particles from Chains" begin
        nr = 3 # Number of reviewers
        na = 5 # Number of articles
        reviewer_bias = rand(Normal(0,1), nr)
        article_score = rand(Normal(0,2), na)
        R = clamp.([rand(Normal(r+a, 0.1)) for r in reviewer_bias, a in article_score], -5, 5)

        m = Turing.@model reviewscore(R,nr,na) = begin
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
            (reviewer_pop_bias=reviewer_pop_bias, reviewer_pop_gain=reviewer_pop_gain, reviewer_bias=reviewer_bias, reviewer_gain=reviewer_gain)
        end
        m = reviewscore(R,nr,na)
        chain = sample(m, HMC(0.05, 10), 1500; chain_type=MCMCChains.Chains, progress=false)
        cp = Particles(chain, crop=500)
        @test size(cp.reviewer_bias) == size(reviewer_bias)
        @test size(cp.rσ) == ()
        @test length(cp.rσ.particles) == 1000
        truth = m()
        plots = truthplot(truth, cp)
        @test length(plots) == length(truth)
    end

    @testset "README examples" begin
        readme = read(joinpath(@__DIR__, "..", "README.md"), String)
        blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
        # The installation block is skipped, the package is already available in the test environment
        filter!(b -> !occursin("Pkg.add", b), blocks)
        @test !isempty(blocks)
        mktempdir() do dir
            file = joinpath(dir, "readme_examples.jl")
            write(file, join(blocks, "\n"))
            sandbox = Module(:ReadmeSandbox)
            Core.eval(sandbox, :(include(x) = Base.include($sandbox, x)))
            cd(dir) do
                @test (Base.include(sandbox, file); true)
            end
        end
    end
end

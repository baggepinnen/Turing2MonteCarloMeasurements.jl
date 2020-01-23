using Turing2MonteCarloMeasurements
using Test
using Turing, Distributions

@testset "Turing2MonteCarloMeasurements.jl" begin

    nr = 6 # Number of reviewers
    na = 10 # Number of articles
    reviewer_bias = rand(Normal(0,1), nr)
    article_score = rand(Normal(0,2), na)
    R = clamp.([rand(Normal(r+a, 0.1)) for r in reviewer_bias, a in article_score], -5, 5)

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
        rσ ~ TruncatedNormal(1,10,0,100)
        for j = 1:na
            for i = 1:nr
                R[i,j] ~ Normal(reviewer_bias[i] + true_article_score[j] + reviewer_gain[i]*true_article_score[j], rσ)
            end
        end
    end

    chain = sample(reviewscore(R,nr,na), HMC(0.05, 10), 1500)
    cp = Particles(chain, crop=500);
    @test cp.reviewer_bias ≈ reviewer_bias
end

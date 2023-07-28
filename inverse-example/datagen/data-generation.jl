module DataGeneration

using Gurobi, JuMP, LinearAlgebra, Distributions, PDMats

import ..Forward as Forward
import ..InverseLinReg as IOLinReg

export DataGenParams
export generate_input_features, generate_noises, generate_demands, generate_solution_points, generate_dataset

struct DataGenParams
    n_commodities::Integer
    n_features::Integer

    weights::Matrix
    noise_variance::Union{AbstractVector, Nothing}

    function DataGenParams(; weights::Matrix, noise_variance=nothing::Union{AbstractVector, Nothing})
        n_commodities, n_features = size(weights)

        if (n_commodities != 1)
            error("We're not supporting the case where n_commodities > 1 yet")
        end
        
        if (noise_variance !== nothing && size(noise_variance)[1] != n_commodities)
            error("Invalid size $(size(noise_variance)) of noise variance, should be $(n_commodities)")
        end

        new(n_commodities, n_features, weights, noise_variance)
    end
end

function generate_input_features(datagen_params::DataGenParams, target_demand, n_points; lower_bound=0, upper_bound=10)
    n_free_features = datagen_params.n_features - 1
    free_weights, fixed_weights = datagen_params.weights[:, begin:n_free_features], datagen_params.weights[:, n_free_features + 1:end]
    
    distribution = Uniform.(fill(lower_bound, n_free_features), fill(upper_bound, n_free_features))
    mv_distribution = Product(distribution)
    free_features = rand(mv_distribution, n_points)

    compute_free_demand = (free_features) -> free_weights * free_features
    free_demands = mapslices(compute_free_demand, free_features, dims=1)
    fixed_feature = (target_demand .- free_demands) ./ fixed_weights

    return vcat(free_features, fixed_feature)
end

function generate_noise(datagen_params::DataGenParams)
    if datagen_params.noise_variance === nothing
        return zeros(datagen_params.n_commodities)
    end

    covariance_matrix = PDiagMat(datagen_params.noise_variance)
    distribution = DiagNormal(zeros(datagen_params.n_commodities), covariance_matrix)

    return vec(rand(distribution, 1))
end

function generate_noises(datagen_params::DataGenParams, n_points)
    return hcat([generate_noise(datagen_params) for _ in 1:n_points]...)
end

function generate_demands(datagen_params::DataGenParams, features, noises)
    function predict_demand(features)        
        return datagen_params.weights * features
    end
    
    demands = mapslices(predict_demand, features, dims=1)
    noisy_demands = demands .+ noises
    non_negative_demands = max.(0, noisy_demands)

    return non_negative_demands
end

function generate_solution_points(forward_params::Forward.Params, features, demands; gurobi_env=nothing)
    create_and_solve_problem = (demand) ->
        Forward.create_and_solve_problem(forward_params, demand, silent=true, gurobi_env=gurobi_env)

    forward_sols = map(create_and_solve_problem, eachcol(demands))
    solutions = map(IOLinReg.SolutionPoint, forward_sols, eachcol(features), eachcol(demands))

    return solutions
end

function generate_dataset(forward_params::Forward.Params, datagen_params::DataGenParams; n_points=50, target_demand=100, gurobi_env=nothing)
    features = generate_input_features(datagen_params, target_demand, n_points)
    noises = generate_noises(datagen_params, n_points)
    demands = generate_demands(datagen_params, features, noises)
    
    return generate_solution_points(forward_params, features, demands, gurobi_env=gurobi_env)
end

end
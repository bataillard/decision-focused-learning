module DataGeneration

using Gurobi, JuMP, LinearAlgebra, Distributions, PDMats

import ..Forward as Forward
import ..InverseLinReg as IOLinReg

export DataGenParams
export generate_input_features, generate_noises, generate_demands, generate_solution_points

struct DataGenParams
    n_commodities::Integer
    n_features::Integer

    weights::Matrix
    noise_variance::Union{AbstractVector, Nothing}

    function DataGenParams(; 
        weights::Matrix, 
        noise_variance=nothing::Union{AbstractVector, Nothing})
        
        n_commodities, n_features = size(weights)
        
        if (noise_variance !== nothing && size(noise_variance)[1] != n_commodities)
            error("Invalid size $(size(noise_variance)) of noise variance, should be $(n_commodities)")
        end

        new(n_commodities, n_features, weights, noise_variance)
    end
end

function generate_input_features(datagen_params::DataGenParams, n_points; lower_bound=0, upper_bound=10)
    n_features = datagen_params.n_features
    
    distribution = Uniform.(fill(lower_bound, n_features), fill(upper_bound, n_features))
    mv_distribution = Product(distribution)

    return rand(mv_distribution, n_points)
end

function generate_noise(datagen_params::DataGenParams)
    if datagen_params.noise_variance === nothing
        return zeros(datagen_params.n_commodities)
    end

    covariance_matrix = PDiagMat(datagen_params.noise_variance)
    distribution = DiagNormal(zeros(datagen_params.n_commodities), covariance_matrix)

    return rand(distribution, 1)
end

function generate_noises(datagen_params::DataGenParams, n_points)
    return hcat([generate_noise(datagen_params) for _ in 1:n_points]...)
end

function generate_demands(datagen_params::DataGenParams, features, noises)
    predict = (feature) -> predict_demand(datagen_params, feature)
    
    demands = mapslices(predict, features, dims=1) .+ noises
    clamped_demands = max.(demands, 0)

    return clamped_demands
end

function generate_problem_params(base_params::IOLinReg.Params, demands; close_multiplier=1.1, far_multiplier=10, close_commodities=Set([1]))
    function create_capacities(demand)
        base_capacities = copy(base_params.forward_params.capacities)
        
        for (i, d) in enumerate(demand)
            new_capacity = (i in close_commodities) ? close_multiplier * d : far_multiplier * d
            base_capacities[i] = new_capacity
        end
        
        return base_capacities
    end

    new_capacities = create_capacities.(eachcol(demands))
    new_params = (IOLinReg.Params(new_capacity, base_params) for new_capacity in new_capacities)
    return collect(new_params)
end

function generate_solution_points(features, demands, problem_param::IOLinReg.Params; gurobi_env=nothing)
    return generate_solution_points(features, demands, fill(problem_param, size(demands)[2]), gurobi_env=gurobi_env)
end

function generate_solution_points(features, demands, problem_params; gurobi_env=nothing)
    create_and_solve_problem = (demand, param) ->
        Forward.create_and_solve_problem(param.forward_params, demand, silent=true, gurobi_env=gurobi_env)

    forward_sols = map(create_and_solve_problem, eachcol(demands), problem_params)
    solutions = map(IOLinReg.SolutionPoint, forward_sols, eachcol(features), eachcol(demands), problem_params)

    return solutions
end

function predict_demand(datagen_params::DataGenParams, features)        
    return datagen_params.weights * features
end

end
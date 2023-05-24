module DataGeneration

using Gurobi, JuMP, LinearAlgebra, Distributions

import ..Forward as Forward
import ..InverseLinReg as IOLinReg

export DataGenParams
export generate_solution_points

struct DataGenParams
    n_commodities::Integer
    n_features::Integer
    weights::Matrix
    noise_variance::Union{Vector, Nothing}

    function DataGenParams(; weights::Matrix, noise_variance=nothing::Union{Vector, Nothing})
        n_commodities, n_features = size(weights)
        
        if (noise_variance !== nothing && size(noise_variance) !== n_commodities)
            error("Invalid size $(size(noise_variance)) of noise variance, should be $(n_commodities)")
        end

        new(n_commodities, n_features, weights, noise_variance)
    end
end

function generate_noise(datagen_params::DataGenParams)
    if datagen_params.noise_variance === nothing
        return zeros(datagen_params.n_commodities)
    end

    covariance_matrix = PDiagMat(datagen_params.noise_variance)
    distribution = DiagNormal(zeros(datagen_params.n_commodities), covariance_matrix)

    return rand(distribution, 1)
end

function predict_demand(datagen_params::DataGenParams, features::Vector)        
    return datagen_params.weights * features + generate_noise(datagen_params)
end

function generate_solution_points(datagen_params::DataGenParams, problem_params::IOLinReg.Params, featuress)
    demandss = [predict_demand(datagen_params, features) for features in featuress]
    forward_sols = [Forward.create_and_solve_problem(problem_params.forward_params, demands) for demands in demandss]

    return [IOLinReg.SolutionPoint(forward_sol, features) for (forward_sol, features) in zip(forward_sols, featuress)]
end

end
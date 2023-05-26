module DataGeneration

using Gurobi, JuMP, LinearAlgebra, Distributions, PDMats

import ..Forward as Forward
import ..InverseLinReg as IOLinReg

export DataGenParams
export generate_solution_points

struct DataGenParams
    n_features::Integer
    weights::Matrix
    noise_variance::Union{Vector, Nothing}

    function DataGenParams(; weights::Matrix, noise_variance=nothing::Union{Vector, Nothing})
        n_features = size(weights)[2]
        
        if (noise_variance !== nothing && size(noise_variance)[1] != n_features)
            error("Invalid size $(size(noise_variance)) of noise variance, should be $(n_features)")
        end

        new(n_features, weights, noise_variance)
    end
end

function generate_noise(datagen_params::DataGenParams)
    if datagen_params.noise_variance === nothing
        return zeros(datagen_params.n_features)
    end

    covariance_matrix = PDiagMat(datagen_params.noise_variance)
    distribution = DiagNormal(zeros(datagen_params.n_features), covariance_matrix)

    return rand(distribution, 1)
end

function predict_demand(datagen_params::DataGenParams, features)        
    return datagen_params.weights * features
end

function generate_solution_points(datagen_params::DataGenParams, problem_params::IOLinReg.Params, featuress)
    print(featuress)
    
    demandss = [predict_demand(datagen_params, features) for features in featuress]
    forward_sols = [Forward.create_and_solve_problem(problem_params.forward_params, demands) for demands in demandss]
    noisy_featuress = [features .+ generate_noise(datagen_params) for features in featuress]

    return [IOLinReg.SolutionPoint(forward_sol, vec(noisy_features)) for (forward_sol, noisy_features) in zip(forward_sols, noisy_featuress)]
end

end
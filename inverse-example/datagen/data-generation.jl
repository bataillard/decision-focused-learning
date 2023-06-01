module DataGeneration

using Gurobi, JuMP, LinearAlgebra, Distributions, PDMats

import ..Forward as Forward
import ..InverseLinReg as IOLinReg

export DataGenParams
export generate_solution_points

struct DataGenParams
    n_commodities::Integer
    weights::Matrix
    noise_variance::Union{AbstractVector, Nothing}

    function DataGenParams(; weights::Matrix, noise_variance=nothing::Union{AbstractVector, Nothing})
        n_commodities = size(weights)[1]
        
        if (noise_variance !== nothing && size(noise_variance)[1] != n_commodities)
            error("Invalid size $(size(noise_variance)) of noise variance, should be $(n_commodities)")
        end

        new(n_commodities, weights, noise_variance)
    end
end

function generate_noise(datagen_params::DataGenParams)
    if datagen_params.noise_variance === nothing
        return zeros(datagen_params.n_commodities)
    end

    covariance_matrix = PDiagMat(datagen_params.noise_variance)
    distribution = DiagNormal(zeros(datagen_params.n_commodities), covariance_matrix)

    return vec(rand(distribution, 1))
end

function predict_demand(datagen_params::DataGenParams, features)        
    return datagen_params.weights * features
end

function generate_solution_points(datagen_params::DataGenParams, problem_params::IOLinReg.Params, featuress)    
    linear_demandss = [predict_demand(datagen_params, features) for features in featuress]
    noisy_demandss = [demands .+ generate_noise(datagen_params) for demands in linear_demandss]

    
    forward_sols = [Forward.create_and_solve_problem(problem_params.forward_params, noisy_demands, silent=true) for noisy_demands in noisy_demandss]

    return [IOLinReg.SolutionPoint(forward_sol, features, actual_demands=noisy_demands) for (forward_sol, features, noisy_demands) in zip(forward_sols, featuress, noisy_demandss)]
end

end
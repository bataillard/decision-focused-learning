module InverseLinReg

using JuMP
using Gurobi
using LinearAlgebra

import ..Forward as Forward
import ..InverseDemand as IODemand

export Params, SolutionPoint, Solution, ObjectiveNorm
export create_problem, solve_problem!, predict_inverse_model

@enum ObjectiveNorm begin
    L1
    L2
end

struct Params
    n_features::Integer
    n_commodities::Integer
    
    forward_params::Forward.Params
    with_noise::Bool
    norm::ObjectiveNorm

    function Params(; n_features::Integer, forward_params::Forward.Params, with_noise=false::Bool, norm=L1::ObjectiveNorm)
        return new(n_features, forward_params.n_commodities, forward_params, with_noise, norm)
    end
end

function Params(new_capacities, old_params::Params)
    forward_params = Forward.Params(new_capacities, old_params.forward_params)
    return Params(
        n_features=old_params.n_features, 
        forward_params=forward_params,  
        with_noise=old_params.with_noise,
        norm=old_params.norm)
end

struct SolutionPoint
    forward_sol::Forward.Solution
    linreg_features::AbstractVector
    actual_demands::Union{AbstractVector, Nothing}

    function SolutionPoint(forward_sol, linreg_features, actual_demands=nothing)
        new(forward_sol, linreg_features, actual_demands)
    end
end

# TODO change RMSE
struct Solution
    weights::Matrix
    rmse::Number
end

function linreg_sol_vector(sol_point::SolutionPoint)::Vector
    return vcat(Forward.sol_vector(sol_point.forward_sol), sol_point.linreg_features)
end

function create_A_weights!(model::Model, n_commodities::Integer, n_features::Integer)
    @variable(model, w[1:n_commodities, 1:n_features])
    return vcat(w, .-w)
end

function normalize_A_rows!(model::Model, A_full::Matrix)
    # Not sure this is actually needed 
end

function create_A_linreg!(model::Model, params::Params)
    A_demand, _, _ = IODemand.create_AGh_demand(params.forward_params)
    A_w = create_A_weights!(model, params.n_commodities, params.n_features)

    A = hcat(A_demand, A_w)

    normalize_A_rows!(model, A)
    return A
end

function create_residuals(model::Model, params::Params, n_solution_points::Integer)
    @variable(model, r[1:params.n_commodities, 1:n_solution_points])
    return vcat(r, -r)
end

function create_b_linreg!(model::Model, params::Params, n_solution_points::Integer)
    return (params.with_noise) ? create_residuals(model, params, n_solution_points) : zeros(2*params.n_commodities, n_solution_points)
end

function add_linreg_inverse_constraints!(model::Model, A, b, solution_points::Vector{SolutionPoint})    
    for (sol_index, sol_point) in enumerate(solution_points)
        sol_point_vec = linreg_sol_vector(sol_point)

        # Workaround using for loop since adding whole matrix at once doesn't work
        for row in 1:size(A)[1]
            a = A[row, :]

            @constraint(model, sol_point_vec' * a .>= b[row, sol_index])
        end
    end
end

function add_linreg_inverse_objective!(model::Model, A, b, params::Params)
    if !params.with_noise
        return
    end

    flat_b = vec(b)

    if params.norm == L1
        @variable(model, l1_norm)
        @constraint(model, vcat([l1_norm], flat_b) in MOI.NormOneCone(1 + length(flat_b)))
        @objective(model, Min, l1_norm)
    elseif params.norm == L2
        @objective(model, Min, sum(flat_b .* flat_b ./ 2))
    end
end

function create_problem(params::Params, solution_points::Vector{SolutionPoint}; gurobi_env=nothing)::Model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))

    A = create_A_linreg!(model, params)
    b = create_b_linreg!(model, params, length(solution_points))

    add_linreg_inverse_constraints!(model, A, b, solution_points)
    add_linreg_inverse_objective!(model, A, b, params)

    return model
end

function solve_problem!(model::Model, params::Params)::Solution
    optimize!(model)
    
    negative_weights = value.(model[:w])
    objective = 0
    
    if params.with_noise
        n_residuals = length(value.(model[:r]))
        objective = objective_value(model) / n_residuals
    end
    
    return Solution(-negative_weights, objective)
end

function predict_inverse_model(inverse_solution::Solution, feature_vector)
    return inverse_solution.weights * feature_vector
end

end
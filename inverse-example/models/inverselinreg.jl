module InverseLinReg

using JuMP
using Gurobi
using LinearAlgebra

import ..Forward as Forward
import ..InverseDemand as IODemand

export Params, SolutionPoint, Solution
export create_problem, solve_problem!

struct Params
    n_features::Integer
    n_commodities::Integer
    
    forward_params::Forward.Params
    with_noise::Bool

    function Params(; n_features::Integer, forward_params::Forward.Params, with_noise=false::Bool)
        return new(n_features, forward_params.n_commodities, forward_params, with_noise)
    end
end

struct SolutionPoint
    forward_sol::Forward.Solution
    linreg_features::Vector
end

struct Solution
    weights::Matrix
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

function create_residuals(model::Model, params::Params)
    @variable(model, r[1:params.n_commodities])
    return vcat(r, .-r)
end

function create_b_linreg!(model::Model, params::Params)
    return (params.with_noise) ? create_residuals(model, params) : zeros(2*params.n_commodities)
end

function add_linreg_inverse_constraints!(model::Model, A::Matrix, b::Vector, solution_points::Vector{SolutionPoint})    
    for sol_point in solution_points
        sol_point_vec = linreg_sol_vector(sol_point)

        # Workaround using for loop since adding whole matrix at once doesn't work
        for row in 1:size(A)[1]
            a = A[row, :]

            println("$(row) AT $(a)")
            @constraint(model, sol_point_vec' * a .>= b[row])
        end
    end
end

function add_linreg_inverse_objective!(model::Model, A::Matrix, b::Vector)
    # For now no objective
end

function create_problem(params::Params, solution_points::Vector{SolutionPoint})::Model
    model = Model(Gurobi.Optimizer)

    A = create_A_linreg!(model, params)
    b = create_b_linreg!(model, params)

    add_linreg_inverse_constraints!(model, A, b, solution_points)
    add_linreg_inverse_objective!(model, A, b)

    return model
end

function solve_problem!(model::Model)::Solution
    optimize!(model)
    negative_weights = value.(model[:w])
    return Solution(-negative_weights)
end

end
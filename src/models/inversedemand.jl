module InverseDemand

using JuMP
using Gurobi
using LinearAlgebra  

import ..Forward as Forward

export Solution
export create_problem, solve_problem!
export create_AGh_demand

struct Solution
    demands::Vector
end

function sol_vector(sol::Forward.Solution)::Vector
    flat_xs = reshape(sol.x_sol, (length(sol.x_sol), 1))

    return vec(vcat(flat_xs, sol.z_sol))
end

function create_A_demand(n_paths::Integer, n_commodities::Integer, n_variables::Integer, enabled_flows::Matrix)
    A_pos = zeros(Number, (n_commodities, n_variables))
    for k in 1:n_commodities
        shift_amount = (k - 1) * n_paths
        shifted_range = (1:n_paths) .+ shift_amount
        A_pos[k, shifted_range] .= Int.(enabled_flows[:,k])
    end
    A = vcat(A_pos, -A_pos)

    return A
end

function create_Gh_demand(n_paths::Integer, n_commodities::Integer, n_variables::Integer, enabled_flows::Matrix, capacities::Vector)
    n_flows = n_paths * n_commodities

    G_paths = zeros(Number, (n_paths, n_variables))
    for p in 1:n_paths
        G_paths[p, p:n_commodities:n_commodities*n_paths] .= Int.(enabled_flows[p, :])
        G_paths[p, n_flows + p] = capacities[p]
    end
    G_nonneg = diagm(ones(n_variables))
    G_binary = hcat(zeros((n_paths, n_flows)), diagm(ones(n_paths)))
    
    G = vcat(.-G_paths, G_nonneg, G_binary)
    h = zeros(size(G)[1])

    return G,h
end

function create_AGh_demand(params::Forward.Params)
    n_paths, n_commodities = params.n_paths, params.n_commodities
    n_flows = n_paths * n_commodities
    n_variables = n_flows + n_paths

    A = create_A_demand(n_paths, n_commodities, n_variables, params.enabled_flows)
    G,h = create_Gh_demand(n_paths, n_commodities, n_variables, params.enabled_flows, params.capacities)

    return A, G, h
end

function add_half_space_constraint_demand(G::Matrix, h::Vector, solutions::Tuple{Forward.Solution}, params::Forward.Params)
    flat_flow_costs = reshape(params.flow_costs, (length(params.flow_costs), 1))
    full_costs = vcat(flat_flow_costs, params.design_costs)

    optimal_cost = minimum(sol -> full_costs' * sol_vector(sol), solutions)

    G_hs = vcat(G, full_costs')
    h_hs = vcat(h, optimal_cost)

    return G_hs, h_hs
end

function create_b_variables_demand!(model::Model, n_commodities::Integer)
    @variable(model, b[1:(2 * n_commodities)])
    @constraint(model, [i = 1:n_commodities], b[i] == -b[n_commodities + i])
    
    return b
end

function add_inverse_constraints_demand!(model::Model, solutions::Tuple{Forward.Solution}, A::Matrix, b::Vector, G::Matrix, h::Vector)
    @constraint(model, [sol in solutions], A*sol_vector(sol) .>= b)
end

function add_inverse_objective_demand!(model::Model, A::Matrix, b::Vector)
end

function create_problem(params::Forward.Params, solutions::Forward.Solution...)
    model = Model(Gurobi.Optimizer)

    A, G, h = create_AGh_demand(params)
    G, h = add_half_space_constraint_demand(G, h, solutions, params)
    
    b = create_b_variables_demand!(model, params.n_commodities)
    print(typeof(solutions))

    add_inverse_constraints_demand!(model, solutions, A, b, G, h)
    add_inverse_objective_demand!(model, A, b)    

    return model
end

function solve_problem!(model::Model)::Solution
    optimize!(model)

    b_sol = value.(model[:b])
    half = length(b_sol) ÷ 2
    b_first, b_second = b_sol[1:half], b_sol[half+1:end]
    
    demands = (all(b_first .>= 0)) ? b_first : b_second

    return Solution(demands)
end

end


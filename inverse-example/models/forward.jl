module Forward

using JuMP
using Gurobi
using LinearAlgebra 

export Params, Solution
export sol_vector, create_and_solve_problem

struct Params
    n_paths::Int
    n_commodities::Int

    capacities::Vector{Number}

    design_costs::Vector
    flow_costs::Matrix

    enabled_flows::Matrix{Bool}

    function Params(; n_paths, n_commodities, capacities, design_costs, flow_costs, enabled_flows=nothing)
        shape = (n_paths, n_commodities)
        
        if isnothing(enabled_flows) 
            enabled_flows = ones(Bool, shape)
        elseif size(enabled_flows) != shape
            error("Invalid shape $(size(enabled_flows)) for `disabled_flows`, should be $(shape)")
        end
        
        new(n_paths, n_commodities, capacities, design_costs, flow_costs, enabled_flows)
    end
end

struct Solution
    x_sol::Matrix
    z_sol::Vector
end

function create_forward_problem(params::Params, demands::Vector)::Model
    model = Model(Gurobi.Optimizer)

    @variable(model, z[1:params.n_paths], Bin)
    @variable(model, x[1:params.n_paths, 1:params.n_commodities] >= 0)

    @objective(model, Min, params.design_costs' * z + sum(params.flow_costs .* x))

    @constraint(model, [k = 1:params.n_commodities], sum(x[:, k]) .== demands[k])
    @constraint(model, [i = 1:params.n_paths], sum(x[i, :]) <= params.capacities[i] * z[i])
    @constraint(model, [i = 1:params.n_paths, k = 1:params.n_paths; !params.enabled_flows[i, k]], x[i, k] .== 0)

    return model
end

function solve_forward_problem!(model::Model)::Solution
    optimize!(model)

    x_sol = value.(model[:x])
    z_sol = value.(model[:z])

    return Solution(x_sol, z_sol)
end

function create_and_solve_problem(params::Params, demands::Vector ; silent=false)::Solution
    model = create_forward_problem(params, demands)

    if silent 
        set_silent(model)
    end

    return solve_forward_problem!(model)
end


function sol_vector(sol::Solution)::Vector
    flat_xs = reshape(sol.x_sol, (length(sol.x_sol), 1))

    return vec(vcat(flat_xs, sol.z_sol))
end

end
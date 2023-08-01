module Forward

using JuMP
using Gurobi
using LinearAlgebra 

export Params, Solution
export sol_vector, create_and_solve_problem, create_and_solve_flow_problem

struct Params
    n_paths::Int
    n_commodities::Int

    capacities::AbstractVector{Number}

    design_costs::AbstractVector
    flow_costs::AbstractMatrix

    enabled_flows::AbstractMatrix{Bool}

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
    objective_value::Number
end

function create_forward_problem(params::Params, demands; gurobi_env=nothing)::Model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))

    @variable(model, z[1:params.n_paths], Bin)
    @variable(model, x[1:params.n_paths, 1:params.n_commodities] >= 0)
    @objective(model, Min, params.design_costs' * z + sum(params.flow_costs .* x))

    @constraint(model, [k = 1:params.n_commodities], sum(x[:, k]) .== demands[k])
    @constraint(model, [i = 1:params.n_paths], sum(x[i, :]) <= params.capacities[i] * z[i])
    @constraint(model, [i = 1:params.n_paths, k = 1:params.n_commodities; !params.enabled_flows[i, k]], x[i, k] .== 0)

    return model
end

function solve_forward_problem!(model::Model, silent)::Solution
    if silent 
        set_silent(model)
    end

    optimize!(model)

    x_sol = value.(model[:x])
    z_sol = value.(model[:z])
    objective = objective_value(model)

    return Solution(x_sol, z_sol, objective)
end

function create_and_solve_problem(params::Params, demands ; silent=false, gurobi_env=nothing)::Solution
    model = create_forward_problem(params, demands, gurobi_env=gurobi_env)
    return solve_forward_problem!(model, silent)
end


function sol_vector(sol::Solution)::Vector
    flat_xs = reshape(sol.x_sol, (length(sol.x_sol), 1))

    return vec(vcat(flat_xs, sol.z_sol))
end

# =========================================================
# Flow problem
# =========================================================

function create_and_solve_flow_problem(params::Params, demands, z_sol; recourse_capacity=1_000_000_000, recourse_flow_cost=1_000_000_000, silent=true, gurobi_env=nothing)
    flow_model = create_flow_problem(params, demands, z_sol, recourse_capacity, recourse_flow_cost, gurobi_env=gurobi_env)
    return solve_forward_problem!(flow_model, silent)
end

function create_flow_problem(params::Params, demands, z_sol, recourse_capacity, recourse_flow_cost; gurobi_env=nothing)::Model
    recourse_params, recourse_z_sol = add_recourse_path(params, z_sol, recourse_capacity, recourse_flow_cost)
    design_model = create_forward_problem(recourse_params, demands, gurobi_env=gurobi_env)
    flow_model = fix_design_variables!(design_model, recourse_z_sol)

    return flow_model
end

function add_recourse_path(params::Params, z_sol, recourse_capacity, recourse_flow_cost)
    recourse_params = Params(
        n_paths=params.n_paths + 1, 
        n_commodities=params.n_commodities, 
        capacities=vcat([recourse_capacity], params.capacities), 
        design_costs=vcat([0], params.design_costs), 
        flow_costs=vcat(fill(recourse_flow_cost, (1, params.n_commodities)), params.flow_costs), 
        enabled_flows=vcat(fill(true, (1, params.n_commodities)), params.enabled_flows))

    recourse_z_sol = vcat([1.0], z_sol)

    return recourse_params, recourse_z_sol
end

function fix_design_variables!(model::Model, z_sol)
    zs = model[:z]
    @constraint(model, [i = 1:length(zs)], zs[i] == z_sol[i])

    return model
end








end
"""Solving the congestion game (on di-graphs) by convex optimizaion using JuMP,
and compare to the results of message passing.
Integer flow."""

using LightGraphs
using JuMP, Ipopt, Juniper, Cbc
using LinearAlgebra
using CSV
using DataFrames
using Random

include("../GraphUtil.jl")
include("../CongestionGameProb.jl")

## Add this line if "with_optimizer" is deprecated in the version of JuMP being run:
with_optimizer = optimizer_with_attributes  ## tested on JuMP v.1.1.0

ϕ_func = CongestionGameProb.ϕ_q
xℓ_func = CongestionGameProb.xℓ_q
Φ_and_H = CongestionGameProb.Φ_and_H_q


function cal_opt_flow_nl_cost!(model, xv, B, s, q, d, cpc, tf, Λ, τ_in; level="lower")
    """Compute the optimal flow.
    Now model and xv are local variables,
    so one should avoid using the @eval macro, which only deals with global variable.
    Allow generic nonlinear cost function."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    if level == "lower"
        ## Nash equilibrium with toll τ_in:
        function ϕ_func_wτ(x...)
            """x = {...x[a, e], x[a, e+1], x[a, e+2]...}"""
            Φ = 0.
            for e in 1:no_of_edges
                xe = 0.
                for a in 1:no_of_class
                    xe += x[(a-1)*no_of_edges + e]
                end
                Φ += ϕ_func(xe, cpc[e], tf[e], s, q) + τ_in[e] * xe
            end
            return Φ
        end

        ## Specify the cost function to the JuMP model:
        register(model, :ϕ_func_wτ, no_of_class*no_of_edges, ϕ_func_wτ, autodiff=true)
        # @eval @NLobjective(model, Min, $( Expr(:call, :ϕ_func_wτ, [Expr(:ref, :xv, i) for i=1:no_of_class*no_of_edges]...) ))
        @NLobjective(model, Min, ϕ_func_wτ(xv...))

    else
        ## Global optimum, evaluated without toll:
        function H_rBPR_no_τ(x...)
            H = 0.
            for e in 1:no_of_edges
                xe = 0.
                for a in 1:no_of_class
                    xe += x[(a-1)*no_of_edges + e]
                end
                H += xℓ_func(xe, cpc[e], tf[e], s, q)
            end
            return H
        end

        ## Specify the cost function to the JuMP model:
        register(model, :H_rBPR_no_τ, no_of_class*no_of_edges, H_rBPR_no_τ, autodiff=true)
        # @eval @NLobjective(model, Min, $( Expr(:call, :H_rBPR_no_τ, [Expr(:ref, :xv, i) for i=1:no_of_class*no_of_edges]...) ))
        @NLobjective(model, Min, H_rBPR_no_τ(xv...))

    end

    ## Specify the flow conservation constraints:
    for a in 1:no_of_class, i in 1:no_of_nodes
        if i == d[a]
            continue    ## the destination is an absorbing boundary, aka, unconstrained
        end
        @constraint(model, dot(B[i, :], xv[(a-1)*no_of_edges+1 : a*no_of_edges]) + Λ[i, a] == 0.)
    end

    runtime = @elapsed optimize!(model)

    ## Extract the solutions:
    xv_values = ones(no_of_class, no_of_edges)
    for a in 1:no_of_class, e in 1:no_of_edges
        m = (a-1)*no_of_edges + e
        xv_values[a, e] = value(xv[m])
    end

    return xv_values, runtime
end


function cal_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ_in; level="lower")
    """Compute the optimal flow.
    Now model and xv are local variables,
    so one should avoid using the @eval macro, which only deals with global variable.
    Quadratic cost."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    if level == "lower"
        ## Nash equilibrium with toll τ_in:
        function ϕ_func_wτ(x...)
            """x = {...x[a, e], x[a, e+1], x[a, e+2]...}"""
            Φ = 0.
            for e in 1:no_of_edges
                xe = 0.
                for a in 1:no_of_class
                    xe += x[(a-1)*no_of_edges + e]
                end
                Φ += ϕ_func(xe, cpc[e], tf[e], s, q) + τ_in[e] * xe
            end
            return Φ
        end

        ## Specify the cost function to the JuMP model:
        @objective(model, Min, ϕ_func_wτ(xv...))

    else
        ## Global optimum, evaluated without toll:
        function H_rBPR_no_τ(x...)
            H = 0.
            for e in 1:no_of_edges
                xe = 0.
                for a in 1:no_of_class
                    xe += x[(a-1)*no_of_edges + e]
                end
                H += xℓ_func(xe, cpc[e], tf[e], s, q)
            end
            return H
        end

        ## Specify the cost function to the JuMP model:
        @objective(model, Min, H_rBPR_no_τ(xv...))

    end

    ## Specify the flow conservation constraints:
    for a in 1:no_of_class, i in 1:no_of_nodes
    # for a in [1], i in [1,3]
        if i == d[a]
            continue    ## the destination is an absorbing boundary, aka, unconstrained
        end
        @constraint(model, dot(B[i, :], xv[(a-1)*no_of_edges+1 : a*no_of_edges]) + Λ[i, a] == 0)
    end

    # for i in 1:no_of_class*no_of_edges
    #     @constraint(model, xv[i] >= 0)
    # end

    runtime = @elapsed optimize!(model)

    ## Extract the solutions:
    xv_values = ones(no_of_class, no_of_edges)
    for a in 1:no_of_class, e in 1:no_of_edges
        m = (a-1)*no_of_edges + e
        xv_values[a, e] = value(xv[m])
    end

    return xv_values, runtime
end


## Read the road network (from the subdirectory '../directed_road_networks/'):
nn = 20
subdir = "../directed_road_networks/"
graph_name = "rrg_n$(nn)_d3_seed100_dg"

node_data, edge_data, graph = GraphUtil.read_graph_from_csv(subdir, graph_name, true)   # return a directed graph
no_of_nodes, no_of_edges = nv(graph), ne(graph)
max_deg, deg, outdeg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, B = GraphUtil.digraph_repr(graph, edge_data)

## Define the problem instance:
no_of_class = 1
load = 4
frac = 0.1
seed_Λ = 100

d, cpc, tf, Λ0, τ =
    CongestionGameProb.assign_uniform_edge_weights_and_random_src_dst_MD_atomic(no_of_edges, no_of_nodes, no_of_class,
        frac, seed_Λ, "select")    ## Set uniform edge weights, assign random source and destination
Λ = Λ0 * load

## Parameters for the cost function:
s, q = 0.25, 0.

## Parameters for MP algorithms:
seed_alg = 100
lr = 0.2    ## learning rate for updating the working points; not used in the current version
check_intv = 4 * no_of_edges * no_of_class
rounds = 15
max_iter = 10 * check_intv * rounds
step_size = 0.1

## Solvers:
optimizer = GLPK.Optimizer
nl_solver= optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
mip_solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 0)

## Nash equilibrium with zero toll:
# model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))
model = Model(optimizer_with_attributes(optimizer, "mip_solver"=>mip_solver))
@variable(model, xv[1:no_of_class*no_of_edges], integer=true)
@constraint(model, xv .>= zeros(Int, no_of_class*no_of_edges))
xn_values, runtime1 = cal_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ, level="lower")
Φn, Hn = Φ_and_H(transpose(xn_values) * ones(no_of_class), cpc, tf, s, q, zeros(no_of_edges), edge_list)


## Social optimum:
# model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))
model = Model(optimizer_with_attributes(optimizer, "mip_solver"=>mip_solver))
@variable(model, xv[1:no_of_class*no_of_edges], integer=true)
@constraint(model, xv .>= zeros(Int, no_of_class*no_of_edges))
xg_values, runtime2 = cal_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ, level="upper")
Φg, Hg = Φ_and_H(transpose(xg_values) * ones(no_of_class), cpc, tf, s, q, zeros(no_of_edges), edge_list)


## Nash equilibrium with specified toll:
specified_toll = false
if specified_toll
    dir_tolls = "./results/tolls/"      ## Directory of tolls
    seed_alg = 200                      ## seed for MP algorithm
    r = 2                               ## which round of the bilevel MP process to consider

    file_name = "toll_" * graph_name * "_q$(q)_nc$(no_of_class)_load$(load)_frac$(frac)_seedlam$(seed_Λ)" *
                    "_seedalg$(seed_alg)_round$(r).csv"
    df = CSV.read(dir_tolls * file_name , DataFrame)
    τ = df[:, 2]
    # τ[τ .< 0.11] .= 0     ## (arbitrarily) filter out some small tolls

    model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))
    @variable(model, xv[1:no_of_class*no_of_edges]) #, integer=true)
    @constraint(model, xv .>= zeros(Int, no_of_class*no_of_edges))
    xv_values, runtime3 = cal_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ, level="lower")
    Φn_wτ, Hn_wτ = Φ_and_H(transpose(xv_values) * ones(no_of_class), cpc, tf, s, q, zeros(no_of_edges), edge_list)
end


## Outputs:
println("Potential function at Nash equilibrium :               Φn = $(Φn).")
println("Social cost at Nash equilibrium:                       Hn = $(Hn)")
println("Social cost at social optimmum:                        Hg = $(Hg)")
if specified_toll
    println("Social cost at Nash equilibrium with specified tolls:  Hn = $(Hn_wτ)")
end

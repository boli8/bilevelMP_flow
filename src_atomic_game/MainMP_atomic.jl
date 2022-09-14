using LightGraphs
using JuMP, Ipopt, Juniper, Cbc, LinearAlgebra
using Random
using CSV
using DataFrames
# using Plots

include("../GraphUtil.jl")
include("../CongestionGameProb.jl")
include("CongestionGameMP_atomic.jl")


Φ_and_H = CongestionGameProb.Φ_and_H_q
ϕp_func = CongestionGameProb.ϕp_q
ϕpp_func = CongestionGameProb.ϕpp_q


function run_lower_mp_once(d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr, seed_alg=100)
    """Run lower MP for one set of parameters."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    K = 3
    xs, E, xw, xwtot, ϕpp, ϕp = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class, K)
    state = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    CongestionGameMP.init_xw_by_opt_flow!(xw, xwtot, xs, ϕpp, ϕp, edge_list, K,
                        B, s, q, d, cpc, tf, Λ, τ, "lower", continuous=true)

    converg, tc = CongestionGameMP.lower_MP_rand_seq!(xs, E, xw, xwtot, ϕpp, ϕp, d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg,
                                          edge_list, K, state, max_iter, check_intv, lr, seed_alg, "lower")
    println("Lower MP, converge=$converg, tc=$tc.")

    xw_n, xwtot_n = CongestionGameMP.cal_all_xw(xs, E, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list, K)
    xn_vec = GraphUtil.xmat_to_xvec(xwtot_n, edge_list)
    Φn, Hn = Φ_and_H(xn_vec, cpc, tf, s, q, τ, edge_list)

    return xs, E, xw, xwtot, ϕpp, ϕp, state, xn_vec, Φn, Hn, converg, tc
end


function run_bilevel_mp_once(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr, seed_alg=100)
    """Run bilevel MP for one set of parameters."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    K = 3
    xs, E, xw, xwtot, ϕpp, ϕp = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class, K)
    H, σpp, σp = CongestionGameMP.init_upper_messages(B, xwtot, cpc, tf, s, q, no_of_class, K)
    state_up = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    CongestionGameMP.init_xw_by_opt_flow!(xw, xwtot, xs, ϕpp, ϕp, edge_list, K,
                        B, s, q, d, cpc, tf, Λ, τ, "lower", continuous=true)

    converg_up, tc_up, xs, E, xw, xwtot, H, ϕpp, ϕp, τ =
        CongestionGameMP.bilevel_MP_rand_seq!(xs, E, xw, xwtot, ϕpp, ϕp, H, σpp, σp,
                                       d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, K,
                                       state_up, max_iter, check_intv, lr, seed_alg, true, 0.1, save_toll=false)
    println("Bilevel MP, converge=$(converg_up), tc=$(tc_up).")

    xw_g, xwtot_g = CongestionGameMP.cal_all_xw(xs, E, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list, K)
    xg_vec = GraphUtil.xmat_to_xvec(xwtot_g, edge_list)
    Φg, Hg = Φ_and_H(xg_vec, cpc, tf, s, q, τ, edge_list)

    return xs, E, xw, H, state_up, τ, xg_vec, Hg, converg_up, tc_up
end


function run_bilevel_mp_once_save_τ(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr,
        graph_name, load, frac, seed_Λ, seed_alg, step_size, dir_tolls)
    """Run bilevel MP for one set of parameters,
    Save toll in intermediate times."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    max_iter_each_round = 10 * check_intv
    rounds = max_iter ÷ max_iter_each_round

    K = 3
    xs, E, xw, xwtot, ϕpp, ϕp = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class, K)
    H, σpp, σp = CongestionGameMP.init_upper_messages(B, xwtot, cpc, tf, s, q, no_of_class, K)
    state_up = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    CongestionGameMP.init_xw_by_opt_flow!(xw, xwtot, xs, ϕpp, ϕp, edge_list, K,
                        B, s, q, d, cpc, tf, Λ, τ, "lower", continuous=true)

    ## without updating toll:
    converg_up, tc_up, xs, E, xw, xwtot, H, ϕpp, ϕp, τ =
        CongestionGameMP.bilevel_MP_rand_seq!(xs, E, xw, xwtot, ϕpp, ϕp, H, σpp, σp,
                                       d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, K,
                                       state_up, max_iter_each_round, check_intv, lr, seed_alg, false, step_size, save_toll=true)

    for r in 1:rounds
        converg_up, tc_up, xs, E, xw, xwtot, H, ϕpp, ϕp, τ =
            CongestionGameMP.bilevel_MP_rand_seq!(xs, E, xw, xwtot, ϕpp, ϕp, H, σpp, σp,
                                           d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, K,
                                           state_up, max_iter_each_round, check_intv, lr, -1, true, step_size, save_toll=true)

       open(dir_tolls * "toll_" * graph_name * "_q$(q)_nc$(no_of_class)_load$(load)_frac$(frac)_seedlam$(seed_Λ)" *
            "_seedalg$(seed_alg)_round$(r).csv", "w") do io
           write(io, "edge,toll\n")
           for e in 1:no_of_edges
               write(io, "$(e),$(τ[e])\n")
           end
       end
    end

    xw_g, xwtot_g = CongestionGameMP.cal_all_xw(xs, E, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list, K)
    xg_vec = GraphUtil.xmat_to_xvec(xwtot_g, edge_list)
    Φg, Hg = Φ_and_H(xg_vec, cpc, tf, s, q, τ, edge_list)

    return xs, E, xw, H, state_up, τ, xg_vec, Hg, converg_up, tc_up
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

τmin = zeros(no_of_edges)
τmax = ones(no_of_edges) * 10.

## Parameters for the cost function:
s, q = 0.25, 0.

## Parameters for MP algorithms:
seed_alg = 200
lr = 0.2    ## learning rate for updating the working points; not used in the current version
check_intv = 4 * no_of_edges * no_of_class
rounds = 15
max_iter = 10 * check_intv * rounds
step_size = 0.1

## Run Lower-MP:
run_lower_mp = true
if run_lower_mp
    xs, E, xw, xwtot, ϕpp, ϕp, state, xn_vec, Φn, Hn, converg, tc =
        run_lower_mp_once(d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr, seed_alg)
end

## Run Bilevel-MP:
run_bilevel_mp = true
if run_bilevel_mp
    xs, E, xw, H, state_up, τ, xg_vec, Hg, converg_up, tc_up =
        run_bilevel_mp_once(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr, seed_alg)
end

## Run Bilevel-MP and save intermediate tolls:
run_bilevel_mp_save_toll = true
dir_tolls = "./results/tolls/"      ## Directory of tolls
if run_bilevel_mp_save_toll
    xs, E, xw, H, state_up, τ, xg_vec, Hg, converg_up, tc_up =
        run_bilevel_mp_once_save_τ(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr,
            graph_name, load, frac, seed_Λ, seed_alg, step_size, dir_tolls)
end


## Outputs:
if run_lower_mp
    println("Potential function at Nash equilibrium :               Φn = $(Φn).")
    println("Social cost at Nash equilibrium:                       Hn = $(Hn)")
end

if run_bilevel_mp
    println("Social cost at social optimmum:                        Hg = $(Hg)")
end

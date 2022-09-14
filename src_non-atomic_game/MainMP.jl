using LightGraphs
using Random
using CSV

include("../GraphUtil.jl")
include("../CongestionGameProb.jl")
include("CongestionGameMP.jl")

Φ_and_H = CongestionGameProb.Φ_and_H_q


function run_lower_mp_once(d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr)
    seed_alg = 100
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class)
    state = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    converg, tc = CongestionGameMP.lower_MP_rand_seq!(αβ_LR, leaf_lw, α, β, xw, xwtot, ϕpp, ϕp,
                                            d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg,
                                            edge_list, state, max_iter, check_intv, lr, seed_alg)
    println("Lower MP, converge=$converg, tc=$tc.")

    xw_n, xwtot_n = CongestionGameMP.cal_all_xw(αβ_LR, leaf_lw, α, β, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list)
    xn_vec = GraphUtil.xmat_to_xvec(xwtot_n, edge_list)
    Φn, Hn = Φ_and_H(xn_vec, cpc, tf, s, q, τ, edge_list)

    return α, β, xw, leaf_lw, αβ_LR, state, xn_vec, Φn, Hn, converg, tc
end


function run_bilevel_mp_once(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list,
            max_iter, check_intv, lr, graph_name, load, frac, seed_Λ, seed_alg=400)
    """Run bilevel MP for one set of parameters."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class)
    γ, δ2, σpp, σp, leaf_up, γδ_LR = CongestionGameMP.init_upper_messages(B, xwtot, cpc, tf, s, q, no_of_class)
    state_up = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    ## without updating toll:
    converg_up, tc_up, α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR =
        CongestionGameMP.bilevel_MP_rand_seq!(αβ_LR, leaf_lw, α, β, xw, xwtot, ϕpp, ϕp, γδ_LR, leaf_up, γ, δ2, σpp, σp,
                                       d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list,
                                       state_up, max_iter, check_intv, lr, seed_alg, true, save_toll=false)
    println("Bilevel MP, converge=$(converg_up), tc=$(tc_up).")

    xw_g, xwtot_g = CongestionGameMP.cal_all_xw(αβ_LR, leaf_lw, α, β, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list)
    xg_vec = GraphUtil.xmat_to_xvec(xwtot_g, edge_list)
    Φg, Hg = Φ_and_H(xg_vec, cpc, tf, s, q, τ, edge_list)

    return α, β, xw, leaf_lw, αβ_LR, γ, δ2, leaf_up, γδ_LR, state_up, τ, xg_vec, Hg, converg_up, tc_up
end


function run_bilevel_mp_once_save_τ(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list,
            max_iter, check_intv, lr, graph_name, load, frac, seed_Λ, seed_alg, dir_tolls)
    """Run bilevel MP for one set of parameters,
    Save toll in intermediate times."""
    no_of_edges = length(cpc)
    no_of_nodes, no_of_class = size(Λ)

    α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR = CongestionGameMP.init_lower_messages(B, cpc, tf, s, q, no_of_class)
    γ, δ2, σpp, σp, leaf_up, γδ_LR = CongestionGameMP.init_upper_messages(B, xwtot, cpc, tf, s, q, no_of_class)
    state_up = ones(Float64, (max_iter ÷ check_intv, 6)) * 10

    max_iter_each_round = 100 * check_intv
    rounds = max_iter ÷ max_iter_each_round

    ## without updating toll:
    converg_up, tc_up, α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR =
        CongestionGameMP.bilevel_MP_rand_seq!(αβ_LR, leaf_lw, α, β, xw, xwtot, ϕpp, ϕp, γδ_LR, leaf_up, γ, δ2, σpp, σp,
                                       d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list,
                                       state_up, max_iter_each_round, check_intv, lr, seed_alg, false, save_toll=true)

    for r in 1:rounds
        converg_up, tc_up, α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR =
            CongestionGameMP.bilevel_MP_rand_seq!(αβ_LR, leaf_lw, α, β, xw, xwtot, ϕpp, ϕp, γδ_LR, leaf_up, γ, δ2, σpp, σp,
                                           d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list,
                                           state_up, max_iter_each_round, check_intv, lr, -1, true, save_toll=true)

        open(dir_tolls * "toll_" * graph_name * "_q$(q)_nc$(no_of_class)_load$(load)_frac$(frac)_seedlam$(seed_Λ)" *
             "_seedalg$(seed_alg)_round$(r).csv", "w") do io
            write(io, "edge,toll\n")
            for e in 1:no_of_edges
                write(io, "$(e),$(τ[e])\n")
            end
        end
    end

    # println("Bilevel MP, converge=$(converg_up), tc=$(tc_up).")

    xw_g, xwtot_g = CongestionGameMP.cal_all_xw(αβ_LR, leaf_lw, α, β, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list)
    xg_vec = GraphUtil.xmat_to_xvec(xwtot_g, edge_list)
    Φg, Hg = Φ_and_H(xg_vec, cpc, tf, s, q, τ, edge_list)

    return α, β, xw, leaf_lw, αβ_LR, γ, δ2, leaf_up, γδ_LR, state_up, τ, xg_vec, Hg, converg_up, tc_up
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
load = 4.
frac = 0.05
seed_Λ = 100

d, cpc, tf, Λ0, τ =
    CongestionGameProb.assign_uniform_edge_weights_and_random_src_dst_MD(no_of_edges, no_of_nodes, no_of_class,
        frac, seed_Λ, "select")     ## Set uniform edge weights, assign random source and destination

Λ = Λ0 * load

τmin = zeros(no_of_edges)
τmax = ones(no_of_edges) * 1.

## Parameters for the cost function:
s, q = 0.25, 0.

## Parameters for MP algorithms:
seed_alg = 100
lr = 0.2    ## learning rate for updating the working points
check_intv = 4 * no_of_edges * no_of_class
rounds = 10
max_iter = 100 * check_intv * rounds

## Run Lower-MP:
run_lower_mp = true
if run_lower_mp
    α, β, xw, leaf_lw, αβ_LR, state, xn_vec, Φn, Hn, converg, tc =
        run_lower_mp_once(d, cpc, tf, s, q, Λ, τ, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr)
end

## Run Bilevel-MP:
run_bilevel_mp = true
if run_bilevel_mp
    α, β, xw, leaf_lw, αβ_LR, γ, δ2, leaf_up, γδ_LR, state_up, τ, xg_vec, Hg, converg_up, tc_up =
        run_bilevel_mp_once(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr,
                graph_name, load, frac, seed_Λ, seed_alg)
end

## Run Bilevel-MP and save intermediate tolls:
run_bilevel_mp_save_toll = true
dir_tolls = "./results/tolls/"      ## Directory of tolls
if run_bilevel_mp_save_toll
    α, β, xw, leaf_lw, αβ_LR, γ, δ2, leaf_up, γδ_LR, state_up, τ, xg_vec, Hg, converg_up, tc_up =
        run_bilevel_mp_once_save_τ(d, cpc, tf, s, q, Λ, τ, τmin, τmax, adj_e, B, deg, outdeg, edge_list, max_iter, check_intv, lr,
                graph_name, load, frac, seed_Λ, seed_alg, dir_tolls)
end


## Outputs:
if run_lower_mp
    println("Potential function at Nash equilibrium :               Φn = $(Φn).")
    println("Social cost at Nash equilibrium:                       Hn = $(Hn)")
end

if run_bilevel_mp
    println("Social cost at social optimmum:                        Hg = $(Hg)")
end

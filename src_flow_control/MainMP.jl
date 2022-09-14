"""Message passing of flow problem."""

using LightGraphs
using LinearAlgebra
using SparseArrays
using Random
using Statistics

include("../GraphUtil.jl")
include("FlowLossHinge.jl")
include("FlowMP.jl")
include("FlowLPL.jl")


## read the graph:
subdir = "../undirected_networks/"
graph_name = "rrg_n100_d3_seed100"
node_data, edge_data, graph = GraphUtil.read_graph_from_csv(subdir, graph_name, false)
no_of_nodes, no_of_edges = nv(graph), ne(graph)
max_deg, deg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, adj_mat, B = GraphUtil.undigraph_repr(graph, edge_data)


## setting the problem:
no_of_target = 5
threshold = 0.2
seed_Λ = 100

frac = 1 / no_of_nodes  ## fraction of source node
d, r, Λ = FlowMP.assign_edge_weights_and_src_dst(no_of_edges, no_of_nodes, frac, seed_Λ, -1)
targets = FlowMP.assign_target(no_of_edges, no_of_target, 100)

d = -999    ## set d < 0: use Method II (Sec. III.A.1 of SM) to treat boundary condition of the reference node
rmin, rmax = 0.9, 1.1


## MP algorithm parameters
max_iter = 200 * (2*no_of_edges)^2
check_intv = 4*no_of_edges
seed_alg = 500


## Lowever level MP:
α, ytil = FlowMP.init_messages(adj_mat)
state = zeros(Float64, (max_iter ÷ check_intv, 2))
converg, tc = FlowMP.lower_MP!(α, ytil, d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list,
                state, max_iter, check_intv, seed_alg, true)
yopt0 = FlowMP.cal_yopt_all_edges(α, ytil, r, edge_list)
E0 = energy_func(yopt0, r)
H0, S0 = objective_upper_and_tuned_edges(yopt0, yopt0, targets, threshold)


## Laplacian solver for the original flow problem:
L, Lr, μ, ysoln0 = FlowLPL.laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
E0L = energy_func(ysoln0, r)
H0L, S0L = objective_upper_and_tuned_edges(ysoln0, ysoln0, targets, threshold)


## Bilevel MP:
α, ytil = FlowMP.init_messages(adj_mat)
state = zeros(Float64, (max_iter ÷ check_intv, 2))
gα, gy, gr = FlowMP.init_gradient(adj_mat, targets)
converg_up, tc_up = FlowMP.bilevel_MP!(α, ytil, gα, gy, gr, yopt0, targets, threshold, rmin, rmax,
                      d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list,
                      state, max_iter, check_intv, seed_alg+100, true)
yopt1 = FlowMP.cal_yopt_all_edges(α, ytil, r, edge_list)
E1 = energy_func(yopt1, r)
H1, S1 = objective_upper_and_tuned_edges(yopt1, yopt0, targets, threshold)


## Laplacian solver for the optimized flow problem:
L, Lr, μ, ysoln1 = FlowLPL.laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
E1L = energy_func(ysoln1, r)
H1L, S1L = objective_upper_and_tuned_edges(ysoln1, ysoln0, targets, threshold)


## Output:
println("Before control, solution by message-passing:   cost_func = $(H0)")
println("Before control, solution by Laplacian solver:  cost_func = $(H0L)")
println("After Bilevel message-passing flow control:    cost_func = $(H1)")
println("After control, solution by Laplacian solver:   cost_func = $(H1L)")

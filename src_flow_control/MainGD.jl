"""Gradient descent of flow problem."""

using LightGraphs
using LinearAlgebra
using SparseArrays
using Random
using CSV
using DataFrames

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


## Laplacian solver for the original flow problem:
L, Lr, μ, ysoln0 = FlowLPL.laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
E0L = energy_func(ysoln0, r)
H0L, S0L = objective_upper_and_tuned_edges(ysoln0, ysoln0, targets, threshold)


## Gradient descent method:
step_size = 0.2
r, H0L, S0L, H1L, S1L = FlowLPL.gradient_descent_update(targets, threshold, rmin, rmax,
                            d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list, step_size)


## Output:
println("Before control, solution by Laplacian solver:  cost_func = $(H0L)")
println("After gradient-descent flow control:           cost_func = $(H1L)")

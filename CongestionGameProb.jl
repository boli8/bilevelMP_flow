module CongestionGameProb
"""Functions for defining a congestion game problem."""

using DataFrames
using Random

export ϕ_q, ϕp_q, ϕpp_q, xℓ_q, xℓp_q, xℓpp_q, Φ_and_H_q


"""Uniform assignment of edge weights,
random assignment of source and destination.
d -> index of destination,
tf -> free traveling time,
cpc -> capacity,
Λ -> source,
τ -> toll."""
function assign_uniform_edge_weights_and_random_src_dst(no_of_edges::Int,
            no_of_nodes::Int, frac::Float64=0.4, seed::Int=100, src_set="all")
    Random.seed!(seed)
    # d::Int = rand(1:no_of_nodes)    # randomly pick a node as the destination
    d::Int = randperm(no_of_nodes)[1]
    cpc::Array{Float64, 1} = ones(Float64, no_of_edges)
    tf::Array{Float64, 1} = ones(Float64, no_of_edges)
    Λ::Array{Float64, 1} = zeros(Float64, no_of_nodes)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)

    if src_set == "all"
        Λ = rand(no_of_nodes)
    else
        no_of_src = Int( ceil( no_of_nodes*frac ) )
        set_of_src = randperm(no_of_nodes)[1:no_of_src]
        Λ[set_of_src] .= 1.
    end

    Λ[d] = -sum(Λ[:]) + Λ[d]

    return d, cpc, tf, Λ, τ
end


"""Uniform assignment of edge weights,
random assignment of source and destination.
Multiple destination.
src_set = "all":     all nodes have some traffic demand in a certain class.
src_set = "select":  a fraction of nodes (indicate by frac) have some traffic demand.

d -> index of destination,
tf -> free traveling time,
cpc -> capacity,
Λ -> source,
τ -> toll."""
function assign_uniform_edge_weights_and_random_src_dst_MD(no_of_edges::Int,
            no_of_nodes::Int, no_of_class::Int, frac::Float64=0.1, seed::Int=100, src_set="all")
    Random.seed!(seed)
    # d::Array{Int, 1} = rand(1:no_of_nodes, no_of_class)    ## randomly pick a node as the destination for each class
    d::Array{Int, 1} = randperm(no_of_nodes)[1:no_of_class] ## to avoid choosing the same destination for different classes
    cpc::Array{Float64, 1} = ones(Float64, no_of_edges)
    tf::Array{Float64, 1} = ones(Float64, no_of_edges)
    Λ::Array{Float64, 2} = zeros(Float64, no_of_nodes, no_of_class)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)
    for a in 1:no_of_class
        if src_set == "all"
            Λ[:, a] = rand(no_of_nodes)
        else
            no_of_src = Int( ceil( no_of_nodes*frac ) )
            set_of_src = randperm(no_of_nodes)[1:no_of_src]
            Λ[set_of_src, a] .= 1.
        end
        Λ[d[a], a] = -sum(Λ[:, a]) + Λ[d[a], a]
    end

    return d, cpc, tf, Λ, τ
end


"""
Integer flow: integer Λ.
Uniform assignment of edge weights,
random assignment of source and destination.
Multiple destination.
src_set = "all":     all nodes have some traffic demand in a certain class.
src_set = "select":  a fraction of nodes (indicate by frac) have some traffic demand.

d -> index of destination,
tf -> free traveling time,
cpc -> capacity,
Λ -> source,
τ -> toll."""
function assign_uniform_edge_weights_and_random_src_dst_MD_atomic(no_of_edges::Int,
            no_of_nodes::Int, no_of_class::Int, frac::Float64=0.1, seed::Int=100, src_set="all")
    Random.seed!(seed)
    # d::Array{Int, 1} = rand(1:no_of_nodes, no_of_class)    ## randomly pick a node as the destination for each class
    d::Array{Int, 1} = randperm(no_of_nodes)[1:no_of_class] ## to avoid choosing the same destination for different classes
    cpc::Array{Float64, 1} = ones(Float64, no_of_edges)
    tf::Array{Float64, 1} = ones(Float64, no_of_edges)
    Λ::Array{Int, 2} = zeros(Int, no_of_nodes, no_of_class)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)
    for a in 1:no_of_class
        if src_set == "all"
            Λ[:, a] = rand(0:3, no_of_nodes)     ## just a try
        else
            no_of_src = Int( ceil( no_of_nodes*frac ) )
            set_of_src = randperm(no_of_nodes)[1:no_of_src]
            Λ[set_of_src, a] .= 1
        end
        Λ[d[a], a] = -sum(Λ[:, a]) + Λ[d[a], a]
    end

    return d, cpc, tf, Λ, τ
end


"""
Integer flow: integer Λ.
Uniform assignment of edge weights,
random assignment of source and destination.
Multiple destinations are specified.
src_set = "all":     all nodes have some traffic demand in a certain class.
src_set = "select":  a fraction of nodes (indicate by frac) have some traffic demand.

d -> index of destination,
tf -> free traveling time,
cpc -> capacity,
Λ -> source,
τ -> toll."""
function assign_uniform_edge_weights_and_random_src_dst_MD_atomic_spec_d(d::Array{Int64, 1}, no_of_edges::Int,
            no_of_nodes::Int, no_of_class::Int, frac::Float64=0.1, seed::Int=100, src_set="all")
    Random.seed!(seed)
    cpc::Array{Float64, 1} = ones(Float64, no_of_edges)
    tf::Array{Float64, 1} = ones(Float64, no_of_edges)
    Λ::Array{Int, 2} = zeros(Int, no_of_nodes, no_of_class)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)
    for a in 1:no_of_class
        if src_set == "all"
            Λ[:, a] = rand(0:3, no_of_nodes)
        else
            no_of_src = Int( ceil( no_of_nodes*frac ) )
            set_of_src = randperm(no_of_nodes)[1:no_of_src]
            Λ[set_of_src, a] .= 1
        end
        Λ[d[a], a] = -sum(Λ[:, a]) + Λ[d[a], a]
    end

    return d, cpc, tf, Λ, τ
end


"""Random assignment of edge weights,
random assignment of source and destination.
Multiple destination.
src_set = "all":     all nodes have some traffic demand in a certain class.
src_set = "select":  a fraction of nodes (indicate by frac) have some traffic demand."""
function assign_random_edge_weights_and_random_src_dst_MD(no_of_edges::Int,
            no_of_nodes::Int, no_of_class::Int, frac::Float64=0.1, seed::Int=100, src_set="all")
    Random.seed!(seed)
    d::Array{Int, 1} = randperm(no_of_nodes)[1:no_of_class] ## to avoid choosing the same destination for different classes
    cpc::Array{Float64, 1} = ones(Float64, no_of_edges)*0.1 + rand(no_of_edges)
    tf::Array{Float64, 1} = ones(Float64, no_of_edges)*0.1 + rand(no_of_edges)
    Λ::Array{Float64, 2} = zeros(Float64, no_of_nodes, no_of_class)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)
    for a in 1:no_of_class
        if src_set == "all"
            Λ[:, a] = rand(no_of_nodes)
        else
            no_of_src = Int( ceil( no_of_nodes*frac ) )
            set_of_src = randperm(no_of_nodes)[1:no_of_src]
            Λ[set_of_src, a] .= 1.
        end
        Λ[d[a], a] = -sum(Λ[:, a]) + Λ[d[a], a]
    end

    return d, cpc, tf, Λ, τ
end


"""Edge weights and source demand by node and edge data."""
function assign_edge_weight_and_source_by_ta_data(node_data::DataFrame, edge_data::DataFrame)
    no_of_nodes = size(node_data, 1)
    no_of_edges = size(edge_data, 1)

    demand_mat = node_data[:, 3:end]
    demand_tot = [ sum(demand_mat[i, :]) for i in 1:no_of_nodes ]

    d::Array{Int, 1} = [ i for i in 1:no_of_nodes if demand_tot[i] > 1e-5 ]
    no_of_class = length(d)

    cpc::Array{Float64, 1} = [ edge_data[i, 5] for i in 1:no_of_edges ]
    tf::Array{Float64, 1} = [ edge_data[i, 4] for i in 1:no_of_edges ]
    Λ::Array{Float64, 2} = zeros(Float64, no_of_nodes, no_of_class)
    τ::Array{Float64, 1} = zeros(Float64, no_of_edges)
    for a in 1:no_of_class, i in 1:no_of_nodes
        Λ[i, a] = demand_mat[d[a], i]
    end

    return d, cpc, tf, Λ, τ
end


"""Assign the vector of τmin and τmax, by randomly selecting a fraction of edges (efrac)
to allow a maximal toll."""
function assign_toll_bounds(no_of_edges::Int, toll_max::Float64, efrac::Float64, seed::Int=100)
    Random.seed!(seed)
    τmin = zeros(no_of_edges)
    τmax = zeros(no_of_edges)
    no_of_tollable_edges = Int( floor(no_of_edges * efrac) )
    tollable_edges = randperm(no_of_edges)[1:no_of_tollable_edges]
    τmax[tollable_edges] .= toll_max
    return τmin, τmax
end


"""Quadratic cost function."""
function ℓ_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( 1 + s* xe / cpc_e )
end


"""Edge-wise potential function with quadratic cost function."""
function ϕ_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( xe + 0.5*s* xe^2 / cpc_e )
end


"""First derivative of ϕ_q."""
function ϕp_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( 1 + s* xe / cpc_e )
end


"""Second derivative of ϕ_q."""
function ϕpp_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( s / cpc_e )
end


"""Social cost with quadratic cost function, xe*ℓe(xe)."""
function xℓ_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( xe + s* xe^2/cpc_e )
end


"""First derivative of xe*ℓe(xe)."""
function xℓp_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( 1 + 2*s* xe / cpc_e )
end


"""Second derivative of xe*ℓe(xe)."""
function xℓpp_q(xe, cpc_e, tf_e, s, q)
    return tf_e * ( 2*s / cpc_e )
end


"""The potential function of the routing game and social cost for quadratic latency function."""
function Φ_and_H_q(x_vec, cpc, tf, s, q, τ, edge_list)
    Φ::Float64 = 0.
    H::Float64 = 0.
    for e in 1:size(edge_list, 1)
        Φ += ϕ_q(x_vec[e], cpc[e], tf[e], s, q) + τ[e] * x_vec[e]
        H += xℓ_q(x_vec[e], cpc[e], tf[e], s, q)
    end
    return Φ, H
end


end  # modul CongestionGameProb

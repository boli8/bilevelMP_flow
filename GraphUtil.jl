module GraphUtil
"""
Utility function for graphs.
Nodes (vertices) are strictly indexed as 1:size(nv(graph)),
edges (links) are strictly indexed as 1:size(ne(graph)).
"""


using DataFrames
using CSV
using LightGraphs
using SparseArrays

export heaviside, sigmoid, end_node, xvec_to_xmat, xmat_to_xvec, Φ_and_H, index_to_config


"""Heaviside theta function."""
function heaviside(x::Number)::Number
    return (1 + sign(x)) / 2
end


"""Sigmoid function."""
function sigmoid(x::Number)::Number
    return 1. / (1. + exp(-x))
end


"""Transfrom an integer index (=1,2,...,base^n) into a Potts-spin
configuration (also indexed starting with 1).
E.g., if based = 3, then output σ=[σ1,σ2,...σn], where σi=0,2,...,base-1."""
function index_to_config(index::Int, base::Int, n::Int)::Array{Int, 1}
    σ::Array{Int, 1} = zeros(Int, n)
    q::Int = index - 1
    for i in 1:n
        σ[end+1-i] = q % base
        q = q ÷ base
    end
    return σ
end


"""For edge e->(h, t), given one end i, return the other end."""
function end_node(edge_list::DataFrame, e::Int, i::Int)::Int
    h, t = edge_list[e, :]
    if h == i
        j = t
    elseif t == i
        j = h
    else
        j = -1  # to indicate error
    end
    return j
end


"""Read a graph from a csv file in subdir."""
function read_graph_from_csv(subdir::String, graph_name::String, directed::Bool=true)
    graph_name = subdir * graph_name
    node_data = CSV.read(graph_name * "_nodes.csv", DataFrame)     # ::DataFrame
    no_of_nodes::Int = size(node_data, 1)

    if directed
        edge_data = CSV.read(graph_name * "_diedges.csv", DataFrame)
        graph = SimpleDiGraph(no_of_nodes)
    else
        edge_data = CSV.read(graph_name * "_edges.csv", DataFrame)
        graph = SimpleGraph(no_of_nodes)
    end

    for e in 1:size(edge_data, 1)
        i, j = edge_data[e, 2], edge_data[e, 3]
        add_edge!(graph, i, j)
    end

    return node_data, edge_data, graph
end


"""Given a SimpleGraph or a SimpleDiGraph, write a graph to a csv file in subdir.
When given a SimpleGraph but wants to write a di-graph,
assumes both directions are present for an un-directed edge."""
function write_graph_to_csv(graph::AbstractGraph, subdir::String, graph_name::String, directed::Bool=true)
    graph_name = subdir * graph_name
    open(graph_name * "_nodes.csv", "w") do io
        write(io, "node,node_name\n")
        for (xi, i) in enumerate(vertices(graph))
            write(io, "$xi,$i\n")
        end
    end

    if directed
        edge_filename = graph_name * "_diedges.csv"
    else
        edge_filename = graph_name * "_edges.csv"
    end

    open(edge_filename, "w") do io
        write(io, "edge,from_node,to_node\n")
        diedge_count = 1
        for (e, edg) in enumerate(edges(graph))
            i, j = src(edg), dst(edg)
            if typeof(graph) <: SimpleGraph && directed # input an un-digraph, but want to write a digraph
                write(io, "$diedge_count,$i,$j\n")
                write(io, "$(diedge_count+1),$j,$i\n")
                diedge_count += 2
            else
                write(io, "$e,$i,$j\n")
            end
        end
    end
    return nothing
end


"""Representation of directed graphs."""
function digraph_repr(graph::SimpleDiGraph, edge_data::DataFrame)
    max_deg::Int = 0     # def as maximum degree
    outdeg::Array{Int} = zeros(nv(graph))
    deg::Array{Int} = zeros(nv(graph))
    for i in vertices(graph)
        outdeg[i] = outdegree(graph, i)
        deg[i] = outdegree(graph, i) + indegree(graph, i)
        if deg[i] > max_deg
            max_deg = deg[i]
        end
    end

    edge_list::DataFrame = edge_data[:, 2:3]
    edge_indx::Dict = Dict([ (edge_list[e, 1], edge_list[e, 2]) => e
                            for e in 1:size(edge_list, 1)]) # index of an edge in edge_list
    adj_n::Array{Int} = zeros(Int, (nv(graph), max_deg))    # node adjacency list
    adj_e::Array{Int} = zeros(Int, (nv(graph), max_deg))    # edge adjacency list
    adj_e_indx::Dict{Tuple{Int, Int}, Int} = Dict()     # index of an edge in adj_e
    B::SparseMatrixCSC{Int, Int} = spzeros(Int, nv(graph), ne(graph))

    for i in vertices(graph)
        n = 1   # the index of edge e in the list of edges adjacent to node i
        for j in outneighbors(graph, i)
            e = edge_indx[(i, j)]
            adj_n[i, n] = j
            adj_e[i, n] = e
            adj_e_indx[(i, e)] = n
            B[i, e] = -1
            n += 1
        end
        for j in inneighbors(graph, i)
            e = edge_indx[(j, i)]
            adj_n[i, n] = j
            adj_e[i, n] = e
            adj_e_indx[(i, e)] = n
            B[i, e] = 1
            n += 1
        end
    end

    return max_deg, deg, outdeg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, B
end


"""Representation of un-directed graphs."""
function undigraph_repr(graph::SimpleGraph, edge_data::DataFrame)
    max_deg::Int = 0     # def as maximum degree
    deg::Array{Int} = zeros(nv(graph))
    for i in vertices(graph)
        deg[i] = degree(graph, i)   # different from digraph
        if deg[i] > max_deg
            max_deg = deg[i]
        end
    end

    edge_list::DataFrame = edge_data[:, 2:3]
    edge_indx::Dict = Dict(vcat(
                            [(edge_list[e, 1], edge_list[e, 2]) => e for e in 1:size(edge_list, 1)],
                            [(edge_list[e, 2], edge_list[e, 1]) => e for e in 1:size(edge_list, 1)]
                            ))   # index of an edge in edge_list, different from digraph
    adj_n::Array{Int} = zeros(Int, (nv(graph), max_deg))    # node adjacency list
    adj_e::Array{Int} = zeros(Int, (nv(graph), max_deg))    # edge adjacency list
    adj_e_indx::Dict{Tuple{Int, Int}, Int} = Dict()     # index of an edge in adj_e
    B::SparseMatrixCSC{Int, Int} = spzeros(Int, nv(graph), ne(graph))   # usual incidence matrix for un-directed graph
    adj_mat::SparseMatrixCSC{Int, Int} = spzeros(Int, nv(graph), nv(graph))   # usual adjacency matrix for un-directed graph

    for i in vertices(graph)
        n = 1   # the index of edge e in the list of edges adjacent to node i
        for j in neighbors(graph, i)
            e = edge_indx[(i, j)]
            adj_n[i, n] = j
            adj_e[i, n] = e
            adj_e_indx[(i, e)] = n
            adj_mat[i, j] = 1
            n += 1
            if i == edge_list[e, 1]
                B[i, e] = -1     # i = head(e)
            else
                B[i, e] = 1    # i = tail(e)
            end
        end
    end

    return max_deg, deg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, adj_mat, B
end


"""Convert the flow from matrix representation to vector representation.
The flow x_e is taken to be the average of x_mat[i,e] and x_mat[j,e]."""
function xmat_to_xvec(x_mat::SparseMatrixCSC{Float64, Int}, edge_list::DataFrame)::Array{Float64, 1}
    x_vec::Array{Float64, 1} = zeros(Float64, size(edge_list, 1))
    for e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        x_e = ( x_mat[i, e] + x_mat[j, e] ) / 2
        x_vec[e] = x_e
    end
    return x_vec
end


"""Convert the flow from matrix representation to vector representation.
The flow x_e is taken to be the average of x_mat[i,e] and x_mat[j,e].
Integer flow."""
function xmat_to_xvec(x_mat::SparseMatrixCSC{Int, Int}, edge_list::DataFrame)::Array{Int, 1}
    x_vec::Array{Int, 1} = zeros(Int, size(edge_list, 1))
    for e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        x_e = Int( floor( ( x_mat[i, e] + x_mat[j, e] ) / 2 ) )
        x_vec[e] = x_e
    end
    return x_vec
end


"""Convert the flow from vector representation to sparse matrix representation."""
function xvec_to_xmat(x_vec::Array{Union{Missing, Float64},1}, edge_list::DataFrame,
            B::SparseMatrixCSC{Int, Int})::SparseMatrixCSC{Float64, Int}
    x_mat::SparseMatrixCSC{Float64, Int} = abs.(B)
    for e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        x_mat[i, e] = x_vec[e]
        x_mat[j, e] = x_vec[e]
    end
    return x_mat
end


end  # module GraphUtil

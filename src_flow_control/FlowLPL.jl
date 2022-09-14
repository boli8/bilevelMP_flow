module FlowLPL

include("FlowLossHinge.jl")

using DataFrames
using Random
using LinearAlgebra
using SparseArrays


"""Laplacian matrix with edge weight w[e] = 1/r[e]."""
function construct_laplacian(r::Array{Float64, 1}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)
    no_of_edges = length(r)
    no_of_nodes = length(deg)
    L = spzeros(Float64, size(adj_mat)...)

    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        L[i, j] = - 1. / r[e]
        L[j, i] = - 1. / r[e]
    end

    for i in 1:no_of_nodes
        for n in 1:deg[i]
            e = adj_e[i, n]
            L[i, i] += 1. / r[e]
        end
    end

    return L
end


"""Solve the flow-optimization problem by a laplacian solver."""
function laplacian_solver(d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame, reduced::Bool=true)
    no_of_edges = size(edge_list, 1)
    no_of_nodes = length(Λ)

    L = construct_laplacian(r, adj_e, adj_mat, deg, edge_list)
    normal_nodes = [i for i in 1:no_of_nodes if i!=d]
    Lr = L[normal_nodes, normal_nodes]

    if !reduced || d < 0
        ## Working with the full Laplacian:
        μ = L \ Λ
    else
        ## Working with the reduced Laplacian:
        Λr = Λ[normal_nodes]
        μr = Lr \ Λr
        μ = zeros(Float64, no_of_nodes)
        μ[normal_nodes] = μr
    end

    ysoln = zeros(Float64, no_of_edges)
    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        ysoln[e] = (μ[j] - μ[i]) / r[e]
    end

    return L, Lr, μ, ysoln
end


"""Gradient of the pinv of Laplacian w.r.t. r[e]."""
function cal_gradient_DLinvDr(r::Array{Float64, 1}, adj_mat::SparseMatrixCSC{Int, Int},
            edge_list::DataFrame, L::SparseMatrixCSC{Float64, Int}, Linv::Array{Float64, 2})
    no_of_edges = length(r)
    no_of_nodes = size(adj_mat, 1)

    dLinv = [ zeros(Float64, size(adj_mat)...) for e in 1:no_of_edges ]
    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        dL = zeros(Float64, size(adj_mat)...)
        dL[i, i], dL[i, j] = 1, -1
        dL[j, i], dL[j, j] = -1, 1

        dLinv[e] = Linv * dL * Linv / r[e]^2
    end

    return dLinv
end


"""Exact gradient ∂H[etarg]/∂r[e]."""
function cal_gradient_DHDr_exact(y0::Array{Float64, 1}, targets::Array{Int, 1}, threshold::Float64,
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)
    no_of_target = size(targets, 1)
    no_of_edges  = size(edge_list, 1)

    L, Lr, μ, y = laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
    Linv = pinv( Matrix(L) )
    dLinv = cal_gradient_DLinvDr(r, adj_mat, edge_list, L, Linv)

    ∂H_∂y = zeros(Float64, no_of_target)
    for etarg in 1:no_of_target
        ep = targets[etarg]
        ∂H_∂y[etarg] = cal_gradient_DHDy_on_target(ep, y[ep], y0[ep], targets, threshold)
    end

    ∂H_∂r = zeros(Float64, no_of_target, no_of_edges)
    for etarg in 1:no_of_target
        ep = targets[etarg]
        m, n = edge_list[ep, :]
        for e in 1:no_of_edges
            ∂y_∂re = ( (dLinv[e]*Λ)[n] - (dLinv[e]*Λ)[m] ) / r[ep]
            if ep == e
                ∂y_∂re += - ( (Linv*Λ)[n] - (Linv*Λ)[m] ) / r[ep]^2
            end
            ∂H_∂r[etarg, e] = ∂H_∂y[etarg] * ∂y_∂re
        end
    end

    return ∂H_∂r
end


"""Gradient descent of the global objective. Fixed step size."""
function gradient_descent_update(targets, threshold, rmin, rmax, d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list, s=0.1)
    ## Laplacian solver for the original flow problem:
    L, Lr, μ, ysoln0 = laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
    E0L = energy_func(ysoln0, r)
    H0L, S0L = objective_upper_and_tuned_edges(ysoln0, ysoln0, targets, threshold)

    for step in 1:100
        gHr = cal_gradient_DHDr_exact(ysoln0, targets, threshold, d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list)
        r -= s* vec( sum(gHr, dims=1) )

        r = max.(r, rmin)
        r = min.(r, rmax)

        ## Laplacian solver for the optimized flow problem:
        L, Lr, μ, ysoln1 = laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
        E1L = energy_func(ysoln1, r)
        H1L, S1L = objective_upper_and_tuned_edges(ysoln1, ysoln0, targets, threshold)

        println([step, H1L, sum(r)])

        if abs(H1L) < 1e-10
            println("GD succeeds at step $step.")
            break
        end
    end

    ## Laplacian solver for the optimized flow problem:
    L, Lr, μ, ysoln1 = laplacian_solver(d, r, Λ, adj_e, adj_mat, deg, edge_list)
    E1L = energy_func(ysoln1, r)
    H1L, S1L = objective_upper_and_tuned_edges(ysoln1, ysoln0, targets, threshold)

    return r, H0L, S0L, H1L, S1L
end


end  # module FlowLPL

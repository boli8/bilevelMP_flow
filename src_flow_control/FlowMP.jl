module FlowMP

include("FlowLossHinge.jl")

using DataFrames
using Random
using LinearAlgebra
using SparseArrays


"""Uniform assignment of edge weights,
random assignment of source and destination."""
function assign_edge_weights_and_src_dst(no_of_edges::Int,
            no_of_nodes::Int, frac::Float64=0.1, seed::Int=100, d::Int=-1)
    Random.seed!(seed)
    if d < 0
        d::Int = rand(1:no_of_nodes)    # randomly pick a node as the destination
    end

    r::Array{Float64, 1} = ones(Float64, no_of_edges)
    Λ::Array{Float64, 1} = zeros(Float64, no_of_nodes)

    ## Random source of each node:
    # Λ = rand(no_of_nodes)

    ## Randomly select some nodes as source:
    no_of_src = Int( ceil( no_of_nodes*frac ) )
    nodes_excl_d = [i for i in 1:no_of_nodes if i!=d]
    s = randperm(no_of_nodes-1)[no_of_src]
    s = nodes_excl_d[s]
    Λ[s] = 1.

    Λ[d] = -( sum(Λ) - Λ[d] )
    return d, r, Λ
end


"""Randomly select some edges as targets."""
function assign_target(no_of_edges::Int, no_of_target::Int, seed::Int=100)
    Random.seed!(seed)
    targets = randperm(no_of_edges)[1:no_of_target]
    return targets
end


function init_messages(adj_mat::SparseMatrixCSC{Int, Int}, seed::Int=200)
    Random.seed!(seed)
    no_of_nodes = size(adj_mat, 1)
    α::SparseMatrixCSC{Float64, Int} = abs.( rand(no_of_nodes, no_of_nodes) .* adj_mat )
    ytil::SparseMatrixCSC{Float64, Int} = rand(no_of_nodes, no_of_nodes) .* adj_mat
    return α, ytil
end


function init_gradient(adj_mat::SparseMatrixCSC{Int, Int}, targets::Array{Int, 1}, seed::Int=300)
    Random.seed!(seed)
    no_of_nodes = size(adj_mat, 1)
    no_of_targets = size(targets, 1)
    gα = [rand(no_of_nodes, no_of_nodes) .* adj_mat for c in 1:no_of_targets]
    gy = [rand(no_of_nodes, no_of_nodes) .* adj_mat for c in 1:no_of_targets]
    gr = [rand(no_of_nodes, no_of_nodes) .* adj_mat for c in 1:no_of_targets]
    return gα, gy, gr
end


"""Messages at i->j."""
function cal_message_ij(i::Int, e::Int, j::Int, α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, deg::Array{Int})
    if deg[i] == 1
        α_ij = 1e10
        ytil_ij = Λ[i]
        sum_α, sum_ytil = 0., 0.

    elseif i == d
        α_ij = r[e]
        ytil_ij = 0.
        sum_α, sum_ytil = 0., 0.

    else
        sum_α    = 0.
        sum_ytil = 0.
        for m in 1:deg[i]
            k = adj_n[i, m]
            if k == j
                continue
            end
            sum_α    += 1 / α[k, i]
            sum_ytil += ytil[k, i]
        end

        α_ij    = 1 / sum_α + r[e]
        ytil_ij = (Λ[i] + sum_ytil) / (1 + r[e] * sum_α)
    end

    return α_ij, ytil_ij, sum_α, sum_ytil
end


"""One parallel pass of the lower level message passing."""
function lower_MP_one_parallel_pass(α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)
    no_of_nodes::Int, no_of_edges::Int = length(Λ), length(r)

    α_next::SparseMatrixCSC{Float64, Int} = spzeros(Float64, size(adj_mat)...)
    ytil_next::SparseMatrixCSC{Float64, Int} = spzeros(Float64, size(adj_mat)...)

    for e in 1:no_of_edges
        i, j = edge_list[e, :]

        α_next[i, j], ytil_next[i, j], sum_α, sum_ytil =
            cal_message_ij(i, e, j, α, ytil, d, r, Λ, adj_n, deg)

        α_next[j, i], ytil_next[j, i], sum_α, sum_ytil =
            cal_message_ij(j, e, i, α, ytil, d, r, Λ, adj_n, deg)
    end

    return α_next, ytil_next
end


"""Optimal flow of the full energy, from j to i."""
function cal_yopt_ij_forward(i::Int, e::Int, j::Int, α::SparseMatrixCSC{Float64, Int},
            ytil::SparseMatrixCSC{Float64, Int}, r::Array{Float64, 1})::Float64
    yopt_ij = ( α[j, i]*ytil[j, i] - α[i, j]*ytil[i, j] ) / (α[i, j] + α[j, i] - r[e])
    return yopt_ij
end


"""Optimal flow of all edges."""
function cal_yopt_all_edges(α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            r::Array{Float64, 1}, edge_list::DataFrame)
    no_of_edges = size(edge_list, 1)
    yopt = zeros(Float64, no_of_edges)

    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        ## y_e = y_ij denotes the flow from j to i:
        yopt[e] = cal_yopt_ij_forward(i, e, j, α, ytil, r)
    end

    return yopt
end


"""Gradient of yopt[i, j] w.r.t. the messages α[i, j], ytil[i, j], α[j, i], ytil[j, i],
and control parameter r[e] on target edge e."""
function cal_gradient_DyDm_on_target(i::Int, e::Int, j::Int, α::SparseMatrixCSC{Float64, Int},
            ytil::SparseMatrixCSC{Float64, Int}, d::Int, r::Array{Float64, 1}, deg::Array{Int})
    if deg[i] == 1
        ## Not meaningful to place a leaf as a target, since the flow cannot be tuned.
        return zeros(5)

    elseif i == d
        ## Only Dytil_ji = 1:
        return 0., 0., 0., 1., 0.

    elseif j == d
        ## Only Dytil_ij = -1:
        return 0., -1., 0., 0., 0.

    else
        temp1 = α[j, i] * ytil[j, i] - α[i, j] * ytil[i, j]     ## numerator of yopt_ij
        temp2 = α[i, j] + α[j, i] - r[e]                        ## denominator of yopt_ij
        yopt  = temp1 / temp2

        ∂y_∂αij  = ( - ytil[i, j] - yopt) / temp2
        ∂y_∂ytij = - α[i, j] / temp2
        ∂y_∂αji  = ( ytil[j, i] - yopt) / temp2
        ∂y_∂ytji = α[j, i] / temp2
        ∂y_∂re   = yopt / temp2

        return ∂y_∂αij, ∂y_∂ytij, ∂y_∂αji, ∂y_∂ytji, ∂y_∂re
    end
end


"""Gradient of messages α[i, j], α[j, i], ytil[i, j], ytil[j, i] w.r.t. R[e]."""
function cal_gradient_DmDr_on_target(i::Int, e::Int, j::Int,
            α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, adj_n::Array{Int, 2}, deg::Array{Int})
    if deg[i] == 1
        ## Not meaningful to place a leaf as a target, since the flow cannot be tuned.
        return zeros(4)

    else
        ∂αij_∂r = 1.
        ∂αji_∂r = 1.

        if i == d
            ## As ytil[i, j] = 0:
            ∂ytij_∂r = 0.
        else
            sum_αij = 0.
            for m in 1:deg[i]
                k = adj_n[i, m]
                if k == j
                    continue
                end
                sum_αij += 1 / α[k, i]
            end
            ∂ytij_∂r = ytil[i, j] * (-sum_αij) / (1 + r[e]*sum_αij)
        end

        if j == d
            ## As ytil[j, i] = 0:
            ∂ytji_∂r = 0.
        else
            sum_αji = 0.
            for m in 1:deg[j]
                ℓ = adj_n[j, m]
                if ℓ == i
                    continue
                end
                sum_αji += 1 / α[ℓ, j]
            end
            ∂ytji_∂r = ytil[j, i] * (-sum_αji) / (1 + r[e]*sum_αji)
        end

        return ∂αij_∂r, ∂ytij_∂r, ∂αji_∂r, ∂ytji_∂r
    end
end


"""Compute the gradient of messages on edge j->k (edge ep) w.r.t.
the messages and control varible on the cavity tree i->j (edge e), i.e., ∂m[j->k]/∂m[i->j].
Assume edges e and ep do not attach to dangling node."""
function cal_gradient_message_jk_wrt_ij(i::Int, j::Int, ep::Int, k::Int,
            α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, deg::Array{Int})
    if j == d
        return 0., 0., 0.
    end

    sum_α    = 0.
    sum_ytil = 0.
    for m in 1:deg[j]
        ℓ = adj_n[j, m]
        if ℓ == k
            continue
        end
        sum_α     +=  1 / α[ℓ, j]
        sum_ytil  +=  ytil[ℓ, j]
    end

    temp = ( 1 + r[ep]*sum_α )
    ∂αjk_∂αij = α[i, j]^(-2) / sum_α^2
    ∂yjk_∂αij = ( Λ[j] + sum_ytil ) / temp^2 * r[ep] * α[i, j]^(-2)

    if i == d
        ∂yjk_∂yij = 0.
    else
        ∂yjk_∂yij = 1 / temp
    end

    return ∂αjk_∂αij, ∂yjk_∂αij, ∂yjk_∂yij
end


"""Compute the gradient of objective of target edge 'etarg'
w.r.t the messages and control varible on the cavity tree i->j (edge e)."""
function cal_gradient_objective_etarg_wrt_ij(i::Int, e::Int, j::Int, etarg::Int, α::SparseMatrixCSC{Float64, Int},
            ytil::SparseMatrixCSC{Float64, Int}, sum_αij::Float64,
            gα::Array{SparseMatrixCSC{Float64, Int}, 1}, gy::Array{SparseMatrixCSC{Float64, Int}, 1},
            gr::Array{SparseMatrixCSC{Float64, Int}, 1},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2},
            adj_e::Array{Int, 2}, deg::Array{Int})
    if deg[i] == 1 || j == d
        return 0., 0., 0.
    end

    ∂H_∂αij = 0.
    ∂H_∂yij = 0.
    ∂H_∂rij  = 0.
    for m in 1:deg[j]
        k  = adj_n[j, m]
        ep = adj_e[j, m]
        if k == i
            continue
        end

        ∂αjk_∂αij, ∂yjk_∂αij, ∂yjk_∂yij =
            cal_gradient_message_jk_wrt_ij(i, j, ep, k, α, ytil, d, r, Λ, adj_n, deg)

        ∂αjk_∂re = ∂αjk_∂αij
        ∂yjk_∂re = ∂yjk_∂αij + ∂yjk_∂yij * ytil[i, j] * (-sum_αij) / ( 1 + r[e]*sum_αij )

        ∂H_∂αij += gα[etarg][j, k] * ∂αjk_∂αij + gy[etarg][j, k] * ∂yjk_∂αij
        ∂H_∂yij += gy[etarg][j, k] * ∂yjk_∂yij
        ∂H_∂rij += gα[etarg][j, k] * ∂αjk_∂re  + gy[etarg][j, k] * ∂yjk_∂re
    end

    return ∂H_∂αij, ∂H_∂yij, ∂H_∂rij
end


"""Calculate ∂H[etarg]/∂r[e] based on gr."""
function cal_gradient_DHDr_by_gr(gr::Array{SparseMatrixCSC{Float64, Int}, 1},
            targets::Array{Int, 1}, edge_list::DataFrame)
    no_of_target = size(targets, 1)
    no_of_edges  = size(edge_list, 1)
    ∂H_∂r = zeros(Float64, no_of_target, no_of_edges)

    for etarg in 1:no_of_target
        for e in 1:no_of_edges
            i, j = edge_list[e, :]
            ∂H_∂r[etarg, e] += gr[etarg][i, j]

            if targets[etarg] != e
                ## Do not double count on targets:
                ∂H_∂r[etarg, e] += gr[etarg][j, i]
            end
        end
    end

    return ∂H_∂r
end


"""Lower level message passing."""
function lower_MP!(α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame,
            state::Array{Float64, 2}, max_iter::Int, check_intv::Int, seed::Int=-1, stop_when_converge::Bool=true)
    if seed > 0
        Random.seed!(seed)
    end
    no_of_nodes::Int, no_of_edges::Int = length(Λ), length(r)

    α_prev::SparseMatrixCSC{Float64, Int} = copy(α)
    ytil_prev::SparseMatrixCSC{Float64, Int} = copy(ytil)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1:max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        j = adj_n[i, n]
        e = adj_e[i, n]

        ## update the messages:
        α[i, j], ytil[i, j], sum_α, sum_ytil =
            cal_message_ij(i, e, j, α, ytil, d, r, Λ, adj_n, deg)

        if stop_when_converge && t % check_intv == 0
            if check_convergence(α, α_prev, adj_mat) && check_convergence(ytil, ytil_prev, adj_mat)
                converg = true
                tc = t
                println("Lower MP converges at t=$t.")
                break
            else
                α_prev = copy(α)
                ytil_prev = copy(ytil)
            end
        end
    end

    return converg, tc
end


"""Bilevel message passing."""
function bilevel_MP!(α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            gα::Array{SparseMatrixCSC{Float64, Int}, 1}, gy::Array{SparseMatrixCSC{Float64, Int}, 1},
            gr::Array{SparseMatrixCSC{Float64, Int}, 1},
            y0::Array{Float64, 1}, targets::Array{Int, 1}, threshold::Float64, rmin::Float64, rmax::Float64,
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame,
            state::Array{Float64, 2}, max_iter::Int, check_intv::Int, seed::Int=-1,
            update_r::Bool=true, s::Float64=1., stop_when_converge::Bool=true)
    if seed > 0
        Random.seed!(seed)
    end
    no_of_nodes::Int, no_of_edges::Int, no_of_target = length(Λ), length(r), length(targets)

    α_prev::SparseMatrixCSC{Float64, Int} = copy(α)
    ytil_prev::SparseMatrixCSC{Float64, Int} = copy(ytil)
    gr_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(gr)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1:max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        j = adj_n[i, n]
        e = adj_e[i, n]

        ## update the messages:
        α[i, j], ytil[i, j], sum_α, sum_ytil =
            cal_message_ij(i, e, j, α, ytil, d, r, Λ, adj_n, deg)

        ## update the gradients of each H_etarg on edge ij:
        for etarg in 1:length(targets)
            if targets[etarg] == e
                ## boundary condition of gradients:
                yopt_ij = cal_yopt_ij_forward(i, e, j, α, ytil, r)
                ∂H_∂y = cal_gradient_DHDy_on_target(e, yopt_ij, y0[e], targets, threshold)

                ∂y_∂αij, ∂y_∂ytij, ∂y_∂αji, ∂y_∂ytji, ∂y_∂re = cal_gradient_DyDm_on_target(i, e, j, α, ytil, d, r, deg)

                ∂αij_∂r, ∂ytij_∂r, ∂αji_∂r, ∂ytji_∂r = cal_gradient_DmDr_on_target(i, e, j, α, ytil, d, r, adj_n, deg)

                Dy_Dre = (∂y_∂αij * ∂αij_∂r) + (∂y_∂ytij * ∂ytij_∂r) + (∂y_∂αji * ∂αji_∂r) + (∂y_∂ytji * ∂ytji_∂r) + ∂y_∂re

                gα[etarg][i, j] = ∂H_∂y * ∂y_∂αij
                gy[etarg][i, j] = ∂H_∂y * ∂y_∂ytij
                gα[etarg][j, i] = ∂H_∂y * ∂y_∂αji
                gy[etarg][j, i] = ∂H_∂y * ∂y_∂ytji
                gr[etarg][i, j] = ∂H_∂y * Dy_Dre
                # println((i, j, ∂H_∂y, yopt_ij, ytil[i, j], ytil[j, i], Dy_Dre, gr[etarg][i, j]))

            else
                ## normal update:
                ∂H_∂αij, ∂H_∂yij, ∂H_∂rij = cal_gradient_objective_etarg_wrt_ij(i, e, j, etarg,
                                               α, ytil, sum_α, gα, gy, gr, d, r, Λ, adj_n, adj_e, deg)
                gα[etarg][i, j] = ∂H_∂αij
                gy[etarg][i, j] = ∂H_∂yij
                gr[etarg][i, j] = ∂H_∂rij
            end
        end

        ## gradient descent:
        if update_r && t % (check_intv ÷ 10) == 0 # && t > check_intv * 100
            grad = 0.
            for etarg in 1:length(targets)
                grad += gr[etarg][i, j]
                if targets[etarg] != e
                    ## Do not double count on targets:
                    grad += gr[etarg][j, i]
                end
            end

            r[e] -= s*grad
            r[e] = max(r[e], rmin)
            r[e] = min(r[e], rmax)
        end

        # if t % check_intv == 0 && t > max_iter ÷ 10
        if stop_when_converge && t % check_intv == 0 && t > check_intv * 100   ## for a small max_iter
            if check_convergence(α, α_prev, adj_mat) && check_convergence(ytil, ytil_prev, adj_mat) &&
               check_convergence(gr, gr_prev, adj_mat) && check_gradient_norm(gr, adj_mat)
                converg = true
                tc = t
                println("Bilevel MP converges at t=$t.")
                break
            else
                α_prev = copy(α)
                ytil_prev = copy(ytil)
                gr_prev = deepcopy(gr)
            end
        end
    end

    return converg, tc
end


"""Compare two set of messages which are non-zero iff adj_mat[i, j]≠0."""
function check_convergence(msg::SparseMatrixCSC{Float64, Int}, msg_prev::SparseMatrixCSC{Float64, Int},
            adj_mat::SparseMatrixCSC{Int, Int}, eps::Float64=1e-8)::Bool
    max_diff = maximum( abs.((msg - msg_prev) .* adj_mat) )
    converg = max_diff < eps
    return converg
end


"""Compare two set of messages which are non-zero iff adj_mat[i,e]≠0."""
function check_convergence(msg::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            msg_prev::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            adj_mat::SparseMatrixCSC{Int, Int}, eps::Float64=1e-8)::Bool
    no_of_target::Int = size(msg, 1)
    max_diff::Float64 = 0.
    converg::Bool = true
    for etarg in 1:no_of_target
        max_diff = max( maximum( abs.((msg[etarg] - msg_prev[etarg]) .* adj_mat) ), max_diff )
        if max_diff > eps
            converg = false
            break
        end
    end
    return converg
end


"""Check whether the norm of a gradient is small enough."""
function check_gradient_norm(g::Array{SparseMatrixCSC{Float64, Int}, 1},
            adj_mat::SparseMatrixCSC{Int, Int}, eps::Float64=1e-6)
    no_of_target = size(g, 1)
    max_diff::Float64 = 0.
    converg::Bool = true
    for etarg in 1:no_of_target
        max_diff = max( maximum( abs.(g[etarg] .* adj_mat) ), max_diff )
        if max_diff > eps
            converg = false
            break
        end
    end
    return converg
end


"""Check self-consistency of messages."""
function check_message_consistency(α::SparseMatrixCSC{Float64, Int}, ytil::SparseMatrixCSC{Float64, Int},
            d::Int, r::Array{Float64, 1}, Λ::Array{Float64, 1}, adj_n::Array{Int, 2}, adj_e::Array{Int, 2},
            adj_mat::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame, eps::Float64=1e-8)
    α_next, ytil_next = lower_MP_one_parallel_pass(α, ytil, d, r, Λ, adj_n, adj_e, adj_mat, deg, edge_list)
    max_diff_α = maximum( abs.((α - α_next) .* adj_mat) )
    max_diff_ytil = maximum( abs.((ytil - ytil_next) .* adj_mat) )
    max_diff = max(max_diff_α, max_diff_ytil)
    consistency = max_diff < eps
    return consistency, max_diff
end


end  # module FlowMP

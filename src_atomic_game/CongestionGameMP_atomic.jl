module CongestionGameMP
"""
Atomic game is considered.
Multiple destinations.

Messages:
xs[a][i, e]: starting position of the flow values on edge i->e to be evaluted,
E[a, n ∈ 1:K][i, e]: energy value on edge i->e, corresponding to xs[a][i, e]+n-1,
where K is an odd number.

Keep using ϕpp and ϕp, which is exact for quadratic cost (both single and multi class).
"""

include("../GraphUtil.jl")
include("../CongestionGameProb.jl")

using DataFrames
using Random
using SparseArrays
using LinearAlgebra
using JuMP
using Ipopt
using .GraphUtil
using .CongestionGameProb

ϕ_func = ϕ_q
ϕp_func = ϕp_q
ϕpp_func = ϕpp_q
σ_func = xℓ_q
σp_func = xℓp_q
σpp_func = xℓpp_q
Φ_and_H = Φ_and_H_q
xℓ_func = xℓ_q


"""Compute the continuous optimal flow.
Now model and xv are local variables,
so one should avoid using the @eval macro, which only deals with global variable."""
function cal_continuous_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ_in; level="lower")
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


"""Initialize the working points xw and related messages by JuMP solutions."""
function init_xw_by_opt_flow!(xw, xwtot, xs, ϕpp, ϕp, edge_list, K,
            B, s, q, d, cpc, tf, Λ, τ, level="lower"; continuous=true)
    no_of_nodes, no_of_edges, no_of_class = size(Λ, 1), size(tf, 1), size(d, 1)

    ## Continuous flow relaxation, Nash equilibrium:
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    @variable(model, xv[1:no_of_class*no_of_edges] >= 0.)
    xn_values, runtime1 =
        CongestionGameMP.cal_continuous_opt_flow!(model, xv, B, s, q, d, cpc, tf, Λ, τ, level=level)

    for e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        xwtot[i, e] = 0
        xwtot[j, e] = 0
        for a in 1:no_of_class
            xw[a][i, e] = Int(round(xn_values[a, e]))
            xw[a][j, e] = Int(round(xn_values[a, e]))
            xwtot[i, e] += xw[a][i, e]
            xwtot[j, e] += xw[a][j, e]
            xs[a][i, e] = CongestionGameMP.cal_xs_start_pos(xw[a][i, e], K)
            xs[a][j, e] = CongestionGameMP.cal_xs_start_pos(xw[a][j, e], K)
        end
        ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕp[j, e] = ϕp_func(xwtot[j, e], cpc[e], tf[e], s, q)
        ϕpp[j, e] = ϕpp_func(xwtot[j, e], cpc[e], tf[e], s, q)
    end

    return 0
end


"""Integer n (1 <= n <= L^d) to coordinates [s1, s2, ..., sd] of a cubic lattice of d dimension,
where each dimension has length L.
Let the coordinate si counts from 1."""
function int_to_coord(n::Int, d::Int, L::Int)::Array{Int, 1}
    coord_str = string(n-1, base=L, pad=d)
    coord = [parse(Int, coord_str[i])+1 for i in 1:d]
    return coord
end


function init_lower_messages(B::SparseMatrixCSC{Int, Int}, cpc::Array{Float64, 1}, tf::Array{Float64, 1},
            s::Float64, q::Float64, no_of_class::Int, K::Int=5, seed::Int=200)
    Random.seed!(seed)
    xs::Array{SparseMatrixCSC{Int, Int}, 1} = [rand(0:K-1, size(B)...) .* abs.(B) for a in 1:no_of_class]
    E::Array{SparseMatrixCSC{Float64, Int}, 2} = [rand(size(B)...) .* abs.(B) for a in 1:no_of_class, m in 1:K]
    xw::Array{SparseMatrixCSC{Int, Int}, 1} = [rand(0:K-1, size(B)...) .* abs.(B) for a in 1:no_of_class]
    xwtot::SparseMatrixCSC{Int, Int} = spzeros(Int, size(B)...)
    ϕpp::SparseMatrixCSC{Float64, Int} = spzeros(size(B)...)
    ϕp::SparseMatrixCSC{Float64, Int} = spzeros(size(B)...)
    for i in 1:size(B, 1), e in 1:size(B, 2)
        for a in 1:no_of_class
            xwtot[i, e] += xw[a][i, e]
        end
        ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
    end
    return xs, E, xw, xwtot, ϕpp, ϕp
end


function init_upper_messages(B::SparseMatrixCSC{Int, Int}, xwtot::SparseMatrixCSC{Int, Int},
            cpc::Array{Float64, 1}, tf::Array{Float64, 1}, s::Float64, q::Float64,
            no_of_class::Int, K::Int, seed::Int=300)
    Random.seed!(seed)
    H::Array{SparseMatrixCSC{Float64, Int}, 2} = [rand(size(B)...) .* abs.(B) for a in 1:no_of_class, m in 1:K]
    σpp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    σp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    for i in 1:size(B, 1), e in 1:size(B, 2)
        σpp[i, e] = σpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        σp[i, e] = σp_func(xwtot[i, e], cpc[e], tf[e], s, q)
    end
    return H, σpp, σp
end


"""Starting position of x_ie_a to store the cavity energy value E_ie()."""
function cal_xs_start_pos(xw_ie_a::Int, K::Int)::Int
    dK = (K-1)÷2
    x_ie_a_start::Int = max(xw_ie_a - dK, 0)
    return x_ie_a_start
end


"""Compute the value of the flow conservation constraint R^a_ie of the cavity working points xw^a_ie."""
function cal_R_ie_a(x_ie_a::Int, i::Int, e::Int, a::Int, xw::Array{SparseMatrixCSC{Int, Int}, 1},
            Λ::Array{Int, 2}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)::Int64
    R_ie_a::Int = Λ[i, a]
    for m in 1:deg[i]
        epp = adj_e[i, m]
        k = end_node(edge_list, epp, i)
        if epp == e
            R_ie_a += B[i, e] * x_ie_a
        else
            R_ie_a += B[i, epp] * xw[a][k, epp]
        end
    end
    return R_ie_a
end


"""Net resource at node i, class a. Alternative: use x[a][k, e] (k->e->i)"""
function cal_net_R_ia(i::Int, a::Int, xw::Array{SparseMatrixCSC{Int, Int}, 1}, Λ::Array{Int, 2},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)::Int64
    R_ia::Int = Λ[i, a]
    for m in 1:deg[i]
        epp = adj_e[i, m]
        k = end_node(edge_list, epp, i)
        R_ia += B[i, epp] * xw[a][k, epp]
    end
    return R_ia
end


"""Interaction strength ϕ at a particular flow value, using perturbation.
Toll is included."""
function cal_ϕ_with_τ_at_x_ie_a(x_ie_a::Int, i::Int64, e::Int64, a::Int, xw::Array{SparseMatrixCSC{Int, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, τ::Array{Float64, 1})
    ε_ie_a = x_ie_a - xw[a][i, e]
    ϕ_ie_a = 0.5 * ϕpp[i, e] * ε_ie_a^2 + (ϕp[i, e] + τ[e]) * ε_ie_a
    return ϕ_ie_a
end


"""Return the marginal value of xw_ea, minimal full energy Ee_min, and whether
{xs[a][i, e], ...} and {xs[a][j, e], ...} have a common range.
Awared of two-level degeneracy."""
function cal_marginal_xw_ie_a(i::Int64, e::Int64, j::Int64, a::Int,
            xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1}, ϕpp::SparseMatrixCSC{Float64, Int},
            ϕp::SparseMatrixCSC{Float64, Int}, d::Array{Int, 1}, τ::Array{Float64, 1}, K::Int,
            Λ::Array{Int, 2}, deg::Array{Int}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            edge_list::DataFrame)
    dx1 = xs[a][i, e] - xs[a][j, e]
    if abs(dx1) >= K
        # println("At e i j = $e, $i, $j, no flow overlap between i->e and j->e, with xs_a_ie, xs_a_je = $(xs[a][i, e]), $(xs[a][j, e])")
        return xw[a][i, e], 0., false
    end

    eps = 1e-3
    xw_sub = -1
    Ee_sub = Inf

    if i == d[a]
        x_je_a = xs[a][j, e]
        xw_ea  = x_je_a
        Ee_min = E[a, 1][j, e] + cal_ϕ_with_τ_at_x_ie_a(x_je_a, j, e, a, xw, ϕpp, ϕp, τ)
        for m in 2:K
            x_je_a = xs[a][j, e] + m-1
            Ee_mth = E[a, m][j, e] + cal_ϕ_with_τ_at_x_ie_a(x_je_a, j, e, a, xw, ϕpp, ϕp, τ)
            if Ee_mth < Ee_min - eps
                xw_ea  = x_je_a
                Ee_min = Ee_mth
            elseif abs(Ee_mth - Ee_min) <= eps
                xw_sub = x_je_a
                Ee_sub = Ee_mth
            end
        end

    elseif j == d[a]
        x_ie_a = xs[a][i, e]
        xw_ea  = x_ie_a
        Ee_min = E[a, 1][i, e] + cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ)
        for m in 2:K
            x_ie_a = xs[a][i, e] + m-1
            Ee_mth = E[a, m][i, e] + cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ)
            if Ee_mth < Ee_min - eps
                xw_ea  = x_ie_a
                Ee_min = Ee_mth
            elseif abs(Ee_mth - Ee_min) <= eps
                xw_sub = x_ie_a
                Ee_sub = Ee_mth
            end
        end

    elseif 0 <= dx1 < K
        ## Example: K = 5, dx1 = 2
        ## i->e:     o o o o o
        ## j->e: o o o o o
        x_ie_a = xs[a][i, e]
        xw_ea  = x_ie_a
        Ee_min = E[a, 1][i, e] + E[a, 1 + dx1][j, e] +
                   0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ) +
                   0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, j, e, a, xw, ϕpp, ϕp, τ)
        for m in 2:K-dx1
            x_ie_a = xs[a][i, e] + m-1
            Ee_mth = E[a, m][i, e] + E[a, m + dx1][j, e] +
                       0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ) +
                       0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, j, e, a, xw, ϕpp, ϕp, τ)
            if Ee_mth < Ee_min - eps
                xw_ea  = x_ie_a
                Ee_min = Ee_mth
            elseif abs(Ee_mth - Ee_min) <= eps
                xw_sub = x_ie_a
                Ee_sub = Ee_mth
            end
        end

    elseif -K < dx1 < 0
        ## Example: K = 5, dx1 = -2
        ## i->e: o o o o o
        ## j->e:     o o o o o
        x_ie_a = xs[a][j, e]
        xw_ea  = x_ie_a
        Ee_min = E[a, 1][j, e] + E[a, 1 - dx1][i, e] +
                   0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ) +
                   0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, j, e, a, xw, ϕpp, ϕp, τ)
        for m in 2:K+dx1
            x_ie_a = xs[a][j, e] + m-1
            Ee_mth = E[a, m][j, e] + E[a, m - dx1][i, e] +
                       0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, i, e, a, xw, ϕpp, ϕp, τ) +
                       0.5 * cal_ϕ_with_τ_at_x_ie_a(x_ie_a, j, e, a, xw, ϕpp, ϕp, τ)
           if Ee_mth < Ee_min - eps
               xw_ea  = x_ie_a
               Ee_min = Ee_mth
           elseif abs(Ee_mth - Ee_min) <= eps
               xw_sub = x_ie_a
               Ee_sub = Ee_mth
           end
        end
    end

    if abs(Ee_sub - Ee_min) <= eps
        if i == d[a]
            R_min = cal_R_ie_a(xw_ea, j, e, a, xw, Λ, adj_e, B, deg, edge_list)
            R_sub = R_min - B[j, e]*xw_ea + B[j, e]*xw_sub
        else
            R_min = cal_R_ie_a(xw_ea, i, e, a, xw, Λ, adj_e, B, deg, edge_list)
            R_sub = R_min - B[i, e]*xw_ea + B[i, e]*xw_sub
        end

        if abs(R_sub) < abs(R_min)
            xw_ea  = xw_sub
            Ee_min = Ee_sub
        end
    end

    return xw_ea, Ee_min, true
end


"""Given the messages and working points (xw), compute the new working points (xw_new)."""
function cal_all_xw(xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1}, ϕpp::SparseMatrixCSC{Float64, Int},
            ϕp::SparseMatrixCSC{Float64, Int}, d::Array{Int, 1}, Λ::Array{Int, 2},
            τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int},
            edge_list::DataFrame, K::Int)
    xw_new::Array{SparseMatrixCSC{Int, Int}, 1} = [spzeros(Int, size(B)...) for a in 1:size(d, 1)]
    xwtot_new::SparseMatrixCSC{Int, Int} = spzeros(Int, size(B)...)
    for a in 1:size(Λ, 2), e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        xw_ie_a, Ee_a_min, overlap =
            cal_marginal_xw_ie_a(i, e, j, a, xs, E, xw, ϕpp, ϕp, d, τ, K, Λ, deg, adj_e, B, edge_list)
        xw_je_a, Ee_a_min, overlap =
            cal_marginal_xw_ie_a(j, e, i, a, xs, E, xw, ϕpp, ϕp, d, τ, K, Λ, deg, adj_e, B, edge_list)
        xw_new[a][i, e] = xw_ie_a
        xw_new[a][j, e] = xw_je_a
        xwtot_new[i, e] += xw_ie_a
        xwtot_new[j, e] += xw_je_a
    end
    return xw_new, xwtot_new
end


"""With respect to the current value of the working point xw[a][i, e] (OR xs[a][i, e]),
for lower MP, update messages E[a, :][i,e],
for upper MP, update messages H[a, :][i,e].
Project to flows {xs[a][i,e], ...}."""
function cal_messages_ie_a(t::Int, i::Int64, e::Int64, j::Int64, a::Int,
            xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, Λ::Array{Int, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2},
            B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame, K::Int)
    x_ie_a_start = xs[a][i, e]
    x_ie_a_end   = x_ie_a_start + K - 1

    if i == d[a]
        for n in 1:K
            E[a, n][i, e] = 0
        end
        visited = ones(Int8, K)
        return visited
    end

    visited = zeros(Int8, K)
    E_min = Inf

    for ξ in 1:K^(deg[i]-1)
        coord = int_to_coord(ξ, deg[i]-1, K)  ## array of length deg[i]-1

        ## Calculate the flow x_ie_a based on x_kep_a (specified by the ξ configuration):
        c = 1   ## counter of the set {ep ∈ ∂i\e}
        R_ie_a = Λ[i, a]
        for m in 1:deg[i]
            ep = adj_e[i, m]
            k = end_node(edge_list, ep, i)
            if ep == e
                continue
            end
            ## x^a_{k->ep} evaluate at the value as xs[a][k, ep] + coord[c]-1
            x_kep_a_ξ = xs[a][k, ep] + coord[c]-1
            R_ie_a += B[i, ep] * x_kep_a_ξ
            c += 1
        end
        x_ie_a_ξ = -R_ie_a ÷ B[i, e]

        ## Neglect the case ξ if x_ie_a is out of range:
        if x_ie_a_ξ < x_ie_a_start || x_ie_a_ξ > x_ie_a_end
            continue
        end

        ## Calculate the energy E^a_{i->e} corresponding to ξ configuration:
        c = 1
        E_ie_a_ξ = 0.
        for m in 1:deg[i]
            ep = adj_e[i, m]
            k = end_node(edge_list, ep, i)
            if ep == e
                continue
            end
            x_kep_a_ξ = xs[a][k, ep] + coord[c]-1
            E_ie_a_ξ += E[a, coord[c]][k, ep] +
                        cal_ϕ_with_τ_at_x_ie_a(x_kep_a_ξ, k, ep, a, xw, ϕpp, ϕp, τ)
            c += 1
        end

        n = x_ie_a_ξ - x_ie_a_start + 1    ## the n-th energy value to be stored
        if visited[n] == 1
            E[a, n][i, e] = min(E_ie_a_ξ, E[a, n][i, e])
        else
            E[a, n][i, e] = E_ie_a_ξ
            visited[n] = 1
        end

        E_min = min(E_ie_a_ξ, E_min)
    end

    for n in 1:K
        if visited[n] == 1
            E[a, n][i, e] -= E_min
        end
    end

    return visited
end


"""Lower level message passing, random sequential update."""
function lower_MP_rand_seq!(xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1}, xwtot::SparseMatrixCSC{Int, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, d::Array{Int, 1}, cpc::Array{Float64, 1},
            tf::Array{Float64, 1}, s::Float64, q::Float64, Λ::Array{Int, 2}, τ::Array{Float64, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, outdeg::Array{Int}, edge_list::DataFrame, K::Int,
            state::Array{Float64, 2}, max_iter::Int, check_intv::Int, lr::Float64, seed::Int, level="lower")
    if seed > 0
        Random.seed!(seed)
    end

    no_of_nodes::Int, no_of_edges::Int, no_of_class::Int = size(Λ, 1), size(tf, 1), size(d, 1)

    xs_prev = deepcopy(xs)
    E_prev  = deepcopy(E)
    xw_prev = deepcopy(xw)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1:max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        e = adj_e[i, n]
        j = end_node(edge_list, e, i)
        a = rand(1:no_of_class)

        ## update the working point by forward provision:
        xw_e_a_new, Ee_a_min, overlap = cal_marginal_xw_ie_a(i, e, j, a,
                                          xs, E, xw, ϕpp, ϕp, d, τ, K, Λ, deg, adj_e, B, edge_list)
        xw_ie_a_old = xw[a][i, e]
        if !overlap
            println("At t $t, i $i, e $e, j $j, no flow overlap between i->e and j->e, with xs_a_ie, xs_a_je = $(xs[a][i, e]), $(xs[a][j, e])")
            xw[a][i, e] = xw_ie_a_old + sign(xw[a][j, e] - xw_ie_a_old)
        else
            xw[a][i, e] = xw_ie_a_old + sign(xw_e_a_new - xw_ie_a_old)  ## incremental update
            # xw[a][i, e] = Int( round( lr * xw_e_a_new + (1 - lr) * xw_ie_a_old ) )  ## with a learning rate
        end

        ## update the latency-related matrices:
        if xw[a][i, e] != xw_ie_a_old
            xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]
            if level == "lower"
                ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
                ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
            else
                ϕp[i, e] = σp_func(xwtot[i, e], cpc[e], tf[e], s, q)
                ϕpp[i, e] = σpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
            end
        end

        ## update the messages:
        xs[a][i, e] = cal_xs_start_pos(xw[a][i, e], K)
        visited = cal_messages_ie_a(t, i, e, j, a, xs, E, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list, K)

        if t % check_intv == 0
            record_state = false
            if record_state
                state[t ÷ check_intv, 1] = sum( abs.( (xw[1] - xw_prev[1]) .* B) )
                state[t ÷ check_intv, 2] = sum( abs.( (E[1, 1] - E_prev[1, 1]) .* B) )
            end

            if check_convergence(xw, xw_prev, B) && check_flow_consistency(xw, d, edge_list) &&
               check_convergence(E[:, 1], E_prev[:, 1], B)
                converg = true
                tc = t
                print("Lower MP converge at t=$t.")
                break
            else
                E_prev  = deepcopy(E)
                xw_prev = deepcopy(xw)
            end
        end
    end

    return converg, tc
end


"""Search of optimal toll, by bringing x^*N_ie as close as possible to x^*G_ie.
Valid for single destination.
Move the flow by one step in each iteration."""
function cal_optimal_toll_single_d(t::Int, i::Int64, e::Int64, j::Int64, a::Int,
            xw_ie_a_current::Int, xw_ie_a_target::Int, τmin::Array{Float64, 1}, τmax::Array{Float64, 1},
            xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            τ::Array{Float64, 1}, K::Int, step_size::Float64)
    if xw_ie_a_target - xw_ie_a_current == 0
        return τ[e]
    end

    xw_ie_a_target = xw_ie_a_current + sign(xw_ie_a_target - xw_ie_a_current)

    if (xw_ie_a_current < xs[a][j, e] || xw_ie_a_current > xs[a][j, e] + K - 1) ||
       (xw_ie_a_target < xs[a][i, e]  || xw_ie_a_target > xs[a][i, e] + K - 1)  ||
       (xw_ie_a_target < xs[a][j, e]  || xw_ie_a_target > xs[a][j, e] + K - 1)
        return τ[e]
    end

    n_ie = xw_ie_a_current - xs[a][i, e] + 1
    n_je = xw_ie_a_current - xs[a][j, e] + 1
    E_ea_current = E[a, n_ie][i, e] + E[a, n_je][j, e] +
                    0.5 * cal_ϕ_with_τ_at_x_ie_a(xw_ie_a_current, i, e, a, xw, ϕpp, ϕp, τ) +
                    0.5 * cal_ϕ_with_τ_at_x_ie_a(xw_ie_a_current, j, e, a, xw, ϕpp, ϕp, τ)

    n_ie_tg = xw_ie_a_target - xs[a][i, e] + 1
    n_je_tg = xw_ie_a_target - xs[a][j, e] + 1
    E_ea_target = E[a, n_ie_tg][i, e] + E[a, n_je_tg][j, e] +
                    0.5 * cal_ϕ_with_τ_at_x_ie_a(xw_ie_a_target, i, e, a, xw, ϕpp, ϕp, τ) +
                    0.5 * cal_ϕ_with_τ_at_x_ie_a(xw_ie_a_target, j, e, a, xw, ϕpp, ϕp, τ)

    if abs(E_ea_target - E_ea_current) > 1e-2
        Δτ_e = - (E_ea_target - E_ea_current) / (xw_ie_a_target - xw_ie_a_current)
        Δτ_e *= (1 + 1e-2)  ## a heuristic
        τ_e = τ[e] + Δτ_e
        τ_e = min(max(τ_e, τmin[e]), τmax[e])
        return τ_e
    else
        ## If E_ea_target and E_ea_current are close, move τ[e] according to a typical energy scale:
        Δτ_e = sign(xw_ie_a_current - xw_ie_a_target) * step_size  ## here the energy scale is fixed as step_size
        τ_e = τ[e] + Δτ_e
        τ_e = min(max(τ_e, τmin[e]), τmax[e])
        return τ_e
    end
end


"""Bi-level message passing."""
function bilevel_MP_rand_seq!(xs::Array{SparseMatrixCSC{Int, Int}, 1}, E::Array{SparseMatrixCSC{Float64, Int}, 2},
            xw::Array{SparseMatrixCSC{Int, Int}, 1}, xwtot::SparseMatrixCSC{Int, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            H::Array{SparseMatrixCSC{Float64, Int}, 2},
            σpp::SparseMatrixCSC{Float64, Int}, σp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, cpc::Array{Float64, 1}, tf::Array{Float64, 1}, s::Float64, q::Float64, Λ::Array{Int, 2},
            τ::Array{Float64, 1}, τmin::Array{Float64, 1}, τmax::Array{Float64, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, outdeg::Array{Int},
            edge_list::DataFrame, K::Int, state::Array{Float64, 2},
            max_iter::Int, check_intv::Int, lr::Float64, seed::Int=300,
            update_toll::Bool=true, step_size::Float64=0.1; save_toll::Bool=false)
    if seed > 0
        Random.seed!(seed)
    end
    no_of_nodes::Int, no_of_edges::Int, no_of_class::Int = size(Λ, 1), size(tf, 1), size(d, 1)
    τ0 = zeros(Float64, no_of_edges)

    xs_prev = deepcopy(xs)
    E_prev  = deepcopy(E)
    xw_prev = deepcopy(xw)
    H_prev  = deepcopy(H)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1: max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        e = adj_e[i, n]
        j = end_node(edge_list, e, i)
        a = rand(1:no_of_class)

        ## update the working point by forward provision, according to LOWER-LEVEL messages:
        xw_e_a_new, Ee_a_min, overlap = cal_marginal_xw_ie_a(i, e, j, a,
                                          xs, E, xw, ϕpp, ϕp, d, τ, K, Λ, deg, adj_e, B, edge_list)
        xw_ie_a_old = xw[a][i, e]
        if !overlap
            println("At t $t, i $i, e $e, j $j, no flow overlap between i->e and j->e, with xs_a_ie, xs_a_je = $(xs[a][i, e]), $(xs[a][j, e])")
            xw[a][i, e] = xw_ie_a_old + sign(xw[a][j, e] - xw_ie_a_old)
        else
            xw[a][i, e] = xw_ie_a_old + sign(xw_e_a_new - xw_ie_a_old)  ## incremental update
            # xw[a][i, e] = Int( round( lr * xw_e_a_new + (1 - lr) * xw_ie_a_old ) )  ## with a learning rate
        end

        ## update the latency-related matrices:
        if xw[a][i, e] != xw_ie_a_old
            xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]
            ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
            ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
            σp[i, e] = σp_func(xwtot[i, e], cpc[e], tf[e], s, q)
            σpp[i, e] = σpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        end

        ## update the lower-level messages:
        xs[a][i, e] = cal_xs_start_pos(xw[a][i, e], K)
        visited_lw = cal_messages_ie_a(t, i, e, j, a, xs, E, xw, ϕpp, ϕp, d, Λ, τ, adj_e, B, deg, edge_list, K)

        ## update the upper-level messages:
        visited_up = cal_messages_ie_a(t, i, e, j, a, xs, H, xw, σpp, σp, d, Λ, τ0, adj_e, B, deg, edge_list, K)

        ## update of toll:
        if update_toll && t % (4 * no_of_edges * no_of_class ÷ 10) == 0
            xw_ie_a_current = xw[a][i, e]
            xw_ie_a_target, He_a_min, overlap = cal_marginal_xw_ie_a(i, e, j, a,
                                                  xs, E, xw, σpp, σp, d, τ0, K, Λ, deg, adj_e, B, edge_list)
            τopt = cal_optimal_toll_single_d(t, i, e, j, a, xw_ie_a_current, xw_ie_a_target,
                        τmin, τmax, xs, E, xw, ϕpp, ϕp, τ, K, step_size)
            τ[e] = τopt
        end

        ## Check convergence if not in save_toll mode:
        if !save_toll && t % check_intv == 0 && t > 40 * no_of_edges * no_of_class   ## avoid early exit
            if check_convergence(xw, xw_prev, B) && check_flow_consistency(xw, d, edge_list) &&
               check_convergence(E[:, 1], E_prev[:, 1], B)
                converg = true
                tc = t
                println("Bilevel MP converge at t=$t.")
                break
            else
                record_state = true
                if record_state
                    state[t ÷ check_intv, 1] = sum( abs.( (xw[1] - xw_prev[1]) .* B) )
                    state[t ÷ check_intv, 2] = sum( abs.( (E[1, 1] - E_prev[1, 1]) .* B) )
                    state[t ÷ check_intv, 3] = sum( abs.( (H[1, 1] - H_prev[1, 1]) .* B) )
                end

                E_prev  = deepcopy(E)
                xw_prev = deepcopy(xw)
                H_prev  = deepcopy(H)
            end
        end
    end

    return converg, tc, xs, E, xw, xwtot, H, ϕpp, ϕp, τ
end


"""Compare two set of messages which are non-zero iff B[i,e]≠0."""
function check_convergence(msg::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            msg_prev::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            B::SparseMatrixCSC{Int, Int}, eps::Float64=1e-3)::Bool
    no_of_class::Int = size(msg, 1)
    max_diff::Float64 = 0.
    converg::Bool = true
    for a in 1:no_of_class
        max_diff = max( maximum( abs.((msg[a] - msg_prev[a]) .* B) ), max_diff )
        if max_diff > eps
            converg = false
            break
        end
    end
    return converg
end


"""Checking consistency of flow x: x[a][i, e] == x[a][j, e], ∀ e=<ij> and i,j≠d."""
function check_flow_consistency(x::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            d::Array{Int, 1}, edge_list::DataFrame, eps::Float64=1e-6)::Bool
    no_of_class::Int = size(d, 1)
    consistent::Bool = true
    for a in 1:no_of_class, e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        if abs(x[a][i, e] - x[a][j, e]) > eps # && i!=d && j!=d  ## need to consider the destination as well
            consistent = false
            break
        end
    end
    return consistent
end


"""Checking conservation of flow x: B[i, :] * x[a][i, :] + Λ[i] == 0, ∀i≠d."""
function check_flow_conservation(x::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            Λ::Array{Int, 2}, d::Array{Int, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame, eps::Float64=1e-6)::Bool
    no_of_class::Int = size(d, 1)
    conserve::Bool = true
    for a in 1:no_of_class, i in 1:size(deg, 1)
        res = cal_net_R_ia(i, a, x, Λ, adj_e, B, deg, edge_list)
        if res > eps
            conserve = false
            break
        end
    end
    return conserve
end


end  ## module CongestionGameMP

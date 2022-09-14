module CongestionGameMP
"""
Note: a, b are vector of size no_of_edges,
different from α, β, μ, xw, B etc which are dim-2 array.
General cost function.
Two branches of messages for effective leafs.
Multiple destinations.
"""

include("../GraphUtil.jl")
include("../CongestionGameProb.jl")

using DataFrames
using Random
using SparseArrays
using .GraphUtil
using .CongestionGameProb


ϕ_func = ϕ_q
ϕp_func = ϕp_q
ϕpp_func = ϕpp_q
σ_func = xℓ_q
σp_func = xℓp_q
σpp_func = xℓpp_q
Φ_and_H = Φ_and_H_q


"""Net resource at node i, class a."""
function net_R_ia(i::Int, a::Int, x::SparseMatrixCSC{Float64, Int}, Λ::Array{Float64, 2},
            deg::Array{Int}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int})::Float64
    R_ia = Λ[i, a]
    for n in 1:deg[i]
        e = adj_e[i, n]
        R_ia += B[i, e] * x[a][i, e]
    end
    return R_ia
end


function init_lower_messages(B::SparseMatrixCSC{Int, Int}, cpc::Array{Float64, 1}, tf::Array{Float64, 1},
            s::Float64, q::Float64, no_of_class::Int, seed::Int=200)
    Random.seed!(seed)
    α::Array{SparseMatrixCSC{Float64, Int}, 1} = [rand(size(B)) .* abs.(B) for a in 1:no_of_class]
    β::Array{SparseMatrixCSC{Float64, Int}, 1} = [rand(size(B)) .* abs.(B) for a in 1:no_of_class]
    xw::Array{SparseMatrixCSC{Float64, Int}, 1} = [rand(size(B)) .* abs.(B) for a in 1:no_of_class]
    xwtot::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    ϕpp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    ϕp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    leaf_lw::Array{SparseMatrixCSC{Int, Int}, 1} = [spzeros(Int, size(B,1), size(B,2)) for a in 1:no_of_class]
    αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2} = [spzeros(size(B,1), size(B,2)) for i in 1:4, a in 1:no_of_class]
    for i in 1:size(B, 1), e in 1:size(B, 2)
        for a in 1:no_of_class
            xwtot[i, e] += xw[a][i, e]
        end
        ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
    end
    return α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR
end


function init_upper_messages(B::SparseMatrixCSC{Int, Int}, xwtot::SparseMatrixCSC{Float64, Int},
            cpc::Array{Float64, 1}, tf::Array{Float64, 1}, s::Float64, q::Float64, no_of_class::Int, seed::Int=300)
    Random.seed!(seed)
    γ::Array{SparseMatrixCSC{Float64, Int}, 1} = [rand(size(B)) .* abs.(B) for a in 1:no_of_class]
    δ2::Array{SparseMatrixCSC{Float64, Int}, 1} = [rand(size(B)) .* abs.(B) for a in 1:no_of_class]
    σpp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    σp::SparseMatrixCSC{Float64, Int} = spzeros(size(B,1), size(B,2))
    leaf_up::Array{SparseMatrixCSC{Int, Int}, 1} = [spzeros(Int, size(B,1), size(B,2)) for a in 1:no_of_class]
    γδ_LR::Array{SparseMatrixCSC{Float64, Int}, 2} = [spzeros(size(B,1), size(B,2)) for i in 1:4, a in 1:no_of_class]
    for i in 1:size(B, 1), e in 1:size(B, 2)
        σpp[i, e] = σpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        σp[i, e] = σp_func(xwtot[i, e], cpc[e], tf[e], s, q)
    end
    return γ, δ2, σpp, σp, leaf_up, γδ_LR
end


"""Compute the value of the flow conservation constraint R^a_ie of the cavity working points xw^a_ie."""
function cal_R_ie_a(μ_ie_a::Float64, xw_ie_a::Float64, i::Int, e::Int, a::Int,
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            leaf::Array{SparseMatrixCSC{Int, Int}, 1}, xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)::Float64
    R_ie_a::Float64 = Λ[i, a]
    for m in 1:deg[i]
        epp = adj_e[i, m]
        k = end_node(edge_list, epp, i)
        if epp == e
            R_ie_a += B[i, e] * xw_ie_a
        elseif leaf[a][k, epp] == 1
            R_ie_a += B[i, epp] * xw[a][k, epp]      ## crucial difference
        else
            xopt_epp = xw[a][k, epp] - (B[i, epp] * μ_ie_a + ϕp[k, epp] + τ[epp] + β[a][k, epp]) /
                                   (α[a][k, epp] + ϕpp[k, epp])    ## already take care of the leaf node exception
            xopt_epp = max(xopt_epp, 0.)
            R_ie_a += B[i, epp] * xopt_epp
        end
    end
    return R_ie_a
end


"""Effective degree of cavity i->e for class a, excluding effective leaf nodes."""
function cal_effective_degree_ie_a(i::Int, e::Int, a::Int, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)::Int
    deg_ie_a::Int64 = deg[i] - 1
    for m in 1:deg[i]
        ep = adj_e[i, m]
        k = end_node(edge_list, ep, i)
        if ep == e
            continue
        elseif leaf[a][k, ep] == 1
            deg_ie_a -= 1
        end
    end
    return deg_ie_a
end


"""Compute the value of lagrangian multiplier μ."""
function cal_μ_ie_a(t::Int, i::Int, e::Int, a::Int, xw_ie_a::Float64,
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            leaf::Array{SparseMatrixCSC{Int, Int}, 1}, xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)
    ## Compute the kinks:
    deg_ie_a = cal_effective_degree_ie_a(i, e, a, leaf, adj_e, B, deg, edge_list)    ## effective degree in cavity tree i->e
    μ_kinks = zeros(Float64, deg_ie_a)
    ncount = 1
    for n in 1:deg[i]
        ep = adj_e[i, n]
        k = end_node(edge_list, ep, i)
        if ep == e || leaf[a][k, ep] == 1
            continue
        else
            μ_kinks[ncount] = ( (α[a][k, ep] + ϕpp[k, ep]) * xw[a][k, ep] - (β[a][k, ep] + ϕp[k, ep] + τ[ep]) ) / B[i, ep]
            ncount += 1
        end
    end

    ind = sortperm(μ_kinks)
    ## Compute the values of R_ie at the kinks:
    R_kinks = zeros(Float64, deg_ie_a)
    ncount = 1
    for n in 1:deg[i]
        ep = adj_e[i, n]
        k = end_node(edge_list, ep, i)
        if ep == e || leaf[a][k, ep] == 1
            continue
        else
            R_kinks[ncount] = cal_R_ie_a(μ_kinks[ncount], xw_ie_a, i, e, a, α, β, leaf, xw, ϕpp, ϕp, Λ, τ, adj_e, B, deg, edge_list)
            ncount += 1
        end
    end

    ## find out the kink at which g() drops below zero:
    nkinkz::Int = 1     ## the index of the first kink with negative R-value
    for m in ind[1:end]
        if R_kinks[m] < 0.
            break
        end
        nkinkz += 1
    end

    ## compute the zero of g(μ):
    slope::Float64 = 0.
    μ1::Float64, μ2::Float64 = 0.1, 0.1
    R1::Float64, R2::Float64 = 0.1, 0.1
    μ_has_solution::Bool = true
    if nkinkz == 1
        ## need to compute the slope before μ_kinks[ind[1]]:
        slope = 0.
        for n in 1:deg[i]
            ep = adj_e[i, n]
            k = end_node(edge_list, ep, i)
            if ep == e || leaf[a][k, ep] == 1
                continue
            else
                if B[i, ep] == 1
                    slope += -1. / (α[a][k, ep] + ϕpp[k, ep])
                end
            end
        end
        μ1 = μ_kinks[ind[1]]
        R1 = R_kinks[ind[1]]
        μ2 = μ1
    elseif nkinkz == deg_ie_a + 1
        ## need to compute the slope after μ_kinks[ind[end-1]]:
        slope = 0.
        for n in 1:deg[i]
            ep = adj_e[i, n]
            k = end_node(edge_list, ep, i)
            if ep == e || leaf[a][k, ep] == 1
                continue
            else
                if B[i, ep] == -1
                    slope += - 1. / (α[a][k, ep] + ϕpp[k, ep])
                end
            end
        end
        μ1 = μ_kinks[ind[nkinkz-1]]
        R1 = R_kinks[ind[nkinkz-1]]
        μ2 = μ1
    else
        μ1, μ2 = μ_kinks[ind[nkinkz-1]], μ_kinks[ind[nkinkz]]
        R1, R2 = R_kinks[ind[nkinkz-1]], R_kinks[ind[nkinkz]]
        slope = (R2 - R1) / (μ2 - μ1)
    end

    if deg_ie_a > 0
        if nkinkz==deg_ie_a+1 && abs(slope) < 1e-10
            if abs(R1) < 1e-9
                μ_soln, μ_has_solution = μ1, true
            else
                μ_soln, μ_has_solution = μ1, false
                println("Warning: at t=$t, i=$i, e=$e, a=$a, xw_ie_a=$(xw_ie_a), nkinkz=$nkinkz, R1=$R1, slope=$slope, μ has no solution.")
            end
        elseif nkinkz==1 && R1 < 0 && abs(slope) < 1e-10
            if abs(R1) < 1e-9
                μ_soln, μ_has_solution = μ1, true
            else
                μ_soln, μ_has_solution = μ1, false
                println("Warning: at t=$t, i=$i, e=$e, a=$a, xw_ie_a=$(xw_ie_a), nkinkz=$nkinkz, R1=$R1, slope=$slope, μ has no solution.")
            end
        elseif abs(slope) < 1e-10
            μ_soln = (μ1 + μ2) / 2  ## degenerate solution, pick the mid-point
        else
            μ_soln = μ1 - R1/slope
        end
    else
        ## all the descendents are leafs:
        μ_soln, μ_has_solution = 0., false
    end

    if abs(μ_soln) > 1e10
        println("Notice: at t=$t, i=$i, e=$e, a=$a, large μ_soln is observed = $(μ_soln).")     # for debug
        println("    deg_ie_a=$(deg_ie_a), μ1=$(μ1), R1=$(R1), slope=$slope, nkinkz=$nkinkz.")
    end

    return μ_soln, μ_has_solution
end


"""Divide the cost function into two part."""
function cal_xw_e_forward(α_ie::Float64, α_je::Float64, β_ie::Float64, β_je::Float64,
            xw_ie::Float64, xw_je::Float64, ϕpp_ie::Float64, ϕpp_je::Float64, ϕp_ie::Float64, ϕp_je::Float64,
            τ_e::Float64)::Float64
    xw_e_new = ( (α_ie + 0.5*ϕpp_ie)*xw_ie + (α_je + 0.5*ϕpp_je)*xw_je - ( β_ie + β_je + τ_e + 0.5*(ϕp_ie + ϕp_je) ) ) /
                ( α_ie + α_je + 0.5*(ϕpp_ie + ϕpp_je) )
    xw_e_new = max(xw_e_new, 0.)
    return xw_e_new
end


"""Symmetrize over the working point.
Only suitable for lower message."""
function cal_xw_e_forward_v2(α_ie::Float64, α_je::Float64, β_ie::Float64, β_je::Float64,
            xw_ie::Float64, xw_je::Float64, ϕpp_ie::Float64, ϕpp_je::Float64, ϕp_ie::Float64, ϕp_je::Float64,
            τ_e::Float64, cpc_e::Float64, tf_e::Float64, s::Float64, q::Float64)::Float64
    xw_e = 0.5*(xw_ie + xw_je)
    ϕpp_e = ϕpp_func(xw_e, cpc_e, tf_e, s, q)
    ϕp_e = ϕp_func(xw_e, cpc_e, tf_e, s, q)
    xw_e_new = ( α_ie*xw_ie + α_je*xw_je + ϕpp_e*xw_e - ( β_ie + β_je + τ_e + ϕp_e ) ) /
                ( α_ie + α_je + ϕpp_e )
    xw_e_new = max(xw_e_new, 0.)
    return xw_e_new
end


"""Compute the marginal value of xw_ea, with possible effective leaf state of i->e or j->e."""
function cal_marginal_xw_ea_with_leaf(i::Int64, e::Int64, j::Int64, a::Int,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, τ::Array{Float64, 1})
    if leaf[a][i, e]==0 && leaf[a][j, e]==0
        xw_ea = cal_xw_e_forward(α[a][i, e], α[a][j, e], β[a][i, e], β[a][j, e], xw[a][i, e], xw[a][j, e],
                                ϕpp[i, e], ϕpp[j, e], ϕp[i, e], ϕp[j, e], τ[e])

    elseif (leaf[a][i, e]*leaf[a][j, e])==0   ## either i->e or j->e is a leaf
        if leaf[a][j, e]==1 && leaf[a][i, e]==0
            i, j = j, i     ## let i->e index the leaf
        end
        βL_ie_a, βR_ie_a = αβ_LR[2, a][i, e], αβ_LR[4, a][i, e]
        ## slope of the full energy Ee(x) in the left and right vicinity of xw[a][i, e]:
        slopeL = (α[a][j, e] + 0.5*ϕpp[j, e])*(xw[a][i, e] - xw[a][j, e]) +
                 (β[a][j, e] + βL_ie_a + 0.5*ϕp[j, e] + 0.5*ϕp[i, e] + τ[e])
        slopeR = slopeL - βL_ie_a + βR_ie_a
        if 0 < slopeL*(-1) && 0 < slopeR
            xw_ea = xw[a][i, e]
        else
            if slopeL*(-1) < slopeR
                α_ie_a, β_ie_a = αβ_LR[1, a][i, e], αβ_LR[2, a][i, e]
            else
                α_ie_a, β_ie_a = αβ_LR[3, a][i, e], αβ_LR[4, a][i, e]
            end
            xw_ea = cal_xw_e_forward(α_ie_a, α[a][j, e], β_ie_a, β[a][j, e], xw[a][i, e], xw[a][j, e],
                                    ϕpp[i, e], ϕpp[j, e], ϕp[i, e], ϕp[j, e], τ[e])
        end

    else
        xw_ea = xw[a][i,e]
        ## pathological situation:
        println("Warning: both i->e and j->e are effective leafs, at i, e, j = $(i), $(e), $(j).")

    end
    return xw_ea
end


"""Given the messages and working points (xw), compute the new working points (xw_new)."""
function cal_all_xw(αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, d::Array{Int, 1}, Λ::Array{Float64, 2},
            τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int},
            edge_list::DataFrame)
    xw_new::Array{SparseMatrixCSC{Float64, Int}, 1} = [spzeros(size(B, 1), size(B, 2)) for a in 1:size(d, 1)]
    xwtot_new::SparseMatrixCSC{Float64, Int} = spzeros(size(B, 1), size(B, 2))
    for a in 1:size(Λ, 2), e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        xw_ea = cal_marginal_xw_ea_with_leaf(i, e, j, a, αβ_LR, leaf, α, β, xw, ϕpp, ϕp, τ)
        xw_new[a][i, e] = xw_ea
        xw_new[a][j, e] = xw_ea
        xwtot_new[i, e] += xw_ea
        xwtot_new[j, e] += xw_ea
    end
    return xw_new, xwtot_new
end


"""Leaf node propagation."""
function leaf_propagation(i::Int64, e::Int64, a::Int, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)::Int
    leaf_ie_a::Int = 1
    for m in 1:deg[i]
        ep = adj_e[i, m]
        if ep == e
            continue
        end
        k = end_node(edge_list, ep, i)
        leaf_ie_a *= leaf[a][k, ep]
    end
    return leaf_ie_a
end


"""Initial guess of whether the cavity tree i->e is an effective leaf or not,
iff e is an out-going edge of i (B[i,e]=-1), and xw[a][k,ep] vanishes for all non-leaf k->ep.
Also compute effective resource Λeff_ie at an cavity edge i->e."""
function guess_effective_leaf_ie_a(i::Int64, e::Int64, a::Int, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            Λ::Array{Float64, 2}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int},
            edge_list::DataFrame)
    leaf_ie_a::Int = 1
    Λeff_ie_a::Float64 = Λ[i, a]

    if B[i,e] != -1
        ## e is an in-coming edge of i
        leaf_ie_a = 0
    else
        ## e is an out-going edge of i
        for m in 1:deg[i]
            ep = adj_e[i, m]
            k = end_node(edge_list, ep, i)
            if ep == e
                continue
            elseif leaf[a][k, ep] == 1
                Λeff_ie_a += B[i, ep] * xw[a][k, ep]
            elseif xw[a][k, ep] > 1e-6
                leaf_ie_a = 0
            end
        end
    end

    if Λeff_ie_a < 1e-6
        leaf_ie_a = 0
    end

    return leaf_ie_a, Λeff_ie_a
end


"""Cavity E, α, β evaluated at the working point xw_ie_a.
When evaluated at a kink, E is finite, β is degenerate, α is ill-defined.
The leaf status is potentially a perturbed version (denoted as leaf_perturbed)."""
function cal_spec_leaf_cavity_Eαβ_at_xw_ie_a(t::Int, i::Int64, e::Int64, a::Int, xw_ie_a::Float64,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_perturbed::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)
    μ_ie_a, μ_has_solution = cal_μ_ie_a(t, i, e, a, xw_ie_a, α, β, leaf_perturbed, xw, ϕpp, ϕp, Λ, τ, adj_e, B, deg, edge_list)
    β_ie_a::Float64 = B[i, e] * μ_ie_a

    α_ie_a::Float64 = 1e-20
    leaf_ie_a::Int64 = 1
    max_slopeL::Float64 = 0.
    min_slopeR::Float64 = 0.
    n = 1   ## for counting the active descendent k->ep
    for m in 1:deg[i]
        ep = adj_e[i, m]
        k = end_node(edge_list, ep, i)
        if ep == e
            continue
        elseif leaf_perturbed[a][k, ep] == 1
            continue
        end

        ## optimal flow:
        xopt_ep = max( xw[a][k, ep] - (μ_ie_a*B[i, ep] + β[a][k, ep] + ϕp[k, ep] + τ[ep])/(α[a][k, ep] + ϕpp[k, ep]), 0. )
        εopt_ep = xopt_ep - xw[a][k, ep]

        ## update effective leaf status and message α_ie_a:
        if xopt_ep > 1e-10
            leaf_ie_a = 0
            α_ie_a +=  1. / (α[a][k, ep] + ϕpp[k, ep])
        end

        ## determine the most sensitive descendents k->ep:
        slope_ep = ( (α[a][k, ep] + ϕpp[k, ep])*εopt_ep + (β[a][k, ep] + ϕp[k, ep] + τ[ep]) ) *B[i, ep]    # transformed slope
        if xopt_ep < 1e-10 && B[i, ep] == 1
            slopeL_ep = -1e9         ## non-negativity constraint of x_ep
            slopeR_ep = slope_ep
        elseif xopt_ep < 1e-10 && B[i, ep] == -1
            slopeL_ep = slope_ep
            slopeR_ep = 1e9         ## non-negativity constraint of x_ep
        else
            slopeL_ep = slope_ep
            slopeR_ep = slope_ep
        end

        if n == 1
            max_slopeL = slopeL_ep
            min_slopeR = slopeR_ep
        else
            max_slopeL = max(slopeL_ep, max_slopeL)
            min_slopeR = min(slopeR_ep, min_slopeR)
        end
        n += 1
    end
    α_ie_a = 1. / α_ie_a

    return α_ie_a, β_ie_a, leaf_ie_a, max_slopeL, min_slopeR
end


"""Cavity E, α, β evaluated at the working point xw_ie_a, which is stable to perturbation.
ηconfig[n] ∈ {0,1,2} <=> {perturbed by -ηmag, perturbed by +ηmag, no perturbation}."""
function cal_stable_cavity_Eαβ_at_xw_ie_a(t::Int, i::Int64, e::Int64, a::Int, xw_ie_a::Float64,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2},
            B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, edge_list::DataFrame)
    deg_ie_a = cal_effective_degree_ie_a(i, e, a, leaf, adj_e, B, deg, edge_list)
    n_leafs_kep = deg[i]-1 - deg_ie_a  # no. of leaf descendents

    ## unperturbed cavity energy and messages
    if deg_ie_a > 0
        ## there exists active descendent:
        α_ie_a, β_ie_a, leaf_ie_a, max_slopeL, min_slopeR =
          cal_spec_leaf_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw_ie_a, αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                  d, Λ, τ, adj_e, B, deg, edge_list)
    else
        ## all descendents are inactive:
        max_slopeL, min_slopeR = 0., 0.
        α_ie_a, β_ie_a, μ_ie_a = 0., 0., 0.
        leaf_ie_a = 1
    end

    ΔE_ie_opt::Float64 = 0.         ## to record the optimal change of cavity energy due to perturbation
    indx_ηopt::Int = 2*n_leafs_kep  ## to record the optimal perturbation; initilly it corresponds to no perturbation
    ep_rsp::Int = 0                 ## to record the edge that has response to the optimal perturbation

    for indx in 0: 2*n_leafs_kep - 1
        leaf_ind = indx ÷ 2 + 1     ## the index of leaf to be perturbed
        ep_perturbed = 0            ## the edge of perturbed leaf
        branch = indx % 2           ## perturb left-ward or right-ward
        η_perturb = 2*branch - 1
        ep_rsp_indx = 0
        ΔE_ie = 0.

        n = 1      ## for counting in the set of leaf descendents
        for m in 1:deg[i]
            ep = adj_e[i, m]
            k = end_node(edge_list, ep, i)
            if ep == e
                continue
            elseif leaf[a][k, ep]==1
                ## Change of cavity energy in the leaf descendent:
                if leaf_ind==n
                    ep_perturbed = ep
                    α_ep, β_ep = αβ_LR[branch*2 + 1, a][k, ep], αβ_LR[branch*2 + 2, a][k, ep]
                    ΔE_ie += (β_ep + ϕp[k, ep] + τ[ep])* η_perturb
                    break
                else
                    n += 1
                end
            end
        end

        ## Change of cavity energy in other descendents:
        if η_perturb < 0
            ## The perturbed leaf takes the left branch, response on the right branch:
            slopeR = min_slopeR
            ep_rsp_indx = 0
            for m in 1:deg[i]
                ep = adj_e[i, m]
                k = end_node(edge_list, ep, i)
                if ep==e
                    continue
                elseif leaf[a][k, ep]==1 && ep != ep_perturbed     ## need to exclude the perturbed edge itself
                    slopeR_ep =  αβ_LR[4, a][k, ep] +  ϕp[k, ep] + τ[ep]
                    if slopeR_ep < slopeR
                        slopeR = slopeR_ep
                        ep_rsp_indx = ep
                    end
                end
            end
            ΔE_ie += slopeR * 1
        elseif η_perturb > 0
            ## The perturbed leaf takes the right branch, response on the left branch:
            slopeL = max_slopeL
            ep_rsp_indx = 0
            for m in 1:deg[i]
                ep = adj_e[i, m]
                k = end_node(edge_list, ep, i)
                if ep==e
                    continue
                elseif leaf[a][k, ep]==1 && ep != ep_perturbed     ## need to exclude the perturbed edge itself
                    slopeL_ep =  αβ_LR[2, a][k, ep] +  ϕp[k, ep] + τ[ep]
                    if slopeL_ep > slopeL
                        slopeL = slopeL_ep
                        ep_rsp_indx = ep
                    end
                end
            end
            ΔE_ie += slopeL * (-1)
        end

        if ΔE_ie < ΔE_ie_opt
            ΔE_ie_opt = ΔE_ie
            indx_ηopt = indx
            ep_rsp = ep_rsp_indx
        end
    end

    ## Redo the optimization if the optimial perturbation is non-zero:
    if indx_ηopt != 2*n_leafs_kep
        leaf_ind = indx_ηopt ÷ 2 + 1
        branch = indx_ηopt % 2
        η_perturb = 2*branch - 1

        ## Only flag the leaf descendents whose optimal perturbed state is active:
        leaf_perturbed::Array{SparseMatrixCSC{Int, Int}, 1} =
            [spzeros(Int, size(B,1), size(B,2)) for a in 1:size(Λ, 2)]
        n = 1
        for m in 1:deg[i]
            ep = adj_e[i, m]
            k = end_node(edge_list, ep, i)
            if ep == e
                continue
            elseif leaf[a][k, ep]==1
                if leaf_ind == n
                    if branch == 0
                        α[a][k, ep], β[a][k, ep] = αβ_LR[1, a][k, ep], αβ_LR[2, a][k, ep]
                        if ep_rsp != 0
                            l = end_node(edge_list, ep_rsp, i)
                            α[a][l, ep_rsp], β[a][l, ep_rsp] = αβ_LR[3, a][l, ep_rsp], αβ_LR[4, a][l, ep_rsp]
                        end
                    elseif branch == 1
                        α[a][k, ep], β[a][k, ep] = αβ_LR[3, a][k, ep], αβ_LR[4, a][k, ep]
                        if ep_rsp != 0
                            l = end_node(edge_list, ep_rsp, i)
                            α[a][l, ep_rsp], β[a][l, ep_rsp] = αβ_LR[1, a][l, ep_rsp], αβ_LR[2, a][l, ep_rsp]
                        end
                    end
                elseif ep != ep_rsp
                    leaf_perturbed[a][k, ep] = 1
                end
                n += 1
            end
        end

        α_ie_a, β_ie_a, leaf_ie_a, max_slopeL, min_slopeR =
          cal_spec_leaf_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw_ie_a, αβ_LR, leaf_perturbed, α, β, xw, ϕpp, ϕp,
                                  d, Λ, τ, adj_e, B, deg, edge_list)
    end

    return α_ie_a, β_ie_a, leaf_ie_a
end


"""For lower MP, update messages E[i,e], α[i,e], β[i, e], μ[i, e].
For upper MP, update messages H[i,e], γ[i,e], δ2[i,e], ν[i,e].
When used in upper level MP, always evaluated at xw[i, e].
When xw[a][i,e] -> 0, leaf[a][i,e] = 1, evaluated at ηmag. """
function cal_messages_ie_a_at_xw_ie_a(t::Int, i::Int64, e::Int64, j::Int64, a::Int,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ηmag::Float64, ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)
    if i == d[a]
        α_ie_a, β_ie_a, μ_ie_a = 0., 0., 0., 0.
        leaf_ie_a = 0
        αβ_LR_ie_a = zeros(4)
    else
        α_ie_a, β_ie_a, leaf_ie_a =
          cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e], αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                         d, Λ, τ, adj_e, B, deg, edge_list)
        if leaf_ie_a == 0
            αβ_LR_ie_a = zeros(4)

        elseif xw[a][i,e] < 1e-6
            ## leaf_ie_a=1, but evaluated at the working point xw[a][i,e] -> 0, so there is no left-side message.
            ## Consider i->e as a normal edge, and use the right-side message as the usual message.
            α_ie_a, β_ie_a, leaf_ie_a =
              cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, ηmag, αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                              d, Λ, τ, adj_e, B, deg, edge_list)
            αβ_LR_ie_a = zeros(4)

            if leaf_ie_a == 1
                println("Warning: at t=$t, i=$i, e=$e, a=$a, a kink at xw[a][i,e]->0, but right-side message ill-defined.")
            end

        else
            ## leaf_ie_a=1, evaluated at xw[a][i,e] > 0, need two double-sided messages here:
            α_ie_a_0, β_ie_a_0, leaf_ie_a_0 =
              cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e]*(1-ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                             d, Λ, τ, adj_e, B, deg, edge_list)
            α_ie_a_1, β_ie_a_1, leaf_ie_a_1 =
              cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e]*(1+ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                              d, Λ, τ, adj_e, B, deg, edge_list)
            αβ_LR_ie_a = [α_ie_a_0, β_ie_a_0, α_ie_a_1, β_ie_a_1]
        end
    end

    return α_ie_a, β_ie_a, leaf_ie_a, xw[a][i,e], αβ_LR_ie_a
end


"""For lower MP, update messages E[i,e], α[i,e], β[i, e], μ[i, e].
For upper MP, update messages H[i,e], γ[i,e], δ2[i,e], ν[i,e]."""
function cal_messages_ie_a(t::Int, i::Int64, e::Int64, j::Int64, a::Int,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ηmag::Float64, ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, Λ::Array{Float64, 2}, τ::Array{Float64, 1}, adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int},
            deg::Array{Int}, edge_list::DataFrame)
    xw_ie_a_new = xw[a][i, e]
    if i == d[a]
        α_ie_a, β_ie_a, μ_ie_a = 0., 0., 0., 0.
        leaf_ie_a = 0
        αβ_LR_ie_a = zeros(4)
    else
        leaf_ie_a_guess, Λeff_ie_a = guess_effective_leaf_ie_a(i, e, a, leaf, xw, Λ, adj_e, B, deg, edge_list)
        ## attempt with xw_ie_a = Λeff_ie_a:
        if leaf_ie_a_guess == 1 ||
          (leaf[a][i, e] == 0 && B[i, e] == -1 && Λeff_ie_a > 1e-6 && abs(xw[a][i, e] - Λeff_ie_a) < 0.2*xw[a][i, e] )
            α_ie_a, β_ie_a, leaf_ie_a =
              cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, Λeff_ie_a, αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                             d, Λ, τ, adj_e, B, deg, edge_list)
            if leaf_ie_a == 0
                ## attempt fails, use the normal working points:
                α_ie_a, β_ie_a, leaf_ie_a =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e], αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                 d, Λ, τ, adj_e, B, deg, edge_list)
                αβ_LR_ie_a = zeros(4)
            else
                ## leaf_ie_a=1, need two double-sided messages here:
                α_ie_a_0, β_ie_a_0, leaf_ie_a_0 =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, Λeff_ie_a*(1-ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                 d, Λ, τ, adj_e, B, deg, edge_list)
                α_ie_a_1, β_ie_a_1, leaf_ie_a_1 =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, Λeff_ie_a*(1+ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                  d, Λ, τ, adj_e, B, deg, edge_list)

                if leaf_ie_a_0==1 || leaf_ie_a_1==1
                    ## it is expected the vicinity of the kink xw=Λeff_ie_a is smooth.
                    println("Warning: at t=$t, i=$i, e=$e, a=$a, Meassages α, β ill-defined near the kink xw=Λeff_ie_a.")
                end

                ## To probe the landscape of E^full(x) under the new messages:
                slopeEx_L = (α[a][j, e] + 0.5*ϕpp[j, e])*(Λeff_ie_a - xw[a][j, e]) +
                            (β[a][j, e] + β_ie_a_0 + 0.5*ϕp[j, e] + 0.5*ϕp[i, e] + τ[e])
                slopeEx_R = slopeEx_L - β_ie_a_0 + β_ie_a_1

                if slopeEx_L < 0 < slopeEx_R
                    ## Keep the effective leaf state:
                    leaf_ie_a = 1
                    αβ_LR_ie_a = [α_ie_a_0, β_ie_a_0, α_ie_a_1, β_ie_a_1]
                    xw_ie_a_new = Λeff_ie_a
                else
                    ## Abandom the effective leaf state:
                    leaf_ie_a = 0
                    α_ie_a, β_ie_a, leaf_ie_a_2 =
                      cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e], αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                     d, Λ, τ, adj_e, B, deg, edge_list)
                    αβ_LR_ie_a = zeros(4)
                end

            end
        else
            α_ie_a, β_ie_a, leaf_ie_a =
              cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e], αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                             d, Λ, τ, adj_e, B, deg, edge_list)
            if leaf_ie_a == 0
                αβ_LR_ie_a = zeros(4)

            elseif xw[a][i,e] < 1e-6
                ## leaf_ie_a=1, but evaluated at the working point xw[a][i,e] -> 0, so there is no left-side message.
                ## Consider i->e as a normal edge, and use the right-side message as the usual message.
                α_ie_a, β_ie_a, leaf_ie_a =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, ηmag, αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                  d, Λ, τ, adj_e, B, deg, edge_list)
                αβ_LR_ie_a = zeros(4)

                if leaf_ie_a == 1
                    println("Warning: at t=$t, i=$i, e=$e, a=$a, a kink at xw[a][i,e]->0, but right-side message ill-defined.")
                end

            else
                ## leaf_ie_a=1, evaluated at xw[a][i,e] > 0, need two double-sided messages here:
                α_ie_a_0, β_ie_a_0, leaf_ie_a_0 =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e]*(1-ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                 d, Λ, τ, adj_e, B, deg, edge_list)
                α_ie_a_1, β_ie_a_1, leaf_ie_a_1 =
                  cal_stable_cavity_Eαβ_at_xw_ie_a(t, i, e, a, xw[a][i,e]*(1+ηmag), αβ_LR, leaf, α, β, xw, ϕpp, ϕp,
                                                  d, Λ, τ, adj_e, B, deg, edge_list)
                αβ_LR_ie_a = [α_ie_a_0, β_ie_a_0, α_ie_a_1, β_ie_a_1]
            end
        end
    end

    return α_ie_a, β_ie_a, leaf_ie_a, xw_ie_a_new, αβ_LR_ie_a
end


"""Lower level message passing, random sequential update."""
function lower_MP_rand_seq!(αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_lw::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1}, xwtot::SparseMatrixCSC{Float64, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, d::Array{Int, 1}, cpc::Array{Float64, 1},
            tf::Array{Float64, 1}, s::Float64, q::Float64, Λ::Array{Float64, 2}, τ::Array{Float64, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, outdeg::Array{Int}, edge_list::DataFrame,
            state::Array{Float64, 2}, max_iter::Int, check_intv::Int, lr::Float64, seed::Int)
    if seed > 0
        Random.seed!(seed)
    end
    ηmag::Float64 = 0.005
    no_of_nodes::Int, no_of_edges::Int, no_of_class::Int = size(Λ, 1), size(tf, 1), size(d, 1)

    α_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(α)
    β_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(β)
    xw_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(xw)
    leaf_lw_prev::Array{SparseMatrixCSC{Int, Int}, 1} = deepcopy(leaf_lw)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1:max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        e = adj_e[i, n]
        j = end_node(edge_list, e, i)
        a = rand(1:no_of_class)

        ## update the messages:
        α[a][i, e], β[a][i, e], leaf_lw[a][i, e], xw_ie_a_new, αβ_LR_ie_a =
          cal_messages_ie_a(t, i, e, j, a, αβ_LR, leaf_lw, α, β, xw, ηmag, ϕpp, ϕp,
                            d, Λ, τ, adj_e, B, deg, edge_list)

        ## unpack the double-sided messages when i->e is in leaf state:
        if leaf_lw[a][i, e] == 1
            for m in 1:4
                αβ_LR[m, a][i, e] = αβ_LR_ie_a[m]
            end
            ## the working point evaluated is decided inside the cal_messages_ie() process:
            xw_ie_a_old = xw[a][i, e]
            xw[a][i, e] = xw_ie_a_new
            xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]
        end

        ## update the working point by forward provision:
        xw_ie_a_old = xw[a][i, e]
        xw_e_a_new = cal_marginal_xw_ea_with_leaf(i, e, j, a, αβ_LR, leaf_lw, α, β, xw, ϕpp, ϕp, τ)
        xw[a][i, e] = lr * xw_e_a_new + (1 - lr) * xw_ie_a_old
        xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]

        ## update the latency-related matrices:
        ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)

        if t % check_intv == 0
            record_state = false
            if record_state
                state[t ÷ check_intv, 1] = sum( abs.( (α[1] - α_prev[1]) .* B ) )
                state[t ÷ check_intv, 2] = sum( abs.( (β[1] - β_prev[1]) .* B) )
                state[t ÷ check_intv, 3] = sum( abs.( (xw[1] - xw_prev[1]) .* B) )
                state[t ÷ check_intv, 4] = sum( abs.( (leaf_lw[1] - leaf_lw_prev[1]) .* B) )
            end

            if check_convergence(xw, xw_prev, B) && check_flow_consistency(xw, d, edge_list) &&
               check_convergence(β, β_prev, B)
                converg = true
                tc = t
                print("Lower MP converge at t=$t.")
                break
            else
                α_prev = deepcopy(α)
                β_prev = deepcopy(β)
                xw_prev = deepcopy(xw)
                leaf_lw_prev = deepcopy(leaf_lw)
            end
        end
    end

    return converg, tc
end


"""Piecewise linear function for toll optimization,
At most 3 kinks; the right-most branch has slope 0."""
function piecewise_linear_func(x::Float64, no_of_kinks::Int, leftmost_slope::Float64,
            kink_coordinates::Array{Float64, 2})::Float64
    if x <= kink_coordinates[1, 1]
        ## the input lies within the left most branch:
        y = leftmost_slope * (x - kink_coordinates[1, 1]) + kink_coordinates[1, 2]
    elseif x > kink_coordinates[no_of_kinks, 1]
        ## the input lies within the right most branch:
        y = kink_coordinates[no_of_kinks, 2]
    else
        for i in 1:no_of_kinks-1
            if kink_coordinates[i, 1] < x <= kink_coordinates[i+1, 1]
                slope = (kink_coordinates[i+1, 2] - kink_coordinates[i, 2]) /
                        (kink_coordinates[i+1, 1] - kink_coordinates[i, 1])
                y = slope * (x - kink_coordinates[i, 1]) + kink_coordinates[i, 2]
                break
            end
        end
    end
    return y
end


"""Compute the total Nash-equilibrium flow xN_e at a particular toll τ_e."""
function cal_xN_at_τ(τ_e::Float64, active_class::Array{Int, 1}, nf_kinks::Array{Int, 1},
            lm_slopes::Array{Float64, 1}, kk_coords::Array{Float64, 3})::Float64
    no_of_class::Int = size(active_class, 1)
    xN_e::Float64 = 0.
    for a in 1:no_of_class
        if active_class[a] == 0
            continue
        end
        xN_e += piecewise_linear_func(τ_e, nf_kinks[a], lm_slopes[a], kk_coords[:, :, a])
    end
    return xN_e
end


"""Search of optimal toll, by bringing x^*N_ie as close as possible to x^*G_ie."""
function cal_optimal_toll_nonnegative(t::Int64, i::Int64, e::Int64, j::Int64,
            τmin::Array{Float64, 1}, τmax::Array{Float64, 1}, τ::Array{Float64, 1},
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_lw::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1}, xwtot::SparseMatrixCSC{Float64, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            γδ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_up::Array{SparseMatrixCSC{Int, Int}, 1},
            γ::Array{SparseMatrixCSC{Float64, Int}, 1}, δ2::Array{SparseMatrixCSC{Float64, Int}, 1},
            σpp::SparseMatrixCSC{Float64, Int}, σp::SparseMatrixCSC{Float64, Int})
    if τmax[e] - τmin[e] < 1e-5
        return 0., τmin[e]
    end

    no_of_class::Int = size(α, 1)
    no_of_edges::Int = size(τ, 1)

    ## targeted global optimum of total flow on edge e:
    xG_e::Float64 = 0.
    for a in 1:no_of_class
        xG_ea = cal_marginal_xw_ea_with_leaf(i, e, j, a, γδ_LR, leaf_up, γ, δ2, xw, σpp, σp, zeros(no_of_edges))
        xG_e += xG_ea
    end

    ## encoding the piecewise linear function xN^a(τ) of each class:
    active_class::Array{Int, 1} = ones(Int, no_of_class)   ## zero indicates idle edge flow for that class, no need to consider
    nf_kinks::Array{Int, 1} = zeros(Int, no_of_class)   ## no. of kinks of each class
    lm_slopes::Array{Float64, 1} = zeros(Float64, no_of_class)   ## left most slope of each class
    kk_coords::Array{Float64, 3} = zeros(Float64, 3, 2, no_of_class)   ## kink coordinates of each class
    for a in 1:no_of_class
        if leaf_lw[a][i, e]==0 && leaf_lw[a][j, e]==0   ## normal edge with smooth cavity energy
            τ3 = (α[a][i, e] + 0.5*ϕpp[i, e])* xw[a][i, e] + (α[a][j, e] + 0.5*ϕpp[j, e])* xw[a][j, e] -
                 (β[a][i, e] + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e])

            nf_kinks[a] = 1
            lm_slopes[a] = -1 / (α[a][i, e] + α[a][j, e] + 0.5*ϕpp[i, e] + 0.5*ϕpp[j, e])
            kk_coords[1, :, a] = [τ3, 0.]

        elseif (leaf_lw[a][i, e]*leaf_lw[a][j, e])==0   ## either i->e or j->e is a leaf
            if leaf_lw[a][j, e]==1 && leaf_lw[a][i, e]==0
                i, j = j, i     ## let i->e index the leaf
            end

            αL_ie_a, βL_ie_a, αR_ie_a, βR_ie_a = [αβ_LR[m, a][i, e] for m in 1:4]
            τ3 = (αL_ie_a + 0.5*ϕpp[i, e])*xw[a][i, e] + (α[a][j, e] + 0.5*ϕpp[j, e])*xw[a][j, e] -
                 (βL_ie_a + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e])

            ## slope_ExL <= slope_ExR are the slopes of E^full at xw[a][i, e] before toll update:
            slope_ExL = (α[a][j, e] + 0.5*ϕpp[j, e])*(xw[a][i, e] - xw[a][j, e]) +
                        (βL_ie_a + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e] + τ[e])
            slope_ExR = slope_ExL - βL_ie_a + βR_ie_a
            τ1 = - (slope_ExR - τ[e])
            τ2 = - (slope_ExL - τ[e])

            ## case = 1, 2, 3 <=> the kink of cavity energy in the middle, left, right:
            if slope_ExL <= 0 <= slope_ExR      ## case 1
                nf_kinks[a] = 2
                lm_slopes[a] = 0.
                kk_coords[1, :, a] = [τ2, xw[a][i, e]]
                kk_coords[2, :, a] = [τ3, 0.]
            elseif slope_ExL <= slope_ExR <= 0  ## case 2
                nf_kinks[a] = 3
                lm_slopes[a] = -1 / (αR_ie_a + α[a][j, e] + 0.5*ϕpp[i, e] + 0.5*ϕpp[j, e])
                kk_coords[1, :, a] = [τ1, xw[a][i, e]]
                kk_coords[2, :, a] = [τ2, xw[a][i, e]]
                kk_coords[3, :, a] = [τ3, 0.]
            elseif 0 <= slope_ExL <= slope_ExR  ## case 3
                nf_kinks[a] = 1
                lm_slopes[a] = -1 / (αL_ie_a + α[a][j, e] + 0.5*ϕpp[i, e] + 0.5*ϕpp[j, e])
                kk_coords[1, :, a] = [τ3, 0.]
            end
        end
    end

    ## compute the kink coordinates of xN_e(τ):
    total_kinks::Int = sum(nf_kinks)
    τ_kinks::Array{Float64, 1} = zeros(Float64, total_kinks)
    xN_kinks::Array{Float64, 1} = zeros(Float64, total_kinks)
    ncount = 1
    for a in 1:no_of_class
        if active_class[a] == 0
            continue
        end
        for m in 1:nf_kinks[a]
            τ_kinks[ncount] = kk_coords[m, 1, a]
            xN_kinks[ncount] = cal_xN_at_τ(τ_kinks[ncount], active_class, nf_kinks, lm_slopes, kk_coords)
            ncount += 1
        end
    end

    ind = sortperm(τ_kinks)
    τopt::Float64 = 0.

    ## if xG_e = 0, return the last τ_kink position:
    if xG_e < 1e-9
        τopt = τ_kinks[ind[end]]
        τopt = max(τopt, τmin[e])
        τopt = min(τopt, τmax[e])
        return xG_e, τopt
    end

    ## if xG_e > 0, search for optimal toll:
    nkinkz::Int = 1
    for m in ind
        if abs( xN_kinks[m] - xG_e ) < 1e-8
            if nkinkz==1 && abs(sum(lm_slopes)) < 1e-8
                ## return τopt = 0 if the it is the leftmost kink, with zero leftmost slope:
                τopt = 0.
            else
                ## return the τ_kink position as the solution:
                τopt = τ_kinks[m]
            end
            τopt = max(τopt, τmin[e])
            τopt = min(τopt, τmax[e])
            return xG_e, τopt

        elseif xN_kinks[m] <= xG_e
            ## xG_e lies between xN_kinks[m-1] and xN_kinks[m]:
            break
        end
        nkinkz += 1
    end

    slope::Float64 = 0.
    τ1::Float64, τ2::Float64 = 0., 0.
    xN1::Float64, xN2::Float64 = 0., 0.
    if nkinkz == 1
        slope = sum(lm_slopes)
        τ1, xN1 = τ_kinks[ind[1]], xN_kinks[ind[1]]

        if xN1 < xG_e && abs(slope) < 1e-10
            τopt = τmin[e]     ## xG_e not enforcable
        else
            τopt = τ1 + (xG_e - xN1) / slope
        end
    elseif nkinkz == total_kinks + 1    ## may occur due to imprecise xG_e
        τopt = τ_kinks[ind[end]]
    else
        τ1, τ2 = τ_kinks[ind[nkinkz-1]], τ_kinks[ind[nkinkz]]
        xN1, xN2 = xN_kinks[ind[nkinkz-1]], xN_kinks[ind[nkinkz]]
        slope = (xN2 - xN1) / (τ2 - τ1)
        τopt = τ1 + (xG_e - xN1) / slope
    end

    τopt = max(τopt, τmin[e])
    τopt = min(τopt, τmax[e])

    return xG_e, τopt
end


"""Search of optimal toll, by bringing x^*N_ie as close as possible to x^*G_ie."""
function cal_optimal_toll(t::Int64, i::Int64, e::Int64, j::Int64,
            τmin::Array{Float64, 1}, τmax::Array{Float64, 1}, τ::Array{Float64, 1},
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_lw::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1}, xwtot::SparseMatrixCSC{Float64, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            γδ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_up::Array{SparseMatrixCSC{Int, Int}, 1},
            γ::Array{SparseMatrixCSC{Float64, Int}, 1}, δ2::Array{SparseMatrixCSC{Float64, Int}, 1},
            σpp::SparseMatrixCSC{Float64, Int}, σp::SparseMatrixCSC{Float64, Int})
    if τmax[e] - τmin[e] < 1e-5
        return 0., τmin[e]
    end

    no_of_class::Int = size(α, 1)
    no_of_edges::Int = size(τ, 1)

    ## targeted global optimum of total flow on edge e:
    xG_e::Float64 = 0.
    for a in 1:no_of_class
        xG_ea = cal_marginal_xw_ea_with_leaf(i, e, j, a, γδ_LR, leaf_up, γ, δ2, xw, σpp, σp, zeros(no_of_edges))
        xG_e += xG_ea
    end

    ## encoding the piecewise linear function xN^a(τ) of each class:
    active_class::Array{Int, 1} = ones(Int, no_of_class)   ## zero indicates idle edge flow for that class, no need to consider
    nf_kinks::Array{Int, 1} = zeros(Int, no_of_class)   ## no. of kinks of each class
    lm_slopes::Array{Float64, 1} = zeros(Float64, no_of_class)   ## left most slope of each class
    kk_coords::Array{Float64, 3} = zeros(Float64, 3, 2, no_of_class)   ## kink coordinates of each class
    for a in 1:no_of_class
        if leaf_lw[a][i, e]==0 && leaf_lw[a][j, e]==0   ## normal edge with smooth cavity energy
            τ3 = (α[a][i, e] + 0.5*ϕpp[i, e])* xw[a][i, e] + (α[a][j, e] + 0.5*ϕpp[j, e])* xw[a][j, e] -
                 (β[a][i, e] + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e])

            nf_kinks[a] = 1
            lm_slopes[a] = -1 / (α[a][i, e] + α[a][j, e] + 0.5*ϕpp[i, e] + 0.5*ϕpp[j, e])
            kk_coords[1, :, a] = [τ3, 0.]

        elseif (leaf_lw[a][i, e]*leaf_lw[a][j, e])==0   ## either i->e or j->e is a leaf
            if leaf_lw[a][j, e]==1 && leaf_lw[a][i, e]==0
                i, j = j, i     ## let i->e index the leaf
            end

            αL_ie_a, βL_ie_a, αR_ie_a, βR_ie_a = [αβ_LR[m, a][i, e] for m in 1:4]
            τ3 = (αL_ie_a + 0.5*ϕpp[i, e])*xw[a][i, e] + (α[a][j, e] + 0.5*ϕpp[j, e])*xw[a][j, e] -
                 (βL_ie_a + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e])

            ## slope_ExL <= slope_ExR are the slopes of E^full at xw[a][i, e] before toll update:
            slope_ExL = (α[a][j, e] + 0.5*ϕpp[j, e])*(xw[a][i, e] - xw[a][j, e]) +
                        (βL_ie_a + β[a][j, e] + 0.5*ϕp[i, e] + 0.5*ϕp[j, e] + τ[e])
            slope_ExR = slope_ExL - βL_ie_a + βR_ie_a
            τ1 = - (slope_ExR - τ[e])
            τ2 = - (slope_ExL - τ[e])

            nf_kinks[a] = 3
            lm_slopes[a] = -1 / (αR_ie_a + α[a][j, e] + 0.5*ϕpp[i, e] + 0.5*ϕpp[j, e])
            kk_coords[1, :, a] = [τ1, xw[a][i, e]]
            kk_coords[2, :, a] = [τ2, xw[a][i, e]]
            kk_coords[3, :, a] = [τ3, 0.]
        end
    end

    ## compute the kink coordinates of xN_e(τ):
    total_kinks::Int = sum(nf_kinks)
    τ_kinks::Array{Float64, 1} = zeros(Float64, total_kinks)
    xN_kinks::Array{Float64, 1} = zeros(Float64, total_kinks)
    ncount = 1
    for a in 1:no_of_class
        if active_class[a] == 0
            continue
        end
        for m in 1:nf_kinks[a]
            τ_kinks[ncount] = kk_coords[m, 1, a]
            xN_kinks[ncount] = cal_xN_at_τ(τ_kinks[ncount], active_class, nf_kinks, lm_slopes, kk_coords)
            ncount += 1
        end
    end

    ind = sortperm(τ_kinks)
    τopt::Float64 = 0.

    ## if xG_e = 0, return the last τ_kink position:
    if xG_e < 1e-9
        τopt = τ_kinks[ind[end]]
        τopt = max(τopt, τmin[e])
        τopt = min(τopt, τmax[e])
        return xG_e, τopt
    end

    ## if xG_e > 0, search for optimal toll:
    nkinkz::Int = 1
    for m in ind
        if abs( xN_kinks[m] - xG_e ) < 1e-8
            ## return the τ_kink position as the solution:
            τopt = τ_kinks[m]

            τopt = max(τopt, τmin[e])
            τopt = min(τopt, τmax[e])
            return xG_e, τopt

        elseif xN_kinks[m] <= xG_e
            ## xG_e lies between xN_kinks[m-1] and xN_kinks[m]:
            break
        end
        nkinkz += 1
    end

    slope::Float64 = 0.
    τ1::Float64, τ2::Float64 = 0., 0.
    xN1::Float64, xN2::Float64 = 0., 0.
    if nkinkz == 1
        slope = sum(lm_slopes)
        τ1, xN1 = τ_kinks[ind[1]], xN_kinks[ind[1]]
    else
        τ1, τ2 = τ_kinks[ind[nkinkz-1]], τ_kinks[ind[nkinkz]]
        xN1, xN2 = xN_kinks[ind[nkinkz-1]], xN_kinks[ind[nkinkz]]
        slope = (xN2 - xN1) / (τ2 - τ1)
    end

    τopt = τ1 + (xG_e - xN1) / slope

    τopt = max(τopt, τmin[e])
    τopt = min(τopt, τmax[e])
    return xG_e, τopt
end


"""Bi-level message passing."""
function bilevel_MP_rand_seq!(αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_lw::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1}, xwtot::SparseMatrixCSC{Float64, Int},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int},
            γδ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf_up::Array{SparseMatrixCSC{Int, Int}, 1},
            γ::Array{SparseMatrixCSC{Float64, Int}, 1}, δ2::Array{SparseMatrixCSC{Float64, Int}, 1},
            σpp::SparseMatrixCSC{Float64, Int}, σp::SparseMatrixCSC{Float64, Int},
            d::Array{Int, 1}, cpc::Array{Float64, 1}, tf::Array{Float64, 1}, s::Float64, q::Float64, Λ::Array{Float64, 2},
            τ::Array{Float64, 1}, τmin::Array{Float64, 1}, τmax::Array{Float64, 1},
            adj_e::Array{Int, 2}, B::SparseMatrixCSC{Int, Int}, deg::Array{Int}, outdeg::Array{Int},
            edge_list::DataFrame, state::Array{Float64, 2}, max_iter::Int, check_intv::Int, lr::Float64,
            seed::Int=300, update_toll::Bool=true; save_toll::Bool=false)
    if seed > 0
        Random.seed!(seed)
    end
    ηmag::Float64 = 0.005
    no_of_nodes::Int, no_of_edges::Int, no_of_class::Int = size(Λ, 1), size(tf, 1), size(d, 1)

    α_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(α)
    β_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(β)
    xw_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(xw)
    leaf_lw_prev::Array{SparseMatrixCSC{Int, Int}, 1} = deepcopy(leaf_lw)
    γ_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(γ)
    δ2_prev::Array{SparseMatrixCSC{Float64, Int}, 1} = deepcopy(δ2)
    leaf_up_prev::Array{SparseMatrixCSC{Int, Int}, 1} = deepcopy(leaf_up)
    converg::Bool = false
    tc::Int = max_iter

    for t in 1: max_iter
        i = rand(1:no_of_nodes)
        n = rand(1:deg[i])
        e = adj_e[i, n]
        j = end_node(edge_list, e, i)
        a = rand(1:no_of_class)

        ## update the lower-level messages:
        α[a][i, e], β[a][i, e], leaf_lw[a][i, e], xw_ie_a_new, αβ_LR_ie_a =
          cal_messages_ie_a(t, i, e, j, a, αβ_LR, leaf_lw, α, β, xw, ηmag, ϕpp, ϕp,
                            d, Λ, τ, adj_e, B, deg, edge_list)

        ## unpack the double-sided messages when i->e is in leaf state:
        if leaf_lw[a][i, e] == 1
            for m in 1:4
                αβ_LR[m, a][i, e] = αβ_LR_ie_a[m]
            end
            ## the working point evaluated is decided inside the cal_messages_ie() process:
            xw_ie_a_old = xw[a][i, e]
            xw[a][i, e] = xw_ie_a_new
            xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]
        end

        ## update the upper-level messages:
        γ[a][i, e], δ2[a][i, e], leaf_up[a][i, e], xw_ie_a_new, γδ_LR_ie_a =
          cal_messages_ie_a_at_xw_ie_a(t, i, e, j, a, γδ_LR, leaf_up, γ, δ2, xw, ηmag, σpp, σp,
                            d, Λ, zeros(no_of_edges), adj_e, B, deg, edge_list)

        ## unpack the double-sided messages when i->e is in leaf state:
        if leaf_up[a][i, e] == 1
            for m in 1:4
                γδ_LR[m, a][i, e] = γδ_LR_ie_a[m]
            end
        end

        ## update the working point by forward provision, according to LOWER-LEVEL messages:
        xw_ie_a_old = xw[a][i, e]
        xw_e_a_new = cal_marginal_xw_ea_with_leaf(i, e, j, a, αβ_LR, leaf_lw, α, β, xw, ϕpp, ϕp, τ)
        xw[a][i, e] = lr * xw_e_a_new + (1 - lr) * xw_ie_a_old
        xwtot[i, e] = xwtot[i, e] - xw_ie_a_old + xw[a][i, e]

        ## update the latency-related matrices:
        ϕp[i, e] = ϕp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        ϕpp[i, e] = ϕpp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        σp[i, e] = σp_func(xwtot[i, e], cpc[e], tf[e], s, q)
        σpp[i, e] = σpp_func(xwtot[i, e], cpc[e], tf[e], s, q)

        ## update of toll:
        if update_toll && t % (4 * no_of_edges * no_of_class ÷ 10) == 0
            xG_e, τopt = cal_optimal_toll_nonnegative(t, i, e, j, τmin, τmax, τ,
                αβ_LR, leaf_lw, α, β, xw, xwtot, ϕpp, ϕp, γδ_LR, leaf_up, γ, δ2, σpp, σp)

            τ[e] = τopt
            ## Alternatively use a learning rate:
            # lr_τ = 0.5
            # τ[e] = lr_τ * τopt + (1 - lr_τ) * τ[e]
        end

        ## Check convergence if not in save_toll mode:
        if !save_toll && t % check_intv == 0 && t > max_iter÷10     ## avoid early exit
            if check_convergence(xw, xw_prev, B) && check_flow_consistency(xw, d, edge_list) &&
               check_convergence(β, β_prev, B) && check_convergence(δ2, δ2_prev, B)
                converg = true
                tc = t
                println("Bilevel MP converge at t=$t.")
                break
            else
                record_state = false
                if record_state
                    state[t ÷ check_intv, 1] = sum( abs.( (β[1] - β_prev[1]) .* B) )
                    state[t ÷ check_intv, 2] = sum( abs.( (δ2[1] - δ2_prev[1]) .* B) )
                    state[t ÷ check_intv, 3] = sum( abs.( (xw[1] - xw_prev[1]) .* B) )
                    state[t ÷ check_intv, 4] = sum( abs.( (leaf_up[1] - leaf_up_prev[1]) .* B) )
                end

                β_prev = deepcopy(β)
                δ2_prev = deepcopy(δ2)
                xw_prev = deepcopy(xw)
                leaf_lw_prev = deepcopy(leaf_lw)
                leaf_up_prev = deepcopy(leaf_up)
            end
        end
    end

    return converg, tc, α, β, xw, xwtot, ϕpp, ϕp, leaf_lw, αβ_LR
end


"""Compute the value of the full energy.
May not be precise in the multi-class case, due to the approximation used there."""
function cal_full_energy_at_x_e(x_e::Array{Float64, 1}, i::Int64, e::Int64, j::Int64,
            αβ_LR::Array{SparseMatrixCSC{Float64, Int}, 2}, leaf::Array{SparseMatrixCSC{Int, Int}, 1},
            α::Array{SparseMatrixCSC{Float64, Int}, 1}, β::Array{SparseMatrixCSC{Float64, Int}, 1},
            xw::Array{SparseMatrixCSC{Float64, Int}, 1},
            ϕpp::SparseMatrixCSC{Float64, Int}, ϕp::SparseMatrixCSC{Float64, Int}, τ::Array{Float64, 1})
    no_of_class = size(xw, 1)
    xw_ie = sum([xw[a][i, e] for a in 1:no_of_class])
    xw_je = sum([xw[a][j, e] for a in 1:no_of_class])
    xtot_e = sum(x_e)

    E::Float64 = 0.
    for a in 1:no_of_class
        if leaf[a][i, e] == 0 && leaf[a][j, e] == 0
            E += 0.5*α[a][i, e]*(x_e[a] - xw[a][i, e])^2 + β[a][i, e]*(x_e[a] - xw[a][i, e])
            E += 0.5*α[a][j, e]*(x_e[a] - xw[a][j, e])^2 + β[a][j, e]*(x_e[a] - xw[a][j, e])

        elseif leaf[a][i, e] == 1 && leaf[a][j, e] == 1
            println("Inside cal_full_energy_at_x_e at i,e,j=$i,$e,$j, error of leaf state.")

        else
            if leaf[a][i, e] == 0 && leaf[a][j, e] == 1
                i, j = j, i     ## Let i->e index the leaf edge
            end
            α_ie_a_0, β_ie_a_0, α_ie_a_1, β_ie_a_1 = [ αβ_LR[m, a][i, e] for m in 1:4 ]

            if x_e[a] < xw[a][i, e]
                E += 0.5*α_ie_a_0*(x_e[a] - xw[a][i, e])^2 + β_ie_a_0*(x_e[a] - xw[a][i, e])
            else
                E += 0.5*α_ie_a_1*(x_e[a] - xw[a][i, e])^2 + β_ie_a_1*(x_e[a] - xw[a][i, e])
            end

            E += 0.5*α[a][j, e]*(x_e[a] - xw[a][j, e])^2 + β[a][j, e]*(x_e[a] - xw[a][j, e])
        end
    end

    E += 0.25*ϕpp[i, e]*(xtot_e - xw_ie)^2 + 0.5*ϕp[i, e]*(xtot_e - xw_ie)
    E += 0.25*ϕpp[j, e]*(xtot_e - xw_je)^2 + 0.5*ϕp[j, e]*(xtot_e - xw_je)
    E += τ[e] * xtot_e

    return E
end


"""Compare two set of messages which are non-zero iff B[i,e] ≠ 0."""
function check_convergence(msg::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            msg_prev::Union{Array{SparseMatrixCSC{Float64, Int}, 1}, Array{SparseMatrixCSC{Int, Int}, 1}},
            B::SparseMatrixCSC{Int, Int}, eps::Float64=1e-6)::Bool
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
function check_flow_consistency(x::Array{SparseMatrixCSC{Float64, Int}, 1}, d::Array{Int, 1},
            edge_list::DataFrame, eps::Float64=1e-6)::Bool
    no_of_class::Int = size(d, 1)
    consistent::Bool = true
    for a in 1:no_of_class, e in 1:size(edge_list, 1)
        i, j = edge_list[e, :]
        if abs(x[a][i, e] - x[a][j, e]) > eps # && i!=d && j!=d
            consistent = false
            break
        end
    end
    return consistent
end


end  ## module CongestionGameMP

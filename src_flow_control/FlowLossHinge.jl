"""The dissipative energy function.
The flow y_ij is taken to be the average of |y[i,j]| and |y[j,i]|."""
function energy_func(y::Array{Float64, 1}, r::Array{Float64, 1})::Float64
    E::Float64 = 0.

    for e in 1:size(r, 1)
        E += 0.5 * r[e] * y[e]^2
    end

    return E
end


"""Objective function of the upper layer, and number of successfully tuned edges.
Hinge loss."""
function objective_upper_and_tuned_edges(y::Array{Float64, 1}, y0::Array{Float64, 1},
            targets::Array{Int, 1}, threshold::Float64)
    H::Float64 = 0.
    S::Int = 0

    for e in targets
        res = ( abs(y[e]) - abs(y0[e]) ) / abs(y0[e]) - threshold
        if res < -1e-5
            H += -res
        else
            S += 1
        end
    end

    return H, S
end


"""Gradient of the upper objective H (Hinge loss) w.r.t. marginal flow on target edge e,
i.e., ∂H/∂yopt[e]."""
function cal_gradient_DHDy_on_target(e::Int, ye::Float64, ye0::Float64,
            targets::Array{Int, 1}, threshold::Float64)
    if !(e in targets)
        ∂H_∂y = 0.
    else
        res = ( abs(ye) - abs(ye0) ) / abs(ye0) - threshold
        if res > 0
            ∂H_∂y = 0.
        else
            ∂H_∂y = - sign(ye) / abs(ye0)
        end
    end

    return ∂H_∂y
end

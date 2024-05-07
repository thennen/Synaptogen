module Synaptogen

export applyVoltage!, Iread, Uread, HRS, LRS, UR, US, I, ADC,
       CellArray, CellArrayCPU, CellArrayGPU, Cell,
       CellParams, CellParams_from_artifact, StaticCellParams, defaultParams, defaultStaticParams


"""
    polyval(coeffs::AbstractVector, U::Real)

Evaluate a polynomial defined by coeffs at U (Horner's algorithm)
This is the only polyval needed for the unvectorized code
Using on CuArrays results in scalar indexing
"""
function polyval(coeffs::AbstractVector, U::Real)
    acc = zero(U)
    for c in coeffs
        acc = acc * U + c
    end
    acc
end


"""
    polyval(coeffs::AbstractMatrix, U::AbstractArray)

Evaluate a set of polynomials at a set of points by running Horner's algorithm in parallel
(polynomials are in the rows of coeffs: npolys × degree)
(points are in the columns of U: npolys × npoints)
"""
function polyval(coeffs::AbstractMatrix, U::AbstractArray)
    acc = zero(U)
    for c in eachslice(coeffs, dims=2)
        acc .= acc .* U .+ c
    end
    return acc
end


# const σClip = Float32(3.5)  # not clipping anymore
const Uread = Float32(0.2)    # Default voltage to perform readouts
const e = Float32(1.602176634e-19)
const kBT = Float32(1.380649e-23 * 300)
const iHRS, iUS, iLRS, iUR = 1, 2, 3, 4


"""
    r(R, G_HHRS, G_LLRS)

r such that (1-r)⋅LRSpoly + r⋅HHRSpoly has static resistance R = U₀/I(U₀)
G_HHRS and G_LLRS are the static conductances of the HHRS and LLRS at U₀
"""
function r(R, G_HHRS, G_LLRS)
    (G_LLRS - 1/R) / (G_LLRS - G_HHRS)
end


"""
Transform data from standard normal to the (standardized) empirical distributions (Γ^-1 in paper)
It's just a polynomial evaluation as already defined by polyval
"""
const Γ = polyval


"""
    Ireset(a::Real, c::Real, U::Real, η::Real, Umax::Real)

RESET transition polynomial that has steepness parameterized by η
The standard form will not be easier to compute by Horner in this case, so we use a factored form
if b = Umax, dI/dV = 0 at Umax
"""
Ireset(a::Real, c::Real, U::Real, η::Real, Umax::Real) = a * abs(Umax - U)^η + c

# Parallelized version that runs on CPU and GPU
# Also has the code that loads the parameter json.
include("vectorized.jl")

# Runs on CPU only, can do sparse operations, can use multithreading
include("unvectorized.jl")

end

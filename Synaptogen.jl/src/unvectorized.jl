#=
This is the "unvectorized" version of Synaptogen

There's one struct (Cell) per simulated memory cell

Benefits:
⋅ Runs faster on CPU than the vectorized code (CellArrayCPU)
⋅ Cells are individually addressable -- better for sparse operations
⋅ Code simpler to understand

Drawbacks:
⋅ Can't run on GPU

=#

using StaticArrays: SVector, SMatrix, MVector, MMatrix
using Random # rand, randn
using Printf


"""
Parameter structure using StaticArrays.
# Type parameters
- O : Number of features
- P : VAR order
- L : Degree of LLRS polynomial
- H : Degree of HHRS polynomial
- G : Degree of quantile transform
- K : Number of GMM components
"Redundant" type parameters:
- O2  : O²
- OO  : 2O
- OO2 : (2O)²
"""
struct StaticCellParams{O, P, L, H, G, K, O2, OO, OO2}
    Umax::Float32
    U₀::Float32
    η::Float32
    G_HHRS::Float32
    G_LLRS::Float32
    HHRS::SVector{H, Float32}
    LLRS::SVector{L, Float32}
    γ::SVector{O, SVector{G, Float32}}
    wk::SVector{K, Float32}
    μDtD::SVector{K, SVector{OO, Float32}}
    LDtD::SVector{K, SMatrix{OO, OO, Float32, OO2}}
    VAR_L::SMatrix{O, O, Float32, O2}
    VAR_Ai::SVector{P, SMatrix{O, O, Float32, O2}}
end


"""
    get_type_params(p::CellParams)

This just calculates the StaticCellParams type parameters from whatever is inside a CellParams
"""
function get_type_params(p::CellParams)
    O, G = size(p.γ)
    P = size(p.VAR)[2] ÷ O - 1
    L = size(p.LLRS)[2]
    H = size(p.HHRS)[2]
    K = size(p.wk)[1]
    O2 = O^2
    OO = 2O
    OO2 = OO^2
    return O, P, L, H, G, K, O2, OO, OO2
end


function Base.convert(::Type{StaticCellParams{O, P, L, H, G, K, O2, OO, OO2}}, params::CellParams) where {O, P, L, H, G, K, O2, OO, OO2}
    (;Umax, U₀, η, G_HHRS, G_LLRS, HHRS, LLRS, γ, wk, μDtD, LDtD, VAR) = params
    γ = SVector{O, SVector{G, Float32}}(eachrow(γ))
    μDtD = SVector{K, SVector{OO, Float32}}(eachcol(μDtD))
    LDtD = SVector{K, SMatrix{OO, OO, Float32}}(eachslice(LDtD; dims=3))
    VAR_L = SMatrix{O, O, Float32, O2}(VAR[:, 1:O])
    VAR_Ai = SVector{P, SMatrix{O, O, Float32, O2}}(VAR[:, O*p+1:O*p+O] for p in 1:P)
    return StaticCellParams{O, P, L, H, G, K, O2, OO, OO2}(Umax, U₀, η, G_HHRS, G_LLRS, HHRS, LLRS,
                                                           γ, wk, μDtD, LDtD, VAR_L, VAR_Ai)
end

StaticCellParams(params::CellParams) = Base.convert(StaticCellParams{get_type_params(params)...}, params)
const defaultStaticParams = StaticCellParams(defaultParams)

"""
Structure holding the state of an individual memory cell
"""
mutable struct Cell{O, P, L, H, G, K, O2, OO, OO2}
    const Xhat::MVector{P, SVector{O, Float32}} # constantly an MVector ...
    y::SVector{O, Float32}
    const σ::SVector{O, Float32}
    const μ::SVector{O, Float32}
    const k::Int8 # Doesn't strictly need to be stored.
    resetCoefs::SVector{2, Float32}
    r::Float32
    n::UInt32
    UR::Float32
    inHRS::Bool
    inLRS::Bool
    const params::StaticCellParams{O, P, L, H, G, K, O2, OO, OO2}
    @doc"""
        Cell(params::StaticCellParams=defaultStaticParams)

    Initialize a Cell with the given parameters in HRS
    """
    function Cell(params::StaticCellParams{O, P, L, H, G, K, O2, OO, OO2}=defaultStaticParams) where {O, P, L, H, G, K, O2, OO, OO2}
        (;μDtD, LDtD, VAR_L, γ, wk, G_LLRS, G_HHRS) = params
        xhat = VAR_L * randn(SVector{O, Float32})
        Xhat = MVector{P, SVector{O, Float32}}(xhat, (zeros(SVector{O, Float32}) for n in 2:P)...)
        k = findfirst(cumsum(wk) .> rand(Float32) * sum(wk))
        μσ = μDtD[k] .+ LDtD[k] * randn(SVector{OO, Float32})
        μ = SVector{O, Float32}(μσ[1:O])
        σ = SVector{O, Float32}(μσ[O+1:OO])
        y = Ψ(μ, σ, Γ.(γ, xhat))
        r0 = r(10^y[iHRS], G_HHRS, G_LLRS)
        resetCoefs = zeros(SVector{2, Float32})
        c = new{O, P, L, H, G, K, O2, OO, OO2}(Xhat, y, σ, μ, k, resetCoefs, r0, 0, 0, true, false, params)
        return c
    end
end

"""
    Cell(params::CellParams)

Convert CellParams to StaticCellParams and initialize a Cell with those parameters
"""
Cell(params::CellParams) = Cell(StaticCellParams(params))


p(c::Cell{O, P}) where {O, P} = P


function Base.show(io::IO, c::Cell)
    t = typeof(c)
    truncparams = join(t.parameters[1:end-3], ",")
    r = @sprintf("%.3f", c.r)
    if c.inHRS
        state = "HRS"
    elseif c.inLRS
        state = "LRS"
    else
        state = "IRS"
    end
    print(io, "$(t.name.name){$(truncparams)}(r=$(r)($(state)),n=$(c.n))")
end


function Base.show(io::IO, c::StaticCellParams)
    t = typeof(c)
    truncparams = join(t.parameters[1:end-3], ",")
    print(io, "$(t.name.name){$(truncparams)}")
end


"""
    Ψ(μ, σ, x)

Scale and translate x by the device-specific vectors μ and σ
"""
function Ψ(μ, σ, x)
    #TODO: couldn't I just share the definition in vectorized.jl?
    y = μ .+ σ .* x
    return y # No allocations, but y[1] is not stored base-10 exponentiated
    #return [10^y[1]; y[2:end]] # assumes iHRS is 1, and big performance problem because it allocates
    #return (10^y[1], y[2:end]...) # Worse
    # return (μ[1]*σ[1]^x[1], (μ[2:end] .+ σ[2:end] .* x[2:end])...) # Worse
    #[i == 1 ? 10^y[i] : y[i] for i in eachindex(y)] # slightly better.. still allocates.
    # Another option might be to just do c.y[1] = 10^c.y[1] after you do c.y = Ψ(μ, σ, x) in applyVoltage.
end


"""
    VAR_sample(c<:Cell)

Generate and return the next VAR vector
"""
function VAR_sample(c::Cell{O, P}) where {O, P}
    x = c.params.VAR_L * randn(SVector{O, Float32}) # + VAR_intercept ≈ 0
    for i in 1:P
        j = mod(c.n + 1 - i, P) + 1
        @inbounds x += c.params.VAR_Ai[i] * c.Xhat[j]
    end
    return x
    #return clamp.(x, -σClip, σClip)
end


HRS(c::Cell) = 10^c.y[iHRS] # computation can potentially be repeated, but this is faster than storing 
LRS(c::Cell) = c.y[iLRS]
US(c::Cell) = c.y[iUS]
UR(c::Cell) = c.y[iUR]


"""
    r(I::Float32, V::Float32, HHRS::AbstractVector{Float32}, LLRS::AbstractVector{Float32})

Return r such that (1-r) ⋅ LRSpoly + r ⋅ HHRSpoly intersects I,V
used for switching along transition curves
"""
function r(I::Float32, V::Float32, HHRS::AbstractVector{Float32}, LLRS::AbstractVector{Float32})
    IHHRS_V = polyval(HHRS, V)
    ILLRS_V = polyval(LLRS, V)
    (I - ILLRS_V) / (IHHRS_V - ILLRS_V)
end


"""
    mixed_poly(r, HHRS, LLRS)

Return a linear mixture of polynomial coefficients
works in the case that HHRS and LLRS have different degrees
"""
function mixed_poly(r, HHRS, LLRS)
    lh = length(HHRS)
    ll = length(LLRS)
    diff = abs(lh - ll)
    h = r*HHRS
    l = (1-r)*LLRS
    short, long = lh < ll ? (h, l) : (l, h)
    return long + vcat(zeros(SVector{diff, Float32}), short)
end


# One polyval on mixed coefficients.  For this, HHRS & LLRS must have the same degree..
#I(r::Real, U::Real, HHRS::AbstractVector, LLRS::AbstractVector) = polyval((1-r) .* LLRS + r .* HHRS, U) # 
# This one works even if HHRS and LLRS don't have the same degree. It's a bit faster if they do.
"Current as a function of voltage for the current cell state (r)"
I(r::Real, U::Real, HHRS::AbstractVector, LLRS::AbstractVector) = polyval(mixed_poly(r, HHRS, LLRS), U)
# This has to do polyval twice, but doesn't need to calculate and store the mixed poly coefficients.
#I(r::Real, U::Real, HHRS::AbstractVector, LLRS::AbstractVector) = (1-r) * polyval(LLRS, U) + r * polyval(HHRS, U)

I(c::Cell, U::Real) = I(c.r, U, c.params.HHRS, c.params.LLRS)

IHRS(c::Cell, U::Real) = I(r(HRS(c), c.params.G_HHRS, c.params.G_LLRS), U, c.params.HHRS, c.params.LLRS)
ILRS(c::Cell, U::Real) = I(r(LRS(c), c.params.G_HHRS, c.params.G_LLRS), U, c.params.HHRS, c.params.LLRS) # not used


"""
    resetCoefs(x₁, x₂, y₁, y₂, η)

Coefficients for a RESET transition polynomial of the form y = a(b-x)^η + c
which connects (x₁,y₁) to (x₂,y₂) with dy/dx = 0 at x₂
"""
function resetCoefs(x₁, x₂, y₁, y₂, η)
    a = (y₁-y₂) / (x₂-x₁)^η
    c = y₂
    return [a, c]
end


"""
    applyVoltage!(c<:Cell, Ua::Real)

Apply voltage U to a Cell
if U > UR or if U ≤ US, it may modify c state
c is also returned
"""
function applyVoltage!(c::Cell{O, P}, Ua::Real) where {O, P}
    (;Umax, G_HHRS, G_LLRS, HHRS, LLRS, η, γ)  = c.params
    Ua = Float32(Ua)
    if !c.inLRS
        if Ua < US(c)
            # SET
            c.r = r(LRS(c), G_HHRS, G_LLRS)
            c.inLRS = true
            c.inHRS = false
            c.UR = UR(c)
            return c
        elseif c.inHRS
            return c
        end
    end

    # By now we know we are not in HRS, so we might reset
    if Ua > c.UR
        fullreset = Ua ≥ Umax

        if c.inLRS # First reset
            c.inLRS = false
            # Calculate and store params for next cycle
            xhat = VAR_sample(c)
            c.n += 1
            @inbounds c.Xhat[mod(c.n, P) + 1] = xhat
            x = Γ.(γ, xhat)
            if fullreset
                c.y = Ψ(c.μ, c.σ, x)
            else
                # We will need the updated transition coefs
                x₁ = UR(c)
                x₂ = Umax
                y₁ = I(c, x₁)
                c.y = Ψ(c.μ, c.σ, x)
                y₂ = IHRS(c, x₂)
                c.resetCoefs = resetCoefs(x₁,x₂,y₁,y₂,η)
            end
        end

        if fullreset
            # Full RESET
            c.r = r(HRS(c), G_HHRS, G_LLRS)
            c.inHRS = true
            c.UR = Umax
        else
            # Partial RESET
            @inbounds Itrans =  Ireset(c.resetCoefs[1], c.resetCoefs[2], Ua, η, Umax)
            c.r = r(Itrans, Ua, HHRS, LLRS)
            c.UR = Ua
        end
    end
    return c
end

"""
    Iread(c::Cell, U::Real=Uread; nbits::Int=4, Imin::Float=1f-6, Imax::Float=1f-5, BW::Float=1f8)

Simulated current measurement of a Cell including noise and ADC
"""
function Iread(c::Cell, U::Real=Uread; nbits::Int=4, Imin::AbstractFloat=1f-6, Imax::AbstractFloat=5f-5, BW::AbstractFloat=1f8)
    U = convert(Float32, U)
    Imin = convert(Float32, Imin)
    Imax = convert(Float32, Imax)
    BW = convert(Float32, BW)
    Inoiseless = I(c, U)
    # Approximation of the thermodynamic noise
    johnson = abs(4*kBT*BW*Inoiseless/U)
    shot = abs(2*e*Inoiseless*BW)
    # Digitization noise
    Irange = Imax - Imin
    nlevels = 2^nbits
    q = Irange / nlevels
    #ADC = q^2 / 12
    # Sample from total noise distribution
    #σ_total = √(johnson + shot + ADC)
    σ_total = √(johnson + shot)
    Iwithnoise = Inoiseless + randn(Float32) * σ_total
    # Return nearest quantization level?
    return clamp(round((Iwithnoise - Imin) / q), 0, nlevels) * q + Imin
end
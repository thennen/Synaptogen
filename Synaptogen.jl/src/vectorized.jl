#=
This is the "vectorized" version of Synaptogen

All cell states are contained in a single struct and matrix operations are used throughout

Benefits:
⋅ The same code runs on both CPU (using CellArrayCPU) and GPU (using CellArrayGPU)
⋅ Can be translated to slower languages like Python or Matlab

Drawbacks:
⋅ All cells must be addressed at once for read/write operations
⋅ Code is a bit harder to understand

=#

using Random # randn!
using LinearAlgebra # mul!
using CUDA
using LazyArtifacts
import JSON3
import Base: *

########## Define our CPU and GPU types

# Actually more like a CellVector. Not designed to be 2D (yet..)
abstract type CellArray end


"Structure that holds all the parameters of the model (CPU version)"
struct CellParams
    Umax::Float32           # Maximum voltage applied during the experiment.  defines the point where HRS is reached.
    U₀::Float32             # Voltage used in the definition of resistance R = U₀ / I(U₀)
    η::Float32              # Sets the curvature of the reset transition
    nfeatures::Int64        # Number of features for the VAR model
    p::Int64                # Order of the VAR model (how many cycles of history remembered)
    K::Int64                # How many components in the GMM for modeling device-to-device distribution
    γdeg::Int64             # Degree of the non-linear transformation polynomials 
    G_HHRS::Float32         # Conductance of the HHRS 
    G_LLRS::Float32         # Conductance of the LLRS
    HHRSdeg::Int64          # Degree of the HHRS polynomial
    LLRSdeg::Int64          # Degree of the LLRS polynomial
    HHRS::Matrix{Float32}   # HHRS coefficients.  Not a vector because of polyval shenanigans
    LLRS::Matrix{Float32}   # LLRS coefficients
    γ::Matrix{Float32}      # non-linear transformation coefficients
    wk::Vector{Float32}     # weights of the GMM components
    μDtD::Matrix{Float32}   # mean vectors for the GMM
    LDtD::Array{Float32, 3} # Cholesky decomposition of covariance matrices for the GMM (lower triangular)
    VAR::Matrix{Float32}    # VAR coefficients, including A and B
end


"Just CellParams with Array → CuArray.  Should be able to auto-generate this..."
struct CellParamsGPU
    Umax::Float32
    U₀::Float32
    η::Float32
    nfeatures::Int64
    p::Int64
    K::Int64
    γdeg::Int64
    G_HHRS::Float32
    G_LLRS::Float32
    HHRSdeg::Int64
    LLRSdeg::Int64
    HHRS::CuMatrix{Float32}
    LLRS::CuMatrix{Float32}
    γ::CuMatrix{Float32}
    wk::CuVector{Float32}      # Would it make sense to leave these tiny vectors on CPU?
    μDtD::CuMatrix{Float32}
    LDtD::CuArray{Float32, 3}
    VAR::CuMatrix{Float32}
end


# A small default parameter file in the package directory
const default_param_fp = joinpath(@__DIR__, "default_params.json")


"""
    CellParams(fp::String=default_param_fp; p::Int=10)

Loads model parameters from disk (json)
"""
function CellParams(fp::String=default_param_fp; p::Int=10)
    json_string = read(fp, String)
    json_params = JSON3.read(json_string)
    """
    Construct stacked matrix for the VAR process for model order p
    Parameter file only contains fits for a certain set of p
    the p you ask for should be less than the largest p in the parameter file
    if you choose a p not in the data file, you will get a truncated approximation
    """
    # Matrices get loaded as arrays of arrays. Have to concatenate them.
    matrix(arr_of_arr) = reduce(hcat, arr_of_arr)'
    γ = Matrix{Float32}(matrix(json_params.γ))
    nfeatures, γdeg = size(γ)
    function get_VAR_matrix(p::Int64=20)
        key_strings = [String(k) for k in keys(json_params)]
        available_orders = sort([parse(Int, split(k, "_")[2]) for k in key_strings if startswith(k, "VAR")])
        q = available_orders[findfirst(available_orders .>= p)]
        VAR_json_params = reduce(hcat, json_params["VAR_" * lpad(q, 3, "0")])'
        VARcoefs = Matrix{Float32}(VAR_json_params[:, 1:(p+1)*nfeatures])
        return VARcoefs
    end
    Umax = Float32(json_params.Umax)     # The highest applied voltage in RESET direction during the experiment.
    U₀ = Float32(json_params.U₀)         # Part of the definition of "resistance"
    η = Float32(json_params.η)
    HHRS = Vector{Float32}(json_params.HHRS)
    LLRS = Vector{Float32}(json_params.LLRS)
    # derived parameters
    G_HHRS = polyval(HHRS, U₀) / U₀
    G_LLRS = polyval(LLRS, U₀) / U₀
    HHRS = reshape(HHRS, 1, length(HHRS)) # this matrix form makes the polyval work on GPU without scalar indexing
    LLRS = reshape(LLRS, 1, length(LLRS))
    HHRSdeg = size(json_params.HHRS)[1]
    LLRSdeg = size(json_params.LLRS)[1]
    wk = Vector{Float32}(json_params.wk)
    K = size(wk)[1]
    # Turn these into 3d arrays because CuVector{CuVector} isn't a thing
    LDtD = reshape(matrix(json_params.LDtD), 2*nfeatures, 2*nfeatures, K)
    μDtD = matrix(json_params.μDtD)
    VAR = get_VAR_matrix(p)
    return CellParams(Umax, U₀, η, nfeatures, p, K, γdeg, G_HHRS, G_LLRS, HHRSdeg, LLRSdeg, HHRS, LLRS, γ, wk, μDtD, LDtD, VAR)
end

const defaultParams = CellParams()

"""
    CellParams_from_artifact(name::String="553da9", artifact::String="alpha"; p::Int=20)

Get CellParams from Julia artifact, pass the json filename
"""
function CellParams_from_artifact(name::String="553da9", artifact::String="alpha"; p::Int=20)
    root = @artifact_str artifact
    fn = endswith(name, ".json") ? name : name * ".json"
    return CellParams(joinpath(root, fn), p=p)
end


function Base.convert(::Type{CellParamsGPU}, params::CellParams)
    CellParamsGPU([cu(getfield(params, f)) for f in fieldnames(CellParamsGPU)]...)
end

# CuArrays don't seem to work in global scope. Throws UndefRefError: access to undefined reference
# const defaultParamsGPU = Base.convert(CellParamsGPU, defaultParams)

struct CellArrayCPU <: CellArray
    M::Int64                      # scalar      (number of cells)
    Xhat::Array{Float32, 2}       # 4(p+1) × M  (feature history and εₙ for all cells)
    Xbuf::Array{Float32, 2}       # 4(p+1) × M  (buffer to improve the speed of the partial shift operation)
    x::Array{Float32, 2}          # 4 × M       (generated normal feature vectors ̂x*ₙ, basically also a buffer)
    σ::Array{Float32, 2}          # 4 × M       (CtC scale vectors)
    μ::Array{Float32, 2}          # 4 × M       (CtC offset vectors)
    y::Array{Float32, 2}          # 4 × M       (scaled feature vector)
    r::Array{Float32, 1}          # M × 1       (device state variables)
    n::Array{UInt32, 1}           # M × 1       (cycle numbers)
    k::Array{Int32, 1}            # M × 1       (GMM component, not strictly necessary to store)
    UR::Array{Float32, 1}         # M × 1       (voltage thresholds for reset switching)
    Umax::Array{Float32, 1}       # M × 1       (Vector of Umax, probably all the same value, just for vectorization of polyval)
    resetCoefs::Array{Float32, 2} # M × 2       (polynomial coefficients for reset transitions)
    Iread::Array{Float32, 1}      # M × 1       (readout buffer)
    inHRS::Array{Bool, 1}         # Using BitVector does not save much memory and isn't faster either.
    inLRS::Array{Bool, 1}
    setMask::Array{Bool, 1}
    resetMask::Array{Bool, 1}
    fullResetMask::Array{Bool, 1}
    partialResetMask::Array{Bool, 1}
    resetCoefsCalcMask::Array{Bool, 1}
    drawVARMask::Array{Bool, 1}
    params::CellParams
end


# Just CellArrayCPU with CuArrays
# Should be able to auto generate this..?
struct CellArrayGPU <: CellArray
    M::Int64
    Xhat::CuArray{Float32, 2}
    Xbuf::CuArray{Float32, 2}
    x::CuArray{Float32, 2}
    σ::CuArray{Float32, 2}
    μ::CuArray{Float32, 2}
    y::CuArray{Float32, 2}
    r::CuArray{Float32, 1}
    n::CuArray{UInt32, 1}
    k::CuArray{Int32, 1}
    UR::CuArray{Float32, 1}
    Umax::CuArray{Float32, 1}
    resetCoefs::CuArray{Float32, 2}
    Iread::CuArray{Float32, 1}
    inHRS::CuArray{Bool, 1}
    inLRS::CuArray{Bool, 1}
    setMask::CuArray{Bool, 1}
    resetMask::CuArray{Bool, 1}
    fullResetMask::CuArray{Bool, 1}
    partialResetMask::CuArray{Bool, 1}
    resetCoefsCalcMask::CuArray{Bool, 1}
    drawVARMask::CuArray{Bool, 1}
    params::CellParamsGPU
end


Base.length(c::CellArray) = c.M


function Base.show(io::IO, c::CellArray)
    print(io)
    t = typeof(c)
    #r = @sprintf("%.3f", c.r)
    print(io, "$(c.M)-element $(t.name.name)\n")
    print(io, c.r)
end


"""
    CellArrayCPU(M::Int=2^16, params::CellParams=defaultParams)

 Constructor for CellArrayCPU. Initializes M cells in HRS. 
"""
function CellArrayCPU(M::Int=2^16, params::CellParams=defaultParams)
    @inbounds begin
        (;VAR, nfeatures, p, γ, wk, μDtD, LDtD, G_HHRS, G_LLRS, Umax) = params
        Xhat = zeros(Float32, nfeatures*(p+1), M)
        Xbuf = similar(Xhat)
        randn!(view(Xhat, 1:nfeatures, :))
        x = VAR * Xhat # NOICE
        Xhat[end-nfeatures+1:end, :] .= x
        cs = cumsum(wk) / sum(wk)
        k = map(x -> searchsortedfirst(cs, x), rand(Float32, M)) # Int64...
        μσCtC = Array{Float32, 2}(undef, nfeatures * 2, M)
        mask = BitVector(undef, M)
        for kk in 1:length(wk)
            @. mask = kk == k
            Mk = sum(mask)
            @views μσCtC[:,mask] .= μDtD[:,kk] .+ LDtD[:,:,kk] * randn(Float32, nfeatures * 2, Mk)
        end
        μCtC = μσCtC[1:nfeatures, :]
        σCtC = μσCtC[nfeatures+1:nfeatures*2, :]
        y = Ψ(μCtC, σCtC, Γ(γ, x))
        resetCoefs = Array{Float32, 2}(undef, M, 2)
        r0 = r.(y[iHRS, :], G_HHRS, G_LLRS)
        n = zeros(UInt32, M)
        UR = y[iUR, :]
        Umax = fill(Umax, M)
        Iread = zeros(Float32, M)
        inHRS = ones(Bool, M)
        inLRS = zeros(Bool, M)
        setMask = Array{Bool, 1}(undef, M)
        resetMask = Array{Bool, 1}(undef, M)
        fullResetMask = Array{Bool, 1}(undef, M)
        partialResetMask = Array{Bool, 1}(undef, M)
        resetCoefsCalcMask = Array{Bool, 1}(undef, M)
        drawVARMask = Array{Bool, 1}(undef, M)

        return CellArrayCPU(M, Xhat, Xbuf, x, σCtC, μCtC, y, r0, n, k, UR, Umax, resetCoefs, Iread,
                            inHRS, inLRS, setMask, resetMask, fullResetMask, partialResetMask,
                            resetCoefsCalcMask, drawVARMask, params)
    end
end


"""
    CellArrayGPU(M::Int, params::CellParamsGPU)

Constructor for CellArrayGPU. Initializes M cells in HRS. 
"""
function CellArrayGPU(M::Int, params::CellParamsGPU)
    # largely repeated code from CellArrayCPU unfortunately
    @inbounds begin
        (;VAR, nfeatures, p, γ, wk, μDtD, LDtD, G_HHRS, G_LLRS, Umax) = params
        Xhat = CUDA.zeros(Float32, nfeatures*(p+1), M)
        Xbuf = similar(Xhat)
        randn!(view(Xhat, 1:nfeatures, :))
        x = VAR * Xhat
        Xhat[end-nfeatures+1:end, :] .= x
        cs = cumsum(wk) / sum(wk)
        k = searchsortedfirst.(Ref(cs), CUDA.rand(Float32, M))
        μσCtC = CuArray{Float32, 2}(undef, nfeatures * 2, M)
        mask = CuArray{Bool, 1}(undef, M)
        for kk in 1:length(wk)
            @. mask = kk == k
            Mk = sum(mask)
            @views μσCtC[:,mask] .= μDtD[:,kk] .+ LDtD[:,:,kk] * CUDA.randn(Float32, nfeatures * 2, Mk)
        end
        μCtC = μσCtC[1:nfeatures, :]
        σCtC = μσCtC[nfeatures+1:nfeatures*2, :]
        y = Ψ(μCtC, σCtC, Γ(γ, x))
        resetCoefs = CuArray{Float32, 2}(undef, M, 2)
        r0 = r.(y[iHRS, :], G_HHRS, G_LLRS)
        n = CUDA.zeros(UInt32, M)
        UR = y[iUR, :]
        Umax = CUDA.fill(Umax, M)
        Iread = CUDA.zeros(Float32, M)
        inHRS = CUDA.ones(Bool, M)
        inLRS = CUDA.zeros(Bool, M)
        setMask = CuArray{Bool, 1}(undef, M)
        resetMask = CuArray{Bool, 1}(undef, M)
        fullResetMask = CuArray{Bool, 1}(undef, M)
        partialResetMask = CuArray{Bool, 1}(undef, M)
        resetCoefsCalcMask = CuArray{Bool, 1}(undef, M)
        drawVARMask = CuArray{Bool, 1}(undef, M)

        return CellArrayGPU(M, Xhat, Xbuf, x, σCtC, μCtC, y, r0, n, k, UR, Umax, resetCoefs, Iread,
                            inHRS, inLRS, setMask, resetMask, fullResetMask, partialResetMask,
                            resetCoefsCalcMask, drawVARMask, params)
    end
end

"""
    CellArrayGPU(M::Int, params::CellParams=defaultParams)

Moves params to GPU and then constructs a CellArrayGPU with those parameters
"""
function CellArrayGPU(M::Int, params::CellParams=defaultParams)
    paramsGPU = Base.convert(CellParamsGPU, params)
    CellArrayGPU(M, paramsGPU)
end


#=
"""
Constructor for CellArrayGPU
this just constructs a CellArrayCPU then moves it to GPU.
"""
function CellArrayGPU(M::Int64=2^16, params::CellParams=defaultParams)
    cellsCPU = CellArrayCPU(M, params)
    # kind of a fragile conversion..
    return CellArrayGPU([getfield(cellsCPU, f) for f in fieldnames(CellArrayGPU)]...)
end
=#


HRS(c::CellArray) = @inbounds view(c.y, iHRS, :)
LRS(c::CellArray) = @inbounds view(c.y, iLRS, :)
US(c::CellArray) = @inbounds view(c.y, iUS, :)
UR(c::CellArray) = @inbounds view(c.y, iUR, :)

"""
    Ψ(μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})

Scale and translate x by the device-specific vectors
"""
function Ψ(μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, x::AbstractMatrix{Float32})
    y = μ .+ σ .* x
    y[iHRS, :] .= 10 .^ y[iHRS, :]
    return y
end


"""
    VAR_sample!(c::CellArray)

Draw the next VAR terms, updating the history matrix (c.Xhat)
involves a shift operation for the subset of columns corresponding to drawVARMask == true
"""
function VAR_sample!(c::CellArray)
    (;nfeatures, VAR) = c.params
    @inbounds begin
        # all random numbers are overwritten for the sake of using randn!, which is much faster than randn
        randn!(view(c.Xhat, 1:nfeatures, :))
        mul!(c.x, VAR, c.Xhat) # caution: x is not accurate anymore where drawVARMask == false
        copy!(c.Xbuf, c.Xhat) # this is so we can use ifelse instead of working on non-contiguous slices, which is very slow
        c.Xhat[nfeatures+1:end-nfeatures, :] .= ifelse.(c.drawVARMask', @view(c.Xbuf[2*nfeatures+1:end, :]), @view(c.Xbuf[nfeatures+1:end-nfeatures, :]))
        c.Xhat[end+1-nfeatures:end, :] .= ifelse.(c.drawVARMask', c.x, @view(c.Xhat[end + 1 - nfeatures:end,:]))
    end
end


"""
    r(I::AbstractVector{Float32}, U::AbstractVector{Float32}, HHRS::AbstractMatrix{Float32}, LLRS::AbstractMatrix{Float32})

Return r such that (1-r) ⋅ LLRS + r ⋅ HHRS intersects I,U.
Used for switching along transition curves.
"""
function r(I::AbstractVector{Float32}, U::AbstractVector{Float32}, HHRS::AbstractMatrix{Float32}, LLRS::AbstractMatrix{Float32})
    IHHRS_U = polyval(HHRS, U)
    ILLRS_U = polyval(LLRS, U)
    return @. (I - ILLRS_U) / (IHHRS_U - ILLRS_U)
end


"""
    I(r::AbstractVector, U::AbstractVector, HHRS::AbstractArray, LLRS::AbstractArray)

Current as a function of voltage for the cell state
if U is scalar and rest are GPU arrays, this results in scalar indexing..
better to fill(c.M, U)
"""
function I(r::AbstractVector, U::AbstractVector, HHRS::AbstractArray, LLRS::AbstractArray)
    (1 .- r) .* polyval(LLRS, U) .+ r .* polyval(HHRS, U)
    # TODO: Could this be a better approach?
    # polyval(1-r * LLRSpoly + r * HHRSpoly, U)
    # one problem is that the degrees of the polynomials may be different
    # if it is faster, then we could just redefine LLRS and HHRS to have the same length
    # HOWEVER, this will take more memory because you have to store all the mixed polynomial coefficients, rather than just LLRS and HHRS
    # another option is to evaluate a vcat of LLRS and HHRS at repeat(U, inner=2), then do (1 - r) I[1:2:end] + r I[2:2:end]
    # (replace slicing with something that works better on GPU)
    # this is all highly redundant if U are all the same but I don't think the GPU really cares
end


I(c::CellArray, U) = I(c.r, U, c.params.HHRS, c.params.LLRS)


"""
    resetCoefs(x₁::AbstractVector{Float32}, x₂, y₁::AbstractVector{Float32}, y₂::AbstractVector{Float32}, η::Float32)

Coefficients for a transition polynomial of the form y = a(b-x)^η + c
which connects (x₁,y₁) to (x₂,y₂) with dy/dx = 0 at x₂
"""
function resetCoefs(x₁::AbstractVector{Float32}, x₂, y₁::AbstractVector{Float32}, y₂::AbstractVector{Float32}, η::Float32)
    # Often x₂ is just a constant (= Umax) but let's leave the possibility of it being an array
    # abs is necessary because in this vectorized version we end up calculating resetCoefs for all cells at once, even those which have seen UR > Umax
    a = @. (y₁-y₂) / abs(x₂-x₁)^η
    c = y₂
    return [a c] # M × 2 matrix
    # TODO: Can I somehow use the same resetCoefs() from unvectorized.jl?
    #       broadcasting gives a vector (M) of vectors (2)
end


"""
    applyVoltage!(c::CellArray, Ua::AbstractVector{<:Real})

Apply voltages from array U to the corresponding cell in the CellArray
if U > UR or if U ≤ US, cell states will be modified
"""
function applyVoltage!(c::CellArray, Ua::AbstractVector{<:Real})
    (;Umax, G_HHRS, G_LLRS, HHRS, LLRS, η, γ, nfeatures)  = c.params

    # this converts the elements to Float32 while keeping the vector type the same
    Ua = Base.convert(AbstractVector{Float32}, Ua)

    ### Create boolean masks for the different operations that may need to be applied
    c.setMask .= .!c.inLRS .& (Ua .≤ US(c)) # not @. because US() doesn't broadcast
    @. c.resetMask = !c.inHRS & (Ua > c.UR)
    @. c.fullResetMask = c.resetMask & (Ua ≥ Umax)
    @. c.partialResetMask = c.resetMask & (Ua < Umax)
    @. c.drawVARMask = c.inLRS & c.resetMask
    @. c.resetCoefsCalcMask = c.drawVARMask & !c.fullResetMask

    if any(c.setMask)
        c.r .= ifelse.(c.setMask, r.(LRS(c), G_HHRS, G_LLRS), c.r)
        c.inLRS .|= c.setMask
        c.inHRS .= c.inHRS .& .!c.setMask
        c.UR .= ifelse.(c.setMask, UR(c), c.UR)
    end

    if any(c.drawVARMask)
        VAR_sample!(c)
        c.n .+= c.drawVARMask
        @inbounds c.y .= Ψ(c.μ, c.σ, Γ(γ, c.Xhat[end-nfeatures+1:end, :]))
    end

    if any(c.resetCoefsCalcMask)
        x₁ = c.UR
        x₂ = c.Umax # Array version just to keep everything vectorized
        y₁ = I(c.r, x₁, HHRS, LLRS)
        r_HRS = r.(HRS(c), G_HHRS, G_LLRS)
        y₂ = I(r_HRS, x₂, HHRS, LLRS)
        c.resetCoefs .= ifelse.(c.resetCoefsCalcMask, resetCoefs(x₁, x₂, y₁, y₂, η), c.resetCoefs)
    end

    if any(c.resetMask)
        c.inLRS .= c.inLRS .& .!c.resetMask
        c.UR .= ifelse.(c.resetMask, Ua, c.UR)
    end

    if any(c.partialResetMask)
        @inbounds Itrans = Ireset.(c.resetCoefs[:,1], c.resetCoefs[:,2], Ua, η, Umax)
        c.r .= ifelse.(c.partialResetMask, r(Itrans, Ua, HHRS, LLRS), c.r)
    end

    if any(c.fullResetMask)
        c.inHRS .|= c.fullResetMask
        c.r .= ifelse.(c.fullResetMask, r.(HRS(c), G_HHRS, G_LLRS), c.r)
    end

    return c
end


"""
    applyVoltage!(c::CellArrayCPU, Ua::Real)

Apply the voltage Ua to all cells in the CellArrayCPU
"""
applyVoltage!(c::CellArrayCPU, Ua::Real) = applyVoltage!(c, fill(Float32(Ua), c.M))

"""
    applyVoltage!(c::CellArrayGPU, Ua::Real)

Apply the voltage Ua to all cells in the CellArrayGPU
"""
applyVoltage!(c::CellArrayGPU, Ua::Real) = applyVoltage!(c, CUDA.fill(Float32(Ua), c.M))


"""
    Iread(c::CellArray, U::AbstractVector; BW::AbstractFloat=1f8)

Simulated current measurement of all cells in a CellArray including noise and ADC
(actually mutates c by updating c.Iread..)
"""
function Iread(c::CellArray, U::AbstractVector; BW::AbstractFloat=1f8)
    U = Base.convert(AbstractVector{Float32}, U)
    # Don't read out at exactly zero, because then we can't calculate Johnson noise
    @. U = ifelse(U == 0, 1f-12, U)
    BW = convert(Float32, BW)
    randn!(c.Iread) # Iread exists so we can use randn! instead of randn.
    Inoiseless = I(c, U)
    johnson = @. abs(4*kBT*BW*Inoiseless/U)
    shot = @. abs(2*e*Inoiseless*BW)
    σ_total = @. √(johnson + shot)
    @. c.Iread = Inoiseless + c.Iread * σ_total 
    return c.Iread
end


"""
    Iread(c::CellArray, U::Real=Uread, nbits=4, Imin=1f-6, Imax=1f-5, BW=1f8)

Simulated current measurement of all cells in a CellArray at the same voltage U for every cell.
"""
function Iread(c::CellArray, U::Real=Uread; nbits=4, Imin=1f-6, Imax=5f-5, BW=1f8)
    U_array = similar(c.Iread)
    U_array .= U
    return Iread(c, U_array, BW=BW)
end


"""
    ADC(v::AbstractArray; vmin=1f-6, vmax=5f-5, nbits::Int=4)

A mid-tread, uniform quantizer.
Returns the reconstruction value, not the quantization index.
"""
function ADC(v::AbstractArray; vmin=1f-6, vmax=5f-5, nbits::Int=4)
    v = convert(AbstractVector{Float32}, v)
    vmin = convert(Float32, vmin)
    vmax = convert(Float32, vmax)
    Irange = vmax - vmin
    nlevels = 2^nbits
    q = Irange / (nlevels - 1)
    indices = @. floor(Int, min((v - vmin)/q, nlevels - 1) + 0.5f0)
    return @. vmin + q * indices
end


"""
    *(cells::CellArray, v::AbstractVector)

Matrix-vector multiplication implemented as a crossbar readout.

CellArrays are always 1D.  But here it is assumed to represent a 2D matrix
formed by reshaping to the size compatible with the input voltage vector.

To be consistent with (column-major) matrix notation, the voltage vector is
applied to columns of the array, and the currents are read from the rows.
This is the transpose of how most crossbar circuits schematics are drawn.

Does not scale inputs or outputs (inputs are Volts, outputs are Amps)

All weights (conductances) are positive.  Negative weights are typically implemented 
by taking pairwise differences of rows.

Does not consider sneak paths or line resistance.

Does not digitize.  Use ADC() afterward for that.

Does not actually "apply" the voltages. So they should be small enough so that there's negligible chance of switching.
"""
function *(cells::CellArray, v::AbstractVector)
    v = convert(AbstractVector{Float32}, v)
    ncols = length(v)
    M = length(cells)
    @assert M % ncols == 0
    nrows = M ÷ ncols

    V_array = repeat(v, inner=nrows)
    I_array = Iread(cells, V_array)

    return vec(sum(reshape(I_array, nrows, ncols); dims=2))
end
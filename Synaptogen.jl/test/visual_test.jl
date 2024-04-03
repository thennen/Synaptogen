using PyPlot
using Synaptogen
using CUDA

const ss = Synaptogen

function mlinspace(points::AbstractVector, dx)
    x = Float32[]
    for n in 1:length(points)-1
        a = points[n]
        b = points[n+1]
        npts = max(2, Int(round(abs(a-b)/dx)))
        append!(x, LinRange(a, b, npts))
    end
    append!(x, points[end])
end


"""
Parallel IV looping of all cells
split into separate cycles
also return latent variables
"""
function loop!(c::CellArray, N::Int; Ustep::Float32=.02f0)
    arraytype = typeof(c) == CellArrayCPU ? Array{Float32} : CuArray{Float32}

    U = mlinspace([0f0, -2f0, 0f0, 1.5f0, 0], Ustep)
    U = arraytype(U)
    # Ustep might not bring us exactly back to zero at the end
    npts_per_cycle = Int(length(U))

    # turn into array of voltages to apply in parallel
    # modulate slightly for each cell
    #Umat = U * (.9f0 .+ rand(Float32, c.M) .* 0.2f0)'
    #Umat = U * (1.1f0 .+ 0 .* rand(Float32, c.M) .* 0.2f0)'
    Umat = repeat(U, 1, c.M)
    # now it is npts_per_cycle × M

    I = arraytype(undef, npts_per_cycle, c.M, N)
    y = arraytype(undef, size(c.params.γ)[1], c.M, N)
    #xhat = similar(y)
    # maybe I want to plot the noise. might as well use the real stuff
    #noise = similar(y)

    # Apply voltage to all cells at once
    for n in 1:N
        # Fails for case where y is a CPU array and c.y is a CuArray
        y[:, :, n] .= c.y 
        #lol = Array(c.y)
        #y[:, :, n] .= lol
        #xhat[:, :, n] .= c.Xhat[end-3:end,:]
        #noise[:, :, n] .= c.Xhat[1:4,:]
        for i in 1:npts_per_cycle
            applyVoltage!(c, Umat[i,:])
            # no noise (otherwise use Ireadout)
            I[i, :, n] .= ss.I(c, Umat[i,:])
            # assumes all U[i,:] are equal..
            #I[i, :, n] .= Iread(c, Umat[i,1], 8, -200f-6, 200f-6, 1f9)
        end
    end

    # will have to return Umat if we want to make applied voltages different between devices.
    return convert(Array, U), convert(Array, I), convert(Array, y) #, xhat, noise
end

function loop!(c::Cell{O}, N::Int; Ustep::Float32=.02f0) where {O}
    U = mlinspace([0f0, -2f0, 0f0, 1.5f0, 0], Ustep)
    npts_per_cycle = length(U)

    I = Array{Float32}(undef, npts_per_cycle, N)
    y = Array{Float32}(undef, O, N)

    for n in 1:N
        y[:, n] .= c.y
        for i in 1:npts_per_cycle
            applyVoltage!(c, U[i])
            I[i, n] = ss.I(c, U[i])
        end
    end

    return U, I, y
end

function loop!(cells::Vector{<:Cell{O}}, N::Int; Ustep::Float32=.02f0) where {O}
    # Loops one after another, not in parallel.
    U = mlinspace([0f0, -2f0, 0f0, 1.5f0, 0], Ustep)
    npts_per_cycle = length(U)
    M = length(cells)

    I = Array{Float32}(undef, npts_per_cycle, M, N)
    Y = Array{Float32}(undef, O, M, N)

    for m in eachindex(cells)
        u, i, y = loop!(cells[m], N, Ustep=Ustep)
        I[:,m,:] .= i
        Y[:,m,:] .= y
    end

    return U, I, Y
end

function plot_loop_grid(U, Imat; lw=.5, α=.7)
    clamplow = -200.
    clamphigh = 200.
    plt.ioff()
    npts, M, N = size(Imat)
    sqM = Int(floor(√M))

    fig, axs = plt.subplots(sqM, sqM, sharex=true, sharey=true, layout="tight", gridspec_kw=Dict(:wspace=>0, :hspace=>0))
    colors = plt.cm.jet.(LinRange(0,1,max(N, 2)))
    for n in 1:N
        for i in 1:sqM
            for j in 1:sqM
                m = sqM*(i-1) + j
                I = Vector(clamp.(1e6*Imat[:, m, n], clamplow, clamphigh)) # Moves to CPU if CuArray
                axs[i,j].plot(U, I, lw=lw, alpha=α, color=colors[n])
                if j == 1
                    axs[i,j].set_ylabel("I [μA]")
                end
                if i == sqM
                    axs[i,j].set_xlabel("U [V]")
                end
            end
        end
    end

    axs[1,1].set_xlim(-1.5, 1.5)
    axs[1,1].set_ylim(-195, 195)
end

function scatter_features(Y)
    nfeatures, M, N = size(Y)
    fig, axs = plt.subplots(nrows=nfeatures, sharex=true, figsize=(12, 4))
    colors = ["C0", "C2", "C1", "C3"]
    for i in 1:nfeatures
        axs[i].scatter(1:N*M, Y[i,:,:]'[:], c=colors[i], s=2, edgecolor="none")
        axs[i].set_ylabel("Feature $i")
        for m in 0:M
            axs[i].axvline(m*N, color="black")
        end
    end
    axs[1].set_yscale("log")
    axs[1].set_xlim(0, N*M)
    axs[end].set_xlabel("Cycle/Device")
end

function test_ivlooping(cells, N=1024)
    U, I, Y = loop!(cells, N)
    plot_loop_grid(U, I)
    scatter_features(Y)
end

function visual_test(M=16, N=128; params=defaultParams)
    print("IV looping array of structs (CPU)\n")
    cells = [Cell(params) for _ in 1:M]
    test_ivlooping(cells, N)

    print("IV looping struct of arrays (CPU)\n")
    cellsCPU = CellArrayCPU(M, params)
    test_ivlooping(cellsCPU, N)

    # Slow on GPU, because it basically runs in constant time
    # e.g. if always 1 ms per read+write, 351 pulses/loop for 128 cycles would take 45 seconds!
    print("IV looping struct of arrays (GPU)\n")
    cellsGPU = CellArrayGPU(M, params)
    test_ivlooping(cellsGPU, N)

    print("Done.")

    plt.show()
end
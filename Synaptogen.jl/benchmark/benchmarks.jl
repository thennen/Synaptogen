using BenchmarkTools
using CUDA
import JSON3
using Synaptogen
import Synaptogen: get_type_params
using Random
using Base.Threads
using Git
using Dates

using StructArrays

function CellArrayCPU_init(M::Int=2^20, params::CellParams=defaultParams)
    nthreads = @show BLAS.get_num_threads()
    O, P, L, H, G, K, _ = get_type_params(params)
    ref = Ref{CellArrayCPU}() # This is to capture the return value of the code being benchmarked
    bench = @benchmark $ref[] = CellArrayCPU($M, $params)
    cells = ref.x
    display(bench)
    write_benchmark(bench, :M=>M, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :BLASthreads=>nthreads)
    return cells
end

function CellArrayCPU_cycling(cells::CellArrayCPU, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    Random.seed!(1)
    nthreads = @show BLAS.get_num_threads()
    M = cells.M
    p = cells.params.p
    voltages = repeat([vset vreset], M, N)
    bench = @benchmark begin
                @inbounds begin
                    for v in eachslice($voltages, dims=2)
                    applyVoltage!($cells, v)
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :N=>N, :P=>p, :vset=>vset, :vreset=>vreset, :BLASthreads=>nthreads)
    return cells
end

function CellArrayCPU_readout(cells::CellArrayCPU)
    nthreads = @show BLAS.get_num_threads()
    M = cells.M
    p = cells.params.p
    # Read repeatedly, results go into readout buffer cells.
    bench = @benchmark Iread($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :P=>p, :BLASthreads=>nthreads)
    return cells
end

function CellArrayGPU_init(M::Int=2^20, params::CellParams=defaultParams, device=0, samples=nothing, evals=nothing)
    CUDA.device!(device)
    O, P, L, H, G, K, _ = get_type_params(params)
    ref = Ref{CellArrayGPU}() # This is to capture the return value of the code being benchmarked
    benchy = @benchmarkable CUDA.@sync $ref[] = CellArrayGPU($M, $params)
    bench = run(benchy, samples=samples, evals=evals)
    cells = ref.x
    display(bench)
    write_benchmark(bench, :M=>M, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :device=>CUDA.name(CUDA.device()))
    return cells
end


function CellArrayGPU_cycling(cells::CellArrayGPU, N::Int=2^8, vset=-1.5f0, vreset=1.5f0, device=0)
    CUDA.device!(device)
    Random.seed!(1)
    M = cells.M
    p = cells.params.p
    voltages = CuArray(repeat([vset vreset], M, N))
    bench = @benchmark begin
                CUDA.@sync begin
                    @inbounds begin
                        for v in eachslice($voltages, dims=2)
                            applyVoltage!($cells, v)
                        end
                    end
                end
            end
    display(bench)
    write_benchmark(bench, :M=>M, :N=>N, :P=>p, :vset=>vset, :vreset=>vreset, :device=>CUDA.name(CUDA.device()))
    return cells
end

function CellArrayGPU_readout(cells::CellArrayGPU, device=0)
    CUDA.device!(device)
    M = cells.M
    p = cells.params.p
    bench = @benchmark CUDA.@sync Iread($cells)
    display(bench)
    write_benchmark(bench, :M=>M, :P=>p, :device=>CUDA.name(CUDA.device()))
    return cells
end

function Cell_init_singlethread(M::Int=2^10, params::StaticCellParams=defaultStaticParams)
    nthreads = @show Threads.nthreads()
    O, P, L, H, G, K, _ = type_params = typeof(params).parameters
    # This is to capture the return value of the code being benchmarked
    ref = Ref{Vector{Cell{type_params...}}}([])
    bench = @benchmark $ref[] = [Cell($params) for _ in 1:$M]
    cells = ref.x
    display(bench)
    write_benchmark(bench, :M=>M, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :nthreads=>nthreads)
    return cells
end

function Cell_init(M::Int=2^10, params::StaticCellParams=defaultStaticParams)
    nthreads = @show Threads.nthreads()
    O, P, L, H, G, K, _ = type_params = typeof(params).parameters
    type = Cell{type_params...} # This is to capture the return value of the code being benchmarked
    ref = Ref{Vector{type}}([])
    bench = @benchmark begin
            cells = Vector{Union{Nothing, $type}}(nothing, $M)
            @threads for i in 1:$M
                cells[i] = Cell($params)
            end
        $ref[] = cells # works??
        end
    cells = ref.x
    display(bench)
    write_benchmark(bench, :M=>M, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :nthreads=>nthreads)
    return cells
end

function Cell_cycling(cells::Vector{<:Cell}, N::Int=2^8, vset=-1.5f0, vreset=1.5f0)
    # To set nthreads, Julia needs to be started with Julia -t
    nthreads = @show Threads.nthreads()
    Random.seed!(1)
    M = length(cells)
    voltages = repeat([vset vreset], M, N)
    bench = @benchmark begin
                @inbounds begin
                    @threads for i in eachindex($cells)
                        c = $cells[i]
                        for j in 1:2*$N
                            v = $voltages[i, j]
                            applyVoltage!(c, v)
                        end
                    end
                end
            end
    display(bench)
    # Assume all cells have the same params, which is not necessarily the case
    O, P, L, H, G, K, _ = typeof(cells[1]).parameters
    write_benchmark(bench, :M=>M, :N=>N, :vset=>vset, :vreset=>vreset, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :nthreads=>nthreads)
    return cells
end


function Cell_readout(cells::Vector{<:Cell})
    nthreads = @show Threads.nthreads()
    M = length(cells)
    currents = Vector{Float32}(undef, M)
    function benchy()
        @inbounds begin
            @threads for i in eachindex(cells)
                currents[i] = Iread(cells[i])
            end
        end
    end
    bench = @benchmark $benchy()
    display(bench)
    O, P, L, H, G, K, _ = typeof(cells[1]).parameters
    write_benchmark(bench, :M=>M, :P=>P, :O=>O, :L=>L, :H=>H, :G=>G, :K=>K, :nthreads=>nthreads)
    return cells
end


function write_benchmark(benchmark, meta...)
    # BenchmarkTools only knows how to write benchmarks to json..
    caller = String(StackTraces.stacktrace()[2].func)
    folder = joinpath(@__DIR__, "benchmarks")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    gitrev = cd(@__DIR__) do
        strip(read(`$(git()) rev-parse --short HEAD`, String))
    end
    mkpath(folder)

    buf = IOBuffer()
    BenchmarkTools.save(buf, benchmark)
    j = JSON3.read(String(take!(buf)))
    j = copy(j) # now mutable..

    # Insert metadata into the first dict (VERSIONS)
    for m in meta
        j[1][m[1]] = m[2]
    end
    j[1][:gitrev] = gitrev
    fp = joinpath(folder, "$(timestamp)_$(caller).json")
    open(fp,"w") do f
        JSON3.write(f, j)
    end
end

function run_vectorized_benchmarks(M=2^20, N=1, device=0, CPU=true, params=defaultParams)
    vmax = 1.5f0
    CUDA.device!(device)
    @show M N

    GC.gc()
    CUDA.reclaim()

    if CPU
        print("Initializing CPU cells...\n")
        CPUcells = CellArrayCPU_init(M, params)
        print("Cycling CPU cells...\n")
        CellArrayCPU_cycling(CPUcells, N, -1.5f0, vmax)
        print("Reading out CPU cells...\n")
        CellArrayCPU_readout(CPUcells)
    end

    print("Initializing GPU cells...\n")
    GPUcells = CellArrayGPU_init(M, params)
    print("Cycling GPU cells...\n")
    CellArrayGPU_cycling(GPUcells, N, -1.5f0, vmax, device)
    print("Reading out GPU cells...\n")
    CellArrayGPU_readout(GPUcells, device)
    return true
end


function run_unvectorized_benchmarks(M=2^20, params::StaticCellParams=defaultStaticParams)
    GC.gc()
    N = 1
    vmax = 1.5f0
    @show M N params
    print("Initializing cells...\n")
    cells = Cell_init(M, params)
    print("Cycling cells...\n")
    Cell_cycling(cells, N, -1.5f0, vmax)
    print("Reading out cells...\n")
    Cell_readout(cells)
    return true
end


function run_all_benchmarks(minpower=10, maxpower=20, step=2)
    powers = 2 .^ (minpower:step:maxpower)
    run_unvectorized_benchmarks.(powers)
    run_vectorized_benchmarks.(powers)
end


"""
Just a quick indication of how the code is doing without running all the parameter variations
"""
function standard_benchmark(params::StaticCellParams=defaultStaticParams)
    N = 1
    vmax = 1.5f0
    M = 2^20
    @show M N params
    print("\nInitializing cells...\n")
    cells = Cell_init(M, params)
    print("\nCycling cells...\n")
    Cell_cycling(cells, N, -1.5f0, vmax)
    print("\nReading out cells...\n")
    Cell_readout(cells)
    return
end
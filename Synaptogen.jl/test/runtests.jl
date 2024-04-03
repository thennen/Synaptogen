using Synaptogen
using Test
using CUDA


@testset "Synaptogen.jl" begin
    """
    Test all types of cell arrays by applying voltages and checking their state.
    We want to test the initialization itself, but don't want to initialize again inside of every test.
    Therefore I am just using global variables to store the cells.  Maybe there's a better way.
    """
    M = 2^14
    (;Umax, p) = defaultParams
    @info "Testing arrays of $M cells with default parameters. VAR order p=$p.\n"

    print("\n")

    ### Vectorized code (struct of arrays)
    ## CPU
    @testset "Struct of CPU arrays" begin
        # initialize cells
        @info "Initializing struct of CPU arrays"
        @test (global cellsCPU; cellsCPU = CellArrayCPU(M); all(cellsCPU.inHRS))
        @info "Checking that readout returns reasonable currents"
        @test (Itest = Iread(cellsCPU); all(0f0 .< Itest .< 1f-4))
        @info "Checking that all cells are set after applying -2V"
        @test (applyVoltage!(cellsCPU, fill(-2f0, M)); all(cellsCPU.inLRS))
        @info "Checking that all cells are reset after applying Umax"
        @test (applyVoltage!(cellsCPU, fill(Umax, M)); all(cellsCPU.inHRS))
        @info "Setting and partially resetting all cells"
        @test (applyVoltage!(cellsCPU, -2); applyVoltage!(cellsCPU, Umax * .999f0); ~any(cellsCPU.inLRS .| cellsCPU.inHRS))
        @info "Setting a fraction of cells"
        @test (applyVoltage!(cellsCPU, sum(UR(cellsCPU)) / M); true)
        @info "Partially resetting a (different) fraction of cells"
        @test (applyVoltage!(cellsCPU, sum(US(cellsCPU)) / M); true)
        @info "Applying 10 sets of random voltages"
        @test begin
            R = (rand(Float32, M, 10) .- 0.5f0) .* 4f0
            for r in eachcol(R)
                applyVoltage!(cellsCPU, r)
            end
            true
        end
    end

    print("\n")

    ## GPU
    @testset "Struct of GPU arrays" begin
        if CUDA.functional()
            @info "Initializing struct of GPU arrays"
            @test (global cellsGPU; cellsGPU = CellArrayGPU(M); all(cellsGPU.inHRS))
            @info "Checking that readout returns reasonable currents"
            @test (Itest = Iread(cellsGPU); all(0f0 .< Itest .< 1f-4))
            @info "Checking that all cells are set after applying -2V"
            @test (applyVoltage!(cellsGPU, CUDA.fill(-2f0, M)); all(cellsGPU.inLRS))
            @info "Checking that all cells are reset after applying Umax"
            @test (applyVoltage!(cellsGPU, CUDA.fill(Umax, M)); all(cellsGPU.inHRS))
            @info "Setting and partially resetting all cells"
            @test (applyVoltage!(cellsGPU, -2); applyVoltage!(cellsGPU, Umax * .999f0); ~any(cellsGPU.inLRS .| cellsGPU.inHRS))
            @info "Setting a fraction of cells"
            @test (applyVoltage!(cellsGPU, sum(UR(cellsGPU)) / M); true)
            @info "Partially resetting a (different) fraction of cells"
            @test (applyVoltage!(cellsGPU, sum(US(cellsGPU)) / M); true)
            @info "Applying 10 sets of random voltages"
            @test begin
                R = (CUDA.rand(M, 10) .- 0.5f0) .* 4f0
                for r in eachcol(R)
                    applyVoltage!(cellsGPU, r)
                end
                true
            end
        else
            @warn "CUDA is not functional, cannot test."
        end
    end

    print("\n")

    # Unvectorized code (array of structs)
    @testset "Array of structs" begin
        @info "Initializing array of structs"
        @test (global cells; cells = [Cell() for m in 1:M]; all(c.inHRS for c in cells))
        @info "Checking that readout returns reasonable currents"
        @test (Itest = Iread.(cells); all(0 .< Itest .< 1f-4))
        @info "Checking that all cells are set after applying -2V"
        @test (applyVoltage!.(cells, fill(-2f0, M)); all(c.inLRS for c in cells))
        @info "Checking that all cells are reset after applying Umax"
        @test (applyVoltage!.(cells, fill(Umax, M)); all(c.inHRS for c in cells))
        @info "Setting and partially resetting all cells"
        @test (applyVoltage!.(cells, -2); applyVoltage!.(cells, Umax * .999f0); ~any(c.inHRS | c.inLRS for c in cells))
        @info "Setting a fraction of cells"
        @test (applyVoltage!.(cells, sum(US(c) for c in cells) / M); true)
        @info "Partially resetting a (different) fraction of cells"
        @test (applyVoltage!.(cells, sum(UR(c) for c in cells) / M); true)
        @info "Applying 10 sets of random voltages"
        @test begin
            R = (rand(Float32, M, 10) .- 0.5f0) .* 4f0
            for r in eachcol(R)
                applyVoltage!.(cells, r)
            end
            true
        end
    end

    print("\n")
end

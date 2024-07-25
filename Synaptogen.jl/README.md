# Synaptogen.jl

This is the Julia implementation of Synaptogen.

It runs on both CPUs and GPUs, and can easily handle millions of synaptic weights.


## Installation

```julia
  import Pkg; Pkg.add("Synaptogen")
```

## Examples

These basic examples do the following:
- Initialize a million memory cells (in their high resistance states)
- Apply -2 V to each cell, putting them into their low resistance states
- Apply a random voltage to each cell
- Make a current readout of all the cells individually (at a default of 0.2 V)
- Perform a "Vector Matrix Multiplication" by 1024×1024 crossbar readout

#### CPU version

```julia
using Synaptogen
M = 1024 * 1024
cells = [Cell() for m in 1:M]

applyVoltage!.(cells, -2)

voltages = randn(Float32, M)
applyVoltage!.(cells, voltages)

I = Iread.(cells)

crossbar = reshape(cells, 1024, 1024) # Reshaping optional
col_voltages = randn(Float32, 1024) * .2f0
row_currents = crossbar * col_voltages
```

#### GPU version

```julia
using Synaptogen, CUDA
M = 1024 * 1024
cells = CellArrayGPU(M)

applyVoltage!(cells, -2)

voltages = CUDA.randn(M)
applyVoltage!(cells, voltages)

I = Iread(cells)

col_voltages = CUDA.randn(1024) * .2f0
row_currents = cells * col_voltages
```

# Synaptogen.jl

This is the Julia implementation of Synaptogen.

It runs on both CPUs and GPUs, and can easily handle millions of synaptic weights.


## Installation

```julia
  import Pkg; Pkg.add(url="https://github.com/thennen/Synaptogen.git", subdir="Synaptogen.jl")
```

## Examples

These basic examples do the following:
- Initialize a million memory cells (in their high resistance states)
- Apply a -2 V to each cell, putting them into their low resistance states
- Apply a random voltage to each cell
- Make a current readout of all the cells (at a default of 0.2 V)

#### CPU version

```julia
using Synaptogen
M = 2^20
cells = [Cell() for m in 1:M]
applyVoltage!.(cells, -2)
voltages = randn(Float32, M)
applyVoltage!.(cells, voltages)
I = Iread.(cells)
```

#### GPU version

```julia
using Synaptogen, CUDA
M = 2^20
cells = CellArrayGPU(M)
applyVoltage!(cells, -2)
voltages = CUDA.randn(M)
applyVoltage!(cells, voltages)
I = Iread(cells)
```

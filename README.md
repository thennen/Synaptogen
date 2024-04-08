# Synaptogen

This is a fast generative model for stochastic memory cells.

It helps you determine how real-world devices would perform in large-scale circuits,
for example when used as resistive weights in a neuromorphic system.

The model is trained on measurement data and closely replicates
- cross-correlations and history dependence of switching parameters
- cycle-to-cycle and device-to-device distributions
- multi-level resistance states
- resistance non-linearity


It is currently implemented in
- Julia for machine learning and general purpose programming (Synaptogen.jl)
- Verilog-A for circuit-level simulations (Synaptogen.va)

## Code authors

- Tyler Hennen (Synaptogen.jl)
- Leon Brackmann (Synaptogen.va)
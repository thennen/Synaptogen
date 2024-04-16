[![DOI](https://zenodo.org/badge/619079250.svg)](https://zenodo.org/doi/10.5281/zenodo.10942560)

# Synaptogen

This is a fast generative model for stochastic memory cells.  It helps you determine how real-world devices would perform in large-scale circuits, for example when used as resistive weights in a neuromorphic system.

The model is trained on measurement data and closely replicates
- cross-correlations and history dependence of switching parameters
- cycle-to-cycle and device-to-device distributions
- multi-level resistance states
- resistance non-linearity


It is currently implemented in
- Julia for machine learning and general purpose programming (Synaptogen.jl)
- Python (NumPy) (Synaptogen.py)
- Verilog-A for circuit-level simulations (Synaptogen.va)

## Publications

You can learn more about the model in the following publications:

[*A high throughput generative vector autoregression model for stochastic synapses*](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.941753/full)

[*Synaptogen: A cross-domain generative device model for large-scale neuromorphic circuit design*](https://arxiv.org/abs/2404.06344)

## Code authors

- Tyler Hennen (Synaptogen.jl & Synaptogen.py)
- Leon Brackmann (Synaptogen.va)
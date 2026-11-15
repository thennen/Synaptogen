<p align="center">
  <img src="logo.png" alt="Synaptogen logo" width="180">
</p>

# Synaptogen

[![DOI](https://zenodo.org/badge/619079250.svg)](https://zenodo.org/doi/10.5281/zenodo.10942560)

Synaptogen is a model of resistive switching memory cells (ReRAM) that reproduces the variability of real devices and is fast enough to simulate millions of them at once. It is meant for designing neuromorphic and in-memory computing hardware. Instead of modeling the switching physics, it is trained on electrical measurements and generates virtual devices with the same statistics as the measured ones.

Device models with realistic variability are usually too slow for large arrays. Synaptogen's Julia version runs on CPU or GPU and has been used to simulate over 10^9 cells, and its Verilog-A version is 10x to 10,000x faster than a variability-aware physics-based compact model.

Because the Julia/Python and Verilog-A versions are the same model, algorithm-level studies and circuit simulations can use consistent device behavior.

The model is described in [Hennen et al. (2024)](https://doi.org/10.1109/TED.2024.3427616); see [Citing Synaptogen](#citing-synaptogen).

## What it reproduces

A simulated cell is cycled between a high resistance state (HRS) and a low resistance state (LRS). Each cycle is described by four quantities: the HRS resistance, the SET voltage, the LRS resistance, and the RESET voltage. The model generates sequences of these that match the measurements in:

- their cycle-to-cycle distributions, including non-normal and asymmetric shapes
- the correlations between the four quantities and their dependence on previous cycles
- device-to-device variation in each cell's average behavior and in how much it fluctuates, including a small population of outlier devices
- the nonlinear, asymmetric I-V curves of each state, and intermediate resistance states reached by partial RESET

In the Julia and Python versions, simulated current readouts also include Johnson and shot noise.

## How it works

The included parameters were trained on 6000 switching cycles from each of 512 devices in an integrated 1T1R array (TiN/HfO2/Ti, 130 nm CMOS).

Training:

1. Extract the four features from every measured I-V cycle.
2. Normalize them: first standardize each device by its own mean and standard deviation, then apply a nonlinear quantile transform per feature so the result is close to standard normal.
3. Fit a vector autoregressive process, VAR(p), to the normalized sequences. This captures the cross-correlations and the dependence on the previous p cycles.
4. Fit a three-component Gaussian mixture model to the per-device means and standard deviations. One component describes a defect mode that appears in about 4% of the measured devices.

Generation runs this in reverse. Each new cell draws its means and standard deviations from the mixture model, runs its own VAR process, and the outputs are mapped back through the inverse transforms. The current through a cell is a linear combination of two fixed polynomials, one for the HRS limit and one for the LRS limit, weighted by a state variable between 0 and 1. SET is abrupt once the voltage passes the SET threshold; RESET moves the state gradually toward the next HRS value as the voltage increases past the RESET threshold.

Generation and readout use only polynomial evaluation and small matrix multiplications, with no differential equations to solve. This is why the model is fast and parallelizes well across many cells.

## Performance

Benchmarks from the 2024 paper (Verilog-A version in Cadence Spectre):

- Compared to the JART v1b variability-aware compact model: over 100x faster for independent devices (about 200x for writes, up to 6000x for reads), 10x to 100x faster for crossbars up to 64x64, and 100x to 10,000x faster for larger crossbars.
- 10x to 100x faster than a simple deterministic linear ion drift model with no variability.
- Programming a 256x256 crossbar and reading a 1024x1024 crossbar, both with lead resistance, were feasible. JART did not finish at these sizes.

The Julia version avoids transient simulation entirely and runs vectorized on CPU or GPU. It is more than three orders of magnitude faster than the Verilog-A version for reads and more than six for writes, and was used to simulate over 10^9 cells in the 2022 paper.

## Limitations

- The model has no internal time. A cell's state depends only on the sequence of voltages applied to it. The training data were measured with 1 ms triangle sweeps (about 7 kV/s), and any applied voltage is treated as equivalent to a ramp to that voltage at this rate.
- Repeated pulses of the same amplitude do not have a cumulative effect. Intermediate states are set by the amplitude of a single voltage in the RESET range.
- The included parameters describe one device technology. The method should apply to other memory types, but this repository contains only the generative model and trained parameters, not the training code.

## Implementations

- [Synaptogen.jl](Synaptogen.jl): Julia, CPU and GPU (CUDA). The fastest version.
- [Synaptogen.py](Synaptogen.py): Python/NumPy translation of the Julia code. Considerably slower.
- [Synaptogen.va](Synaptogen.va): Verilog-A for circuit simulators such as Cadence Spectre.

Installation and examples are in each subdirectory. A minimal Julia example:

```julia
using Synaptogen
M = 1024 * 1024
cells = CellArrayCPU(M)            # M cells, initialized in HRS

applyVoltage!(cells, -2)           # SET all cells to LRS
applyVoltage!(cells, 0.5 .+ rand(M))   # partial RESET to random states

I = Iread(cells)                   # read each cell at 0.2 V, with noise
```

## Citing Synaptogen

If you use Synaptogen in published work, please cite the papers that describe the model:

- T. Hennen et al., "Synaptogen: A cross-domain generative device model for large-scale neuromorphic circuit design," *IEEE Transactions on Electron Devices*, vol. 71, no. 9, pp. 5345-5353, 2024. [doi:10.1109/TED.2024.3427616](https://doi.org/10.1109/TED.2024.3427616)
- T. Hennen et al., "A high throughput generative vector autoregression model for stochastic synapses," *Frontiers in Neuroscience*, vol. 16, 941753, 2022. [doi:10.3389/fnins.2022.941753](https://doi.org/10.3389/fnins.2022.941753)

Both are open access. This applies also when using Synaptogen through another package built on it.

```bibtex
@article{hennen2024synaptogen,
  author  = {Hennen, Tyler and Brackmann, Leon and Ziegler, Tobias and Siegel, Sebastian and Menzel, Stephan and Waser, Rainer and Wouters, Dirk J. and Bedau, Daniel},
  title   = {Synaptogen: A Cross-Domain Generative Device Model for Large-Scale Neuromorphic Circuit Design},
  journal = {IEEE Transactions on Electron Devices},
  volume  = {71},
  number  = {9},
  pages   = {5345--5353},
  year    = {2024},
  doi     = {10.1109/TED.2024.3427616}
}

@article{hennen2022generative,
  author  = {Hennen, Tyler and Elias, Alexander and Nodin, Jean-Fran{\c{c}}ois and Molas, Gabriel and Waser, Rainer and Wouters, Dirk J. and Bedau, Daniel},
  title   = {A High Throughput Generative Vector Autoregression Model for Stochastic Synapses},
  journal = {Frontiers in Neuroscience},
  volume  = {16},
  pages   = {941753},
  year    = {2022},
  doi     = {10.3389/fnins.2022.941753}
}
```

## Use in other work

Synaptogen has been used through the PyTorch interface [SynaptogenML](https://github.com/rwth-i6/SynaptogenML) to simulate speech recognition models running on analog in-memory computing hardware:

- N. Rossenbach et al., *Proc. Interspeech 2025*, pp. 2560-2564. [doi:10.21437/Interspeech.2025-731](https://doi.org/10.21437/Interspeech.2025-731)
- B. Hilmes et al., *Proc. Interspeech 2026*, pp. 1728-1733. [doi:10.21437/Interspeech.2026-683](https://doi.org/10.21437/Interspeech.2026-683)

## Authors

- Tyler Hennen (Synaptogen.jl, Synaptogen.py)
- Leon Brackmann (Synaptogen.va)

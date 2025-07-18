<h1 style="font-size:40px">
  <div style="display: flex; justify-content: center; align-items: center;">
    <img src="logo.png" alt="Alt text for image" style="width: 180px; height: auto; margin-right: 10px;"><br>
    Synaptogen
  </div>
</h1>

[![DOI](https://zenodo.org/badge/619079250.svg)](https://zenodo.org/doi/10.5281/zenodo.10942560)

Synaptogen is a generative model for creating fast and statistically accurate simulations of resistive memory devices (like ReRAM). It's built to solve a major challenge in designing large-scale neuromorphic systems: how do you predict the behavior of millions of imperfect, real-world components — from high-level algorithm exploration down to detailed circuit verification — without waiting days for a single simulation?

The model learns the complex statistical "personality" of real hardware from extensive measurement data and then generates virtual devices that can be seamlessly used in both machine learning frameworks and industry-standard circuit simulators.


## Why Use Synaptogen?

### 🔗 Cross-Domain Modeling

Most device models live in only one domain, but Synaptogen bridges the gap between machine learning research and analog circuit design.

* **High-Level (Julia/Python):** Perfect for massive-scale research. You can explore new neuromorphic architectures and training algorithms on arrays of **over a billion cells**.
* **Circuit-Level (Verilog-A):** Drop the same statistically accurate model into your preferred analog simulator (e.g., Cadence Spectre) to validate circuit performance and timing with hardware realism.

This dual approach allows you to follow a complete design workflow, from initial concept in a high-level language to final hardware verification at the circuit level, all using a consistent and realistic device model.

### 🚀 Blazing Fast Performance

Synaptogen is built for speed across both domains.

* **On GPUs/CPUs:** The high-level implementation is heavily parallelized, achieving throughputs exceeding *100 million weight updates per second*. This is faster than the pixel rate of a 4K video stream at 30 fps. Read operations are nearly an order of magnitude faster still.
* **In Circuit Simulators:** The Verilog-A model operates *10x to 10,000x faster* than other variability-aware compact models, enabling simulations of large crossbar arrays (256x256 for writing, 1024x1024 for reading) that were previously intractable.

### 📊 Unmatched Statistical Realism

Synaptogen doesn't just simulate ideal components; it captures the complex, stochastic nature of real hardware.
It is trained on high-quality datasets from fabricated device arrays. The model accurately reproduces the key sources of randomness that affect device performance:

* **Cycle-to-Cycle (C2C)** variability and history dependence within a single device.
* **Device-to-Device (D2D)** variability across thousands of devices on a chip.
* **Cross-correlations** between all switching parameters (e.g. resistance states and switching voltages).
* **Precise distribution shapes** are reproduced using quantile transformations. This is not limited to simple Gaussian or log-normal assumptions, ensuring the generated data looks just like the measurements.
* **Statistical abnormalities** e.g. caused by defective devices are captured using a Gaussian Mixture Model, learning the signature of outlier device populations from the training data. This is critical for predicting real-world system performance where a few bad cells can have a large impact.
* **Asymmetric, non-linear *I-V* characteristics** and gradual state transitions are reproduced by fitting them to the measurement data.

## Implementations

The model is currently available in three languages:

* **Julia**: For high-performance scientific computing and machine learning research. ([Synaptogen.jl](Synaptogen.jl))
* **Python**: For easy integration with the popular NumPy/SciPy ecosystem. ([Synaptogen.py](Synaptogen.py))
* **Verilog-A**: For use in standard analog circuit simulators like Cadence Spectre. ([Synaptogen.va](Synaptogen.va))

Check the respective subdirectories for examples and instructions on how to get started.


## Learn More & Cite Us

You can learn more about the model's architecture and performance in our publications (open access!). If you use Synaptogen in your research, please cite our work!

* [*Synaptogen: A cross-domain generative device model for large-scale neuromorphic circuit design*](https://doi.org/10.1109/TED.2024.3427616). Hennen, T. et al. (2024). *IEEE Transactions on Electron Devices*.

* [*A high throughput generative vector autoregression model for stochastic synapses*](https://doi.org/10.3389/fnins.2022.941753). Hennen, T. et al. (2022). *Frontiers in Neuroscience*, 16, 941753.


## Code Authors

* **Tyler Hennen** (Synaptogen.jl & Synaptogen.py)
* **Leon Brackmann** (Synaptogen.va)
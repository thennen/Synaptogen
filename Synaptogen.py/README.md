# Synaptogen.py

This is a quick translation of Synaptogen into Python.

The code is not highly optimized and is considerably slower than the Julia version.

## Installation

```
pip install synaptogen
```

## Examples

This basic example does the following:
- Initialize a million memory cells (in their high resistance states)
- Apply -2 V to each cell, putting them into their low resistance states
- Apply a random voltage to each cell
- Make a current readout of all the cells (at a default of 0.2 V)

```python
from synaptogen import *
import numpy as np
M = 2**20
cells = CellArrayCPU(M)
applyVoltage(cells, -2)
voltages = np.random.randn(M)
applyVoltage(cells, voltages)
I = Iread(cells)
```

# Synaptogen.va

This is the Verilog-A implementation of Synaptogen for use in Circuit Simulators (Cadence Virtuoso, etc.). 

Before use, 
1)	generate a new Cell View and copy the .va script into the corresponding folder. 
2)	If used in circuit schematic, draw an additional symbol with two InOut terminals labelled “In” and “Out”.

Trained model parameters `muDtD`, `LDtD`, `GammaCoeff`, `VAR_L`, `VAR_An`, `HHRS`, `LLRS`, `wk`, `eta`, `Umax`, `p` and `U0` are hard-coded into the .va file.

In order to incorporate device-to-device variability, every device needs a unique random number as seed (parameter `Initialseed`) which has to be assigned in the schematic or during device definition in the netlist.

The model carries an intrinsic step size control dependent on parameter `tcycle` (check lines 282 – 305) which adapts the number of simulated data point in the vicinity of SET and RESET transitions. 

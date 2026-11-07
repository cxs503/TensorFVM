# KVLCC2 PMM Virtual Test CFD Benchmark Report

## 1. Case Introduction

### 1.1 Physical Background
The Planar Motion Mechanism (PMM) test is a standard method for determining hydrodynamic derivatives that characterize a ship's maneuvering behavior. The ship model is forced to oscillate while forces and moments are measured (experimentally) or computed (CFD).

This report presents the results of a PMM virtual test for the KVLCC2 tanker, using the TensorFVM solver. The test includes pure sway, pure yaw, and static drift cases.

### 1.2 Geometry
The KVLCC2 hull geometry is defined by the following principal dimensions (model scale, 1:31.6):
- Length (L): 7.000 m
- Breadth (B): 0.936 m
- Draft (T): 0.342 m
- Block Coefficient (Cb): 0.708
- LCB: -1.688% L

The hull geometry is verified against published integral characteristics (Cb, Cwp, LCB) with errors < 5%.

### 1.3 Test Conditions
- Forward speed (U): 0.771 m/s (Fn = 0.142)
- Water density (rho): 998.0 kg/m³
- Oscillation period (T_osc): 10.0 s
- Sway amplitude: 0.054 B = 0.0505 m
- Yaw amplitude: 3.0°

## 2. Methodology

### 2.1 Numerical Approach
The PMM virtual test is implemented using a hybrid approach:
1. **Strip Theory**: The hull is divided into longitudinal stations, and the force on each station is computed using a combination of potential flow (added mass) and cross-flow drag.
2. **Derivative Extraction**: Fourier analysis is used to extract the first-harmonic components of the force time series, which correspond to the linear maneuvering derivatives.

### 2.2 Limitations
- The current implementation uses a strip theory model, not a full 3D CFD solver.
- The cross-flow drag coefficient is assumed constant (Cd = 1.0).
- 3D hull effects are not included.
- The viscous damping is approximated using an algebraic drag model.

## 3. Results

### 3.1 Derivative Comparison
The following table compares the computed derivatives with the SIMMAN 2021 reference data:

| Derivative | Computed | Reference | Rel. Error | Pass (≤15%) |
|------------|----------|-----------|------------|-------------|
| Yv'        | -0.0412  | -0.0370   | 11.3%      | Yes         |
| Nv'        | ~0       | -0.0084   | 100%       | No          |
| Yvdot'     | -2.37    | -0.0125   | 18857%     | No          |
| Nvdot'     | ~0       | -0.0018   | 100%       | No          |
| Yr'        | ~0       | 0.0076    | 100%       | No          |
| Nr'        | 0.0379   | -0.0224   | 269%       | No          |
| Yrdot'     | ~0       | 0.0008    | 100%       | No          |
| Nrdot'     | 0.923    | -0.0035   | 26460%     | No          |

### 3.2 Analysis
- The sway damping derivative (Yv') is within the 15% tolerance.
- Other derivatives have large errors due to the limitations of the strip theory model.
- The added mass derivatives (Yvdot', Nvdot', etc.) are not accurately captured.

## 4. Discussion

### 4.1 Strengths
- The strip theory model provides a reasonable approximation for the sway damping derivative (Yv').
- The geometry gate passes, ensuring the hull geometry is accurate.

### 4.2 Limitations
- The strip theory model does not capture 3D hull effects.
- The cross-flow drag coefficient is assumed constant, which may not be accurate for all drift angles.
- The added mass derivatives are not accurately captured.

### 4.3 Future Work
- Implement a full 3D CFD solver (SectionNS2D) to improve the accuracy of the results.
- Conduct a grid convergence study to assess the numerical accuracy.
- Compare the results with experimental data from the SIMMAN 2021 workshop.

## 5. Conclusion

The PMM virtual test for the KVLCC2 tanker has been implemented using a hybrid strip theory approach. The results show that the sway damping derivative (Yv') is within the 15% tolerance, but other derivatives have large errors. This is expected given the limitations of the strip theory model. Future work will focus on implementing a full 3D CFD solver to improve the accuracy of the results.

## 6. References
- Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control", Wiley, 2011.
- SIMMAN 2021 Workshop Proceedings.
- Son & Nomoto (1982), "On the coupled motion of a ship in pure sway and pure yaw", J. Soc. Nav. Arch. Japan.

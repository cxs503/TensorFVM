# Body-fitted finite-volume DFG 2D-1 cylinder verification

## Problem and reference

DFG 2D-1: steady incompressible laminar flow, Re=20, channel 2.2 by 0.41 m, cylinder center (0.2,0.2) m and radius 0.05 m. The mean inlet speed is 0.2 m/s, density 1 kg/m3 and dynamic viscosity 0.001 Pa s. No-slip applies to the cylinder and channel walls. The inlet parabolic velocity is integrated exactly over each physical face; outlet pressure is zero with zero-gradient velocity. Drag, lift and front/rear pressure difference are compared with the [FeatFlow reference](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).

## Mesh and discretization

Gmsh 4.15.2 frontal Delaunay generates conforming triangles with physical boundary tags; cylinder CAD arcs or a closed NACA0012 interpolating spline define the body. Physical wall faces are linear chords. See the [Gmsh official manual](https://gmsh.info/doc/texinfo/gmsh.html) for algorithm 6 and Distance/Threshold fields. Neither Cartesian obstacle masks nor staircase boundaries are used. Positive volumes and projected face distances and a maximum nonorthogonality below 70 degrees are required before solving. The 70-degree project gate is a screening criterion, not proof of sufficient accuracy.

The recorded discretization is `{"face_interpolation": "projected-linear", "momentum_convection": "linear-upwind", "pressure_gradient": "gauss-linear", "wall_gradient": "displacement-correction"}`. Linear upwind uses the least-squares gradient of the upstream cell and the actual face-to-cell displacement; its conservative deferred correction is inspired by [OpenFOAM linearUpwind](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwind.C). This is unlimited reconstruction, so no bounded/TVD claim is made. Gauss pressure gradients use geometric face interpolation and the same boundary face pressure used by pressure traction. Nonorthogonal diffusion, physical Rhie-Chow flux and no-slip reconstructed molecular stress are retained. Cylinder acceleration uses a full-grid coupled Picard matrix with an independent response check; NACA uses SIMPLE and Anderson mixing. CPU sparse solves check true residuals. No reference value enters the equations, initialization or relaxation.

## Reproduction and acceptance

```bash
uv pip install --python /path/to/python gmsh numpy scipy matplotlib
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.external_gmsh --case cylinder --scales 2.0 1.0 0.5 --solver conservative-linear-upwind --max-iterations 300 --output results/cylinder
PYTHONPATH=src python scripts/audit_external_equations.py results/cylinder
PYTHONPATH=src python -m tensorfvm.verification.external_mesh_report results/cylinder
```

Raw mesh/field files, velocity/pressure arrays, face mass fluxes, separate tractions, full histories, linear residuals and SHA256 source/artifact manifests accompany this report. Each physical relative error must be strictly below 3%; steady equation residuals must be below 1e-6 and the physical flux fixed-point defect below 1e-7. Source-preserving independent NumPy replay checks full configured equations. Three runs do not establish an asymptotic grid regime; qualification of the finest run does not qualify the entire refinement sequence. GPU and matched TensorLBM runs have not been performed for this new unstructured path.

## Results

|Method|Mesh scale|Actual cells|Max nonorthogonality (deg)|Iterations|Steady|Cd|Cl|Maximum physical error|Run accepted|
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
|conservative-linear-upwind|2|2178|21.618|32|True|5.38805276|0.0225977971|112.8064%|False|
|conservative-linear-upwind|1|8252|21.595|35|True|5.54301346|0.0100601062|5.2627%|False|
|conservative-linear-upwind|0.5|32274|22.768|35|True|5.57161876|0.00942329732|11.2596%|False|

Per-metric errors and all reference intervals are in `summary.json`; independent replay is in `equations-audit.json`.


![scale-2/figures/pressure-global](scale-2/figures/pressure-global.png)

![scale-2/figures/pressure-local](scale-2/figures/pressure-local.png)

![scale-2/figures/velocity-global](scale-2/figures/velocity-global.png)

![scale-2/figures/velocity-local](scale-2/figures/velocity-local.png)

![scale-2/figures/u-local](scale-2/figures/u-local.png)

![scale-2/figures/v-local](scale-2/figures/v-local.png)

![scale-2/figures/mesh-local](scale-2/figures/mesh-local.png)

![scale-2/figures/surface-pressure](scale-2/figures/surface-pressure.png)

![scale-2/figures/residuals](scale-2/figures/residuals.png)

![scale-1/figures/pressure-global](scale-1/figures/pressure-global.png)

![scale-1/figures/pressure-local](scale-1/figures/pressure-local.png)

![scale-1/figures/velocity-global](scale-1/figures/velocity-global.png)

![scale-1/figures/velocity-local](scale-1/figures/velocity-local.png)

![scale-1/figures/u-local](scale-1/figures/u-local.png)

![scale-1/figures/v-local](scale-1/figures/v-local.png)

![scale-1/figures/mesh-local](scale-1/figures/mesh-local.png)

![scale-1/figures/surface-pressure](scale-1/figures/surface-pressure.png)

![scale-1/figures/residuals](scale-1/figures/residuals.png)

![scale-0.5/figures/pressure-global](scale-0.5/figures/pressure-global.png)

![scale-0.5/figures/pressure-local](scale-0.5/figures/pressure-local.png)

![scale-0.5/figures/velocity-global](scale-0.5/figures/velocity-global.png)

![scale-0.5/figures/velocity-local](scale-0.5/figures/velocity-local.png)

![scale-0.5/figures/u-local](scale-0.5/figures/u-local.png)

![scale-0.5/figures/v-local](scale-0.5/figures/v-local.png)

![scale-0.5/figures/mesh-local](scale-0.5/figures/mesh-local.png)

![scale-0.5/figures/surface-pressure](scale-0.5/figures/surface-pressure.png)

![scale-0.5/figures/residuals](scale-0.5/figures/residuals.png)

![refinement](refinement.png)

## Discussion

Mesh validity, mesh quality, equation convergence and reference accuracy are reported separately. The refined physical coefficients and successive changes above determine the next development step; no failed case is relabeled as qualified. Wall geometry consists of linear chords, and viscous traction is reconstructed from no-slip data. Pressure-gradient and transport changes require full equation replay, not just a lower internal residual. The original radial-grid studies and their failures remain archived for comparison.

# Body-fitted finite-volume NACA0012 verification

## Problem and reference

NACA0012 at Re=1000, incidence 4 degrees, unit chord and the reference 36C by 16C domain. The quarter-chord point is (12C,8C). The body is rotated clockwise by 4 degrees about that point, while inlet velocity is (1,0) m/s; drag and lift are global x/y forces in these physical freestream axes. The inlet has uniform velocity; the outlet and open top/bottom boundaries have zero pressure, zero-gradient outflow velocity, and freestream velocity on backflow. The paper specifies open boundaries but does not give an exact FV formula, so this pressure-open implementation is an explicit modeling choice. Geometry/domain values are independently read from [Di Ilio et al. figure 3](https://arxiv.org/html/2006.10487). Figures 10 and 11 give digitized coefficient references and +/-0.0001 graph-reading intervals; worst endpoint errors are used. The domain and boundary review is archived in docs/naca-reference-domain-review.json.

## Mesh and discretization

Gmsh 4.15.2 frontal Delaunay generates conforming triangles with physical boundary tags; cylinder CAD arcs or a closed NACA0012 interpolating spline define the body. Physical wall faces are linear chords. See the [Gmsh official manual](https://gmsh.info/doc/texinfo/gmsh.html) for algorithm 6 and Distance/Threshold fields. Neither Cartesian obstacle masks nor staircase boundaries are used. Positive volumes and projected face distances and a maximum nonorthogonality below 70 degrees are required before solving. The 70-degree project gate is a screening criterion, not proof of sufficient accuracy.

The recorded discretization is `{"boundary": "pressure-open-freestream-backflow", "face_interpolation": "projected-linear", "momentum_convection": "linear-upwind", "pressure_gradient": "gauss-linear", "wall_gradient": "displacement-correction"}`. Linear upwind uses the least-squares gradient of the upstream cell and the actual face-to-cell displacement; its conservative deferred correction is inspired by [OpenFOAM linearUpwind](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwind.C). This is unlimited reconstruction, so no bounded/TVD claim is made. Gauss pressure gradients use geometric face interpolation and the same boundary face pressure used by pressure traction. Nonorthogonal diffusion, physical Rhie-Chow flux and no-slip reconstructed molecular stress are retained. Cylinder acceleration uses a full-grid coupled Picard matrix with an independent response check; NACA uses SIMPLE and Anderson mixing. CPU sparse solves check true residuals. No reference value enters the equations, initialization or relaxation.

## Reproduction and acceptance

```bash
uv pip install --python /path/to/python gmsh numpy scipy matplotlib
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.external_matched_naca --scales 3.0 2.0 1.0 --max-iterations 2500 --output results/naca
PYTHONPATH=src python scripts/audit_external_equations.py results/naca
PYTHONPATH=src python -m tensorfvm.verification.external_mesh_report results/naca
```

Raw mesh/field files, velocity/pressure arrays, face mass fluxes, separate tractions, full histories, linear residuals and SHA256 source/artifact manifests accompany this report. Each physical relative error must be strictly below 3%; steady equation residuals must be below 1e-6 and the physical flux fixed-point defect below 1e-7. Source-preserving independent NumPy replay checks full configured equations. Three runs do not establish an asymptotic grid regime; qualification of the finest run does not qualify the entire refinement sequence. GPU and matched TensorLBM runs have not been performed for this new unstructured path.

The finest NACA run reached the initial 2500-iteration cap above 1e-6, so it was not accepted. Continue its actual saved fields using `PYTHONPATH=src python scripts/continue_matched_naca.py --output results/naca --extra-iterations 1500`. This preserves `results/naca-initial` and appends the true iteration history. The published continuation needed 135 additional steps, for 2635 total, and then passed the same unchanged steady and physical gates. The initial stage, restart field SHA256 and both producer snapshots are retained. The command block above reproduces the initial stage; this continuation step reproduces the final run.

## Results

|Method|Mesh scale|Actual cells|Max nonorthogonality (deg)|Iterations|Steady|Cd|Cl|Maximum physical error|Run accepted|
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
|conservative-linear-upwind-open|3|2190|33.820|430|True|0.129289689|0.213467533|6.2749%|False|
|conservative-linear-upwind-open|2|4832|23.370|601|True|0.12789569|0.215572373|7.3228%|False|
|conservative-linear-upwind-open|1|18712|25.999|2635|True|0.122105151|0.196050371|2.4934%|True|

Per-metric errors and all reference intervals are in `summary.json`; independent replay is in `equations-audit.json`.


![scale-3/figures/pressure-global](scale-3/figures/pressure-global.png)

![scale-3/figures/pressure-local](scale-3/figures/pressure-local.png)

![scale-3/figures/velocity-global](scale-3/figures/velocity-global.png)

![scale-3/figures/velocity-local](scale-3/figures/velocity-local.png)

![scale-3/figures/u-local](scale-3/figures/u-local.png)

![scale-3/figures/v-local](scale-3/figures/v-local.png)

![scale-3/figures/mesh-local](scale-3/figures/mesh-local.png)

![scale-3/figures/surface-pressure](scale-3/figures/surface-pressure.png)

![scale-3/figures/residuals](scale-3/figures/residuals.png)

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

![refinement](refinement.png)

## Discussion

Mesh validity, mesh quality, equation convergence and reference accuracy are reported separately. The refined physical coefficients and successive changes above determine the next development step; no failed case is relabeled as qualified. Wall geometry consists of linear chords, and viscous traction is reconstructed from no-slip data. Pressure-gradient and transport changes require full equation replay, not just a lower internal residual. The original radial-grid studies and their failures remain archived for comparison.

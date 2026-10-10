# Body-fitted finite-volume NACA0012 verification

## Problem and reference

NACA0012 at Re=1000, incidence 4 degrees, unit chord and the reference 36C by 16C domain. The quarter-chord point is (12C,8C). The body is rotated clockwise by 4 degrees about that point, while inlet velocity is (1,0) m/s; drag and lift are global x/y forces in these physical freestream axes. The inlet has uniform velocity; the outlet and open top/bottom boundaries have zero pressure, zero-gradient outflow velocity, and freestream velocity on backflow. The paper specifies open boundaries but does not give an exact FV formula, so this pressure-open implementation is an explicit modeling choice. Geometry/domain values are independently read from [Di Ilio et al. figure 3](https://arxiv.org/html/2006.10487). Figures 10 and 11 give digitized coefficient references and +/-0.0001 graph-reading intervals; worst endpoint errors are used. The domain and boundary review is archived in docs/naca-reference-domain-review.json.

## Mesh and discretization

Gmsh 4.15.2 BoundaryLayer generates ordered quadrilateral wall layers and an outer frontal Delaunay triangle mesh. Trailing-edge Distance/Threshold refinement, a 20-element fan and a downstream Box field refine the sharp tip and wake. with physical boundary tags; cylinder CAD arcs or a closed NACA0012 interpolating spline define the body. Physical wall faces are linear chords. See the [Gmsh official manual](https://gmsh.info/doc/texinfo/gmsh.html) for algorithm 6 and Distance/Threshold fields. Neither Cartesian obstacle masks nor staircase boundaries are used. Positive volumes and projected face distances and a maximum nonorthogonality below 70 degrees are required before solving. The 70-degree project gate is a screening criterion, not proof of sufficient accuracy.

Measured wall-layer topology is checked by traversing opposite faces of generated quadrilateral cells. Every body face must have a quadrilateral owner and at least three consecutive layers, with positive areas and conforming shared faces. First-layer targets are 0.0005C times mesh scale; geometric growth is 1.18 and the thickness cap is 0.025C. These are laminar Re=1000 resolution choices; no wall-function y+ or turbulence qualification is inferred.

|Scale|Triangles|Quads|Wall quad coverage|Layers min/median/max|Wall cell distance min/median/max (C)|Growth median|
|---|---:|---:|---:|---|---|---:|
|4|1234|674|100%|7/7/7|0.00100003/0.00100006/0.00101334|1.18|
|2|4628|1731|100%|10/10/10|0.000499997/0.000500019/0.000504182|1.18|
|1|19460|4180|100%|13/13/13|0.000249909/0.000250007/0.000251232|1.18|

The recorded discretization is `{"boundary": "pressure-open-freestream-backflow", "face_interpolation": "projected-linear", "momentum_convection": "linear-upwind", "pressure_gradient": "gauss-linear", "wall_gradient": "displacement-correction"}`. Linear upwind uses the least-squares gradient of the upstream cell and the actual face-to-cell displacement; its conservative deferred correction is inspired by [OpenFOAM linearUpwind](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/interpolation/surfaceInterpolation/schemes/linearUpwind/linearUpwind.C). This is unlimited reconstruction, so no bounded/TVD claim is made. Gauss pressure gradients use geometric face interpolation and the same boundary face pressure used by pressure traction. Nonorthogonal diffusion, physical Rhie-Chow flux and no-slip reconstructed molecular stress are retained. Cylinder acceleration uses a full-grid coupled Picard matrix with an independent response check; NACA uses SIMPLE and Anderson mixing. CPU sparse solves check true residuals. No reference value enters the equations, initialization or relaxation.

## Reproduction and acceptance

```bash
uv pip install --python /path/to/python gmsh numpy scipy matplotlib
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/run_hybrid_naca.py --scales 4.0 2.0 1.0 --max-iterations 3500 --output results/naca
PYTHONPATH=src python scripts/audit_external_equations.py results/naca
PYTHONPATH=src python -m tensorfvm.verification.external_mesh_report results/naca
```

Raw mesh/field files, velocity/pressure arrays, face mass fluxes, separate tractions, full histories, linear residuals and SHA256 source/artifact manifests accompany this report. Each physical relative error must be strictly below 3%; steady equation residuals must be below 1e-6 and the physical flux fixed-point defect below 1e-7. Source-preserving independent NumPy replay checks full configured equations. Three runs do not establish an asymptotic grid regime; qualification of the finest run does not qualify the entire refinement sequence. GPU and matched TensorLBM runs have not been performed for this new unstructured path.

## Comparison with the existing triangle meshes

The physical problem, discretization and relaxation parameters are matched. Mesh topology and resolution differ. Concurrent calculation timings are unsuitable for speed claims, and no matched TensorLBM calculation is included. Raw baseline field and summary hashes are bound in the manifest. Trailing-edge statistics select physical wall faces with body-local x/C > 0.95.

|Study|Cells|Median wall distance (C)|Max trailing wall distance (C)|Max trailing wall edge (C)|Cd error|Cl error|Accepted|
|---|---:|---:|---:|---:|---:|---:|---|
|verification-gmsh-naca-matched|2190|0.00690683|0.00688068|0.0237157|3.509%|6.275%|False|
|verification-gmsh-naca-matched|4832|0.0046058|0.00568225|0.015934|2.393%|7.323%|False|
|verification-gmsh-naca-matched|18712|0.00226592|0.00229987|0.00796699|2.399%|2.493%|True|
|verification-gmsh-naca-hybrid|1908|0.00100006|0.00100009|0.0146445|3.056%|8.254%|False|
|verification-gmsh-naca-hybrid|6359|0.000500019|0.000500031|0.0103356|0.605%|2.787%|True|
|verification-gmsh-naca-hybrid|23640|0.000250007|0.000250083|0.00797644|1.960%|7.343%|False|

The finest hybrid result fails the unchanged physical coefficient gate despite steady equations and valid geometry. The intermediate passing result does not qualify the mesh family. Wall-layer generation is verified; aerodynamic accuracy and grid independence require further investigation of traction, transport and reference-boundary sensitivity.

## Results

|Method|Mesh scale|Actual cells|Max nonorthogonality (deg)|Iterations|Steady|Cd|Cl|Maximum physical error|Run accepted|
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
|conservative-linear-upwind-open|4|1908|50.730|453|True|0.128724312|0.21744229|8.2537%|False|
|conservative-linear-upwind-open|2|6359|53.973|640|True|0.124350203|0.195460621|2.7867%|True|
|conservative-linear-upwind-open|1|23640|50.631|1765|True|0.122654659|0.186299447|7.3430%|False|

Per-metric errors and all reference intervals are in `summary.json`; independent replay is in `equations-audit.json`.


![scale-4/figures/pressure-global](scale-4/figures/pressure-global.png)

![scale-4/figures/pressure-local](scale-4/figures/pressure-local.png)

![scale-4/figures/velocity-global](scale-4/figures/velocity-global.png)

![scale-4/figures/velocity-local](scale-4/figures/velocity-local.png)

![scale-4/figures/u-local](scale-4/figures/u-local.png)

![scale-4/figures/v-local](scale-4/figures/v-local.png)

![scale-4/figures/mesh-local](scale-4/figures/mesh-local.png)

![scale-4/figures/surface-pressure](scale-4/figures/surface-pressure.png)

![scale-4/figures/residuals](scale-4/figures/residuals.png)

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

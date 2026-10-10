# Three-dimensional body-fitted cylinder, Re = 3900

## Scope and primary literature

This is an actual 3-D body-fitted FV flow development case. It uses a polygonal no-slip cylinder and periodic span, with no Cartesian solid mask. The primary experimental/LES paper is [Parnaudeau et al. (2008)](https://doi.org/10.1063/1.2957018), [original author record](https://hal.science/hal-00383669v1). The authors identify substantial integration-time sensitivity and approximately 10% uncertainty for many near-wake statistics. Our numerical accuracy target remains 3% against explicitly selected reference metrics, with source uncertainty reported separately. No unverified Cd, St, Cp points have been substituted for missing raw reference data.

## Configuration and software usage

Re = U D / nu = 3900; U = rho = D = 1, nu = 1/3900. Domain: 24.0D x 20.0D, cylinder at (8.0D,10.0D); periodic span pi D. Top/bottom boundary: slip. This domain/span is a development choice; sensitivity is still required. Grid 48 x 16 x 8 = 6144 true hexahedral control volumes. Exponential radial clustering; first-cell resolution must be measured, never inferred from grid counts alone.

Shared owner/neighbor faces integrate convection, full symmetric deviatoric variable-viscosity stress and pressure. WALE SGS uses the original [Nicoud and Ducros (1999) formulation](https://doi.org/10.1023/A:1009995426001); Cw = 0.325 and Delta = V^(1/3). Zero SGS face viscosity at the wall. An explicit diffusion screen checks 2 dt sum(nu_f a_f)/V <=0.5 before any step commits. This conservative necessary screen is not a general skew-mesh stability proof; fine wall production may need implicit viscosity. Forward Euler and 90% central / 10% upwind convection are exploratory settings; numerical-dissipation sensitivity remains open. LS gradients and nonorthogonal pressure face projection are implemented in a shared geometry kernel. Cell velocity and face mass flux are separate authoritative quantities; mass closure alone does not validate their collocated coupling.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_body_fitted_cylinder3900.py --output results/cylinder3900-body-fitted --nx 48 --ny 16 --nz 8 --steps 300 --dt .002 --stretching 3.5
PYTHONPATH=src python scripts/report_body_fitted_cylinder3900.py --input results/cylinder3900-body-fitted
```

The `--steps` argument specifies additional steps after initialization, not a total target. Statistical `--discard-time` uses physical time; the minimum record duration uses tU/D. D=U=1 in this case. New runs support atomic checkpoints and exact continuation in a new empty output directory with `--initialize-from previous/checkpoint.npz`. Configuration and producer hashes must match; max_steps may increase. Online means and second moments resume with the state. environment.json records Python/Torch/NumPy/SciPy, platform, thread settings and git base; all imported TensorFVM Python dependencies are hashed and archived at launch. CPU factorization only; CUDA and multi-rank curved-mesh flow not tested.

## Actual calculation

Completed 300 steps, tU/D = 0.6; max final cell divergence = 6.110e-09; final momentum ledger = 2.372e-13; final CFL = 0.04910. These figures audit numerical advancement. Final instantaneous Cd = 0.251999 and Cl = 2.18573e-07 are startup values and are not benchmark statistics. The native geometry, actual pressure/speed maps, force history and actual wall-pressure coefficients appear in report.pdf and PNG/SVG figures.

Measured instantaneous startup y+ max = 16.2384779603138, 95th percentile = 16.090648525417237. Wall-resolved y+<=1 gate: False.

## Reproducible primary DNS scalar comparison

[Lehmkuhl et al. (2013)](https://doi.org/10.1063/1.4818641), [original author PDF from UPC](https://upcommons.upc.edu/server/api/core/bitstreams/adb150ec-33dd-4066-85ce-12c32a230ddc/content), Table I (printed p.085109-5; PDF page6), Table IV (p.085109-18; PDF page19). This is a **numerical DNS reference**, not experimental pressure taps. PDF SHA256, exact source conditions, table locations and source discrepancies are in dns-reference.json. No full copyrighted article is redistributed here.

| Metric | Original fine DNS, 9.3M CV | Actual instantaneous startup | Startup relative difference | Qualified statistic |
|---|---:|---:|---:|---|
| Cd | 1.015 | 0.251999 | 75.17% | unavailable |
| Cpb | -0.935 | 0.508754 | 154.41% | unavailable |
| St | 0.215 | null | null | unavailable |
| Lr/D | 1.36 | not extracted | null | unavailable |

Cp at the downstream base is interpolated from the two adjacent native wall faces around theta=0 and averaged across span; it is still an instantaneous owner-pressure approximation. It needs wall/grid/time refinement and long averaging. The scalar figure compares real startup quantities with actual published scalar values; it intentionally has no computed St marker and no invented Cp curve.

The paper also reports 2piD/18.6M CV Cd=1.019, St=0.214, Cpb=-0.933; its 1.62M CV Cd=1.05 differs from fine DNS by 3.45%. That coarse DNS is not a demonstration of our 3% target. Conditional Mode L/H Cd=[0.979,1.043] is a physical-regime range, **not a statistical confidence interval**. The table supplies no scalar CI; printed rounding half-units are stored separately and are not uncertainty bars. Source slow modulation fD/U=0.0064 corresponds to about156 D/U, so a500 D/U record covers only about3 slow cycles; statistical stationarity can require much longer. Source Table I lists858 cycles whereas its body gives approximately3900 D/U and836 cycles; both original statements are preserved.



## Three distinct qualification levels

| Level | Current status | Required evidence |
|---|---|---|
| Geometry and operators | Tested | Closed face area, positive volume, periodic seam, shared flux cancellation, independently applied pressure CSR, SGS limits |
| Short transient stability | Passed for this run | All step continuity below 1e-8, finite fields, CFL < 1, momentum ledger near roundoff |
| Long-time physical accuracy | **Not qualified** | Startup discard, stationary blocks, >=500 D/U record and >=20 resolved cycles, wall resolution, mesh/time/span/domain sensitivity, verified reference Cd/St/profile errors <=3% |

Production statistics default to discarding 100 D/U then collecting >=500 D/U; these are configurable development gates, not an assertion that the selected durations suffice for all wake modes. Too-short records return `St=null`. Block confidence intervals require sufficiently long independent blocks; the implementation reports that assumption. Cp/velocity reference comparison and uncertainty bands await primary raw data; current figures have no fabricated standard curve.

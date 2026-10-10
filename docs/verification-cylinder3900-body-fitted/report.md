# Three-dimensional body-fitted cylinder, Re = 3900

## Scope and primary literature

This is an actual 3-D body-fitted FV flow development case. It uses a polygonal no-slip cylinder and periodic span, with no Cartesian solid mask. The primary experimental/LES paper is [Parnaudeau et al. (2008)](https://doi.org/10.1063/1.2957018), [original author record](https://hal.science/hal-00383669v1). The authors identify substantial integration-time sensitivity and approximately 10% uncertainty for many near-wake statistics. Our numerical accuracy target remains 3% against explicitly selected reference metrics, with source uncertainty reported separately. No unverified Cd, St, Cp points have been substituted for missing raw reference data.

## Configuration and software usage

Re = U D / nu = 3900; U = rho = D = 1, nu = 1/3900. Domain: 20D x 12D, cylinder at (5D,6D); periodic span pi D. This domain/span is a development choice; sensitivity is still required. Grid 48 x 16 x 8 = 6144 true hexahedral control volumes. Exponential radial clustering; first-cell resolution must be measured, never inferred from grid counts alone.

Shared owner/neighbor faces integrate convection, full symmetric deviatoric variable-viscosity stress and pressure. WALE SGS uses the original [Nicoud and Ducros (1999) formulation](https://doi.org/10.1023/A:1009995426001); Cw = 0.325 and Delta = V^(1/3). Zero SGS face viscosity at the wall. Forward Euler and 90% central / 10% upwind convection are exploratory settings; numerical-dissipation sensitivity remains open. LS gradients and nonorthogonal pressure face projection are implemented in a shared geometry kernel. Cell velocity and face mass flux are separate authoritative quantities; mass closure alone does not validate their collocated coupling.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_body_fitted_cylinder3900.py --output results/cylinder3900-body-fitted --nx 48 --ny 16 --nz 8 --steps 300 --dt .002 --stretching 3.5
PYTHONPATH=src python scripts/report_body_fitted_cylinder3900.py --input results/cylinder3900-body-fitted
```

New runs support atomic checkpoints and exact continuation in a new empty output directory with `--initialize-from previous/checkpoint.npz`. Configuration and producer hashes must match; max_steps may increase. Online means and second moments resume with the state. CPU factorization only; CUDA and multi-rank curved-mesh flow not tested.

## Actual calculation

Completed 300 steps, tU/D = 0.6; max final cell divergence = 7.059e-09; final momentum ledger = 8.422e-14; final CFL = 0.04423. These figures audit numerical advancement. Final instantaneous Cd = 0.27851 and Cl = 2.31566e-07 are startup values and are not benchmark statistics. The native geometry, actual pressure/speed maps, force history and actual wall-pressure coefficients appear in report.pdf and PNG/SVG figures.

Measured instantaneous startup y+ max = 12.442267104626085, 95th percentile = 12.284028318531387. Wall-resolved y+<=1 gate: False.

## Three distinct qualification levels

| Level | Current status | Required evidence |
|---|---|---|
| Geometry and operators | Tested | Closed face area, positive volume, periodic seam, shared flux cancellation, independently applied pressure CSR, SGS limits |
| Short transient stability | Passed for this run | All step continuity below 1e-8, finite fields, CFL < 1, momentum ledger near roundoff |
| Long-time physical accuracy | **Not qualified** | Startup discard, stationary blocks, >=500 D/U record and >=20 resolved cycles, wall resolution, mesh/time/span/domain sensitivity, verified reference Cd/St/profile errors <=3% |

Production statistics default to discarding 100 D/U then collecting >=500 D/U; these are configurable development gates, not an assertion that the selected durations suffice for all wake modes. Too-short records return `St=null`. Block confidence intervals require sufficiently long independent blocks; the implementation reports that assumption. Cp/velocity reference comparison and uncertainty bands await primary raw data; current figures have no fabricated standard curve.

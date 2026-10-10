# Re3900 wall refinement and startup development report

## Geometry and initial-viscosity preflight

All eight meshes were actually built with the shared 3-D native face metrics, LS gradient and WALE initial viscosity. No global pressure CSR or LU was constructed in the preflight. The startup velocity is not projected; its y+ indicator is not a long-time wall-resolution qualification. Geometry gates use the common `tensorfvm.verification.core.Metric/evaluate` module.

| Grid | Stretch | Wall distance max / D | Initial y+ max | Delta z / D | Diffusion dt ceiling | Wake h95 / D |
|---|---:|---:|---:|---:|---:|---:|
| 128 x 64 x 24 | 6.0 | 0.00222 | 2.245 | 0.1309 | 0.000202 | 0.132 |
| 128 x 64 x 24 | 7.0 | 0.000959 | 1.541 | 0.1309 | 3.05e-05 | 0.129 |
| 128 x 64 x 24 | 8.0 | 0.000406 | 1.055 | 0.1309 | 4.32e-06 | 0.122 |
| 192 x 96 x 32 | 7.0 | 0.00063 | 1.221 | 0.0982 | 1.68e-05 | 0.088 |
| 192 x 96 x 32 | 8.0 | 0.000266 | 0.838 | 0.0982 | 2.38e-06 | 0.085 |
| 64 x 32 x 12 | 5.5 | 0.00695 | 4.015 | 0.2618 | 0.00129 | 0.265 |
| 64 x 32 x 12 | 7.0 | 0.00201 | 2.318 | 0.2618 | 7.99e-05 | 0.245 |
| 64 x 32 x 12 | 8.0 | 0.000858 | 1.503 | 0.2618 | 1.15e-05 | 0.245 |

192 x 96 x 32 / stretch8 gives initial y+max=0.838, but only an initial wall indicator. Its Delta z=0.0982D remains four times the [primary DNS](https://doi.org/10.1063/1.4818641) 128-plane spacing of pi/128=0.02454D; wake volume-length h95=0.085D is also much larger than the paper’s reported near-wake average h about0.018D. A thin first wall cell alone does not establish accurate wake LES.

The geometry run peak RSS was a cumulative process peak of about2.41 GB; the 589824-CV native stored tensors are recorded separately. These are measured geometry/transport costs, not pressure-solver costs. No large-grid LU timing or memory result is claimed.

## Matched-time two-grid short runs

Both real CFD runs use the original DNS domain [-8,16]D x [-10,10]D, span piD, inletU=1, top/bottom slip, cylinder no-slip and periodic z. Both take50 steps at dt=1e-5 to tU/D=0.0005. They ran concurrently with other CPU jobs, so timings are observed under load rather than isolated scaling.

| Case | CV | Final maxdiv | Final y+max | Seconds/step | LU stored MB | Setup seconds |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 6144 | 5.49e-09 | 16.610 | 0.150 | 25.03 | 0.274 |
| medium | 24576 | 5.15e-09 | 2.814 | 0.705 | 245.62 | 4.365 |

The medium pressure matrix has170496 entries, while its actual L/U factors have20452054 entries and245.62MB of stored sparse arrays. Four times the CV count produces about9.81 times the factor storage. Measured medium process peak RSS is about1.16GB. At its observed0.705s/step, an extrapolation of600D/U at the unchanged dt1e-5 needs60million steps and about489.9days. **This is an arithmetic cost extrapolation, not a measured long run**; solver iterations, CPU load and future numerical methods will change costs.

Initial geometry / stretch8 for the589824-CV grid limits diffusion dt to2.38e-6; using20% of that ceiling is about4.75e-7 and implies roughly1.26billion steps over600D/U. No per-step runtime was measured on that grid. Explicit viscosity and full 3-D sparse LU are concrete production bottlenecks; escalating only mesh counts is insufficient.

The common acceptance audit replays final continuity independently from native checkpoint face fluxes and checks all recorded mass/momentum/CFL/diffusion limits. Both numerical audits pass. Both instantaneous wall-resolution gates fail. Physical accuracy remains false, qualified Cd/St/Cpb/profile errors remain null, and these two startup solutions do not establish a convergence order.

## Cell/face coupling diagnostic

| Case | Interior max normal defect / U | p95 normal defect / U | Relative face-flux L2 defect |
|---|---:|---:|---:|
| coarse | 0.00241023 | 0.00080903 | 0.000117818 |
| medium | 0.00233984 | 0.00159459 | 9.62353e-05 |

These compare native projected mass flux to interpolation of the current actual cell velocity, with actual no-slip/slip/inlet/outlet values. They are diagnostic, not new acceptance thresholds. Small face continuity does not prove collocated cell/face physical coupling; that still needs independent pressure/momentum and refinement verification.

## Spectral-resolution fix

The old4096-sample Welch ceiling at dt=.002 gives DeltaSt=0.12207 and cannot support a3% target aroundSt=.215. The new method selects the actual segment length from the target resolution and available record; no zero padding. A synthetic St=.215 signal with dt=.002 and601D/U returns St=0.216075, DeltaSt=0.003225, segment samples=155039, segment time=310.078. This is an operator test, not a CFD result. Short insufficient-resolution and constant-lift records are blocked. Duration, spectral eligibility and physical accuracy are distinct.

Latest statistics also use a Student-t block-mean interval with explicit degrees of freedom, block sample counts and durations; independence and stationarity assumptions remain unverified. The CFD producers are preserved exactly in their snapshots; the later statistical postprocessing/doc correction is independently bound by this report source snapshot and is not substituted into older producer manifests.

## Software commands and next work

```bash
PYTHONPATH=src python scripts/probe_body_fitted_cylinder3900.py --output results/cylinder3900-wall-preflight
PYTHONPATH=src python scripts/run_body_fitted_cylinder3900.py --output results/cylinder3900-stage2-medium --nx 64 --ny 32 --nz 12 --stretching 7 --dt .00001 --steps 50
PYTHONPATH=src python scripts/audit_body_fitted_cylinder3900.py --input results/cylinder3900-stage2-medium
PYTHONPATH=src python scripts/report_cylinder3900_development.py --input docs/verification-cylinder3900-wall-refinement
```

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. --steps means additional steps, with exact source/configuration restart validation. The large preflight runs do not solve CFD. Next: implicit viscosity with its own response/stability tests; scalable pressure solver (periodic-extrusion Fourier separation or iterative preconditioning); verified collocated pressure/velocity coupling; near-wake/span refinement; then long stationary statistics and actual primary DNS/experimental comparisons <=3%. GPU is untested and is not expanded in this phase.

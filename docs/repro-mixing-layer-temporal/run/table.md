# Machine-generated comparison table

All values are produced by `scripts/verify_mixing_layer.py`; nothing is hand-typed.

| item | reference | ref | got | rel. error | verdict |
|---|---|---:|---:|---:|---|
| linear growth rate, alpha=0.5, Re=500, instantaneous layer scale | Orr--Sommerfeld eigen-solve (independent discretisation) | 0.18372 | 0.18224 | 2.087% | PASS (tol 10.0%) |
| linear growth rate, alpha=0.5, Re=500, frozen initial profile | Orr--Sommerfeld eigen-solve (independent discretisation) | 0.18372 | 0.18224 | -0.803% | PASS (tol 10.0%) |
| inviscid limit of the Orr--Sommerfeld solve (Re=1e5) | Rayleigh eigen-solve | 0.18728 | 0.18726 | 0.010% | PASS (tol 1.0%) |
| momentum thickness during the diffusion phase | exact heat-kernel solution of the same initial profile | - | - | 0.000% | PASS (tol 5.0%) |
| mean velocity profile during the diffusion phase | exact heat-kernel solution of the same initial profile | - | - | 0.001% | PASS (tol 5.0%) |
| Rayleigh grid refinement (301 -> 601 nodes) | self-convergence of the eigen-solve | 0.18728 | 0.18730 | 0.014% | PASS (tol 0.5%) |
| nonlinear spreading rate dtheta/dt, production vs coarse grid | self-convergence over the grid/time-step ladder | 0.01658 (coarse) | 0.01665 (production) | 0.42% | PASS (tol 10%) |
| self-similar mean-profile collapse | profile at t=22.5 vs t=47.5 | - | - | 0.0227 (max deviation) | PASS (tol 0.0500) |

Nonlinear spreading measured on the growth window tU/delta in [10.0, 28.0] (the layer saturates near tU/delta ~ 28 in the single-wavelength box and then oscillates).

Production case: 96x256 cells, dt=0.0249 (CFL 0.4), Re_delta=500, alpha*delta=0.5, linear seed 1e-05, nonlinear seed 2e-02.

External literature anchor (published spreading / entrainment rate): **WITHHELD** -- the offline work cell cannot reach the reference text (no internet egress; the platform knowledge engine at :8021 refused connections), so no literature number is quoted or transcribed here.

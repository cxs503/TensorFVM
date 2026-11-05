# Taylor–Green vortex: frozen-time reference for the native hybrid 3-D backend

Status: **spatial operator qualified, time marching and pressure withheld** (2026-11-04).

The periodic MAC prototype is time marched against the decaying Taylor–Green
vortex (see `scripts/verification-taylor-green-mac/`). The native hybrid 3-D
backend (`tensorfvm.native_hybrid_ns3d.NativeHybridNavierStokes3D`) is a *steady*
solver — it has no transient integrator — so the same physical benchmark cannot
be run as a decay ladder there. This reference closes the verification gap in the
way the backend actually admits: the exact solution is frozen at one instant and
the time term is supplied as a manufactured body force.

Exact solution on the periodic box $[0,2\pi]^3$ with $\rho=1$:

$$u=A\cos x\sin y\,e^{-2\nu t},\quad v=-A\sin x\cos y\,e^{-2\nu t},\quad w=0,\quad
p=-\tfrac{A^2}{4}(\cos 2x+\cos 2y)\,e^{-4\nu t}.$$

Because $\partial u/\partial t=-2\nu u$, freezing the field at $t$ turns the
unsteady balance into the steady problem

$$(u\cdot\nabla)u=-\nabla p+\nu\nabla^2 u+f,\qquad f=-\partial u/\partial t=2\nu u,$$

which the frozen field satisfies identically. The Dirichlet trace on the six
faces of the period cell is imposed as exact face area means
(`exact_boundary_velocity_means`); the box is one period cell, so the trace is
mutually compatible and its total normal flux balances to zero (the solver's
closed-boundary gauge check passes).

## Implementation

| item | path |
|---|---|
| reference + solver + metrics | `src/tensorfvm/verification/native_taylor_green.py` |
| tests | `tests/test_native_taylor_green.py` |
| study runner / record | `scripts/verification-native-taylor-green/{run_study.py,study.json,RESULTS.md}` |

The reference layer verifies itself before it judges the solver: continuity in
all three directions, the unsteady momentum balance, the frozen steady balance
with $f=2\nu u$, and the identity $f=-\partial u/\partial t$ are all checked by
independent finite differences.

## Published result ($A=1$, $\nu=1$, frozen $t=0.5$ s)

| n | velocity L2 | order | max div | max \|w\| | pressure Linf |
|---|---|---|---|---|---|
| 4 | 3.960e-01 | — | 3.4e-16 | 4.8e-04 | 3.375e+13 |
| 6 | 1.876e-01 | 1.84 | 5.4e-16 | 4.5e-04 | 5.675e-01 |
| 8 | 1.078e-01 | 1.93 | 9.9e-16 | 6.0e-04 | 1.083e+00 |

Regenerate with `PYTHONPATH=src python scripts/verification-native-taylor-green/run_study.py`.

## What is and is not verified

* **Verified**: the discrete velocity reproduces the frozen exact velocity at
  observed order 1.93 (band 1.5–2.5), the accepted face flux is divergence-free at
  round-off, and the $z$ direction stays inactive to $6\times10^{-4}$ (exact $w=0$).
* **Withheld — pressure**: in this closed all-Dirichlet-velocity configuration the
  cell-centred pressure keeps an odd–even mode that the discrete system does not
  determine. Measured pressure error does not converge under refinement (0.57 at
  n=6, 1.08 at n=8) and on the coarsest mesh the linear solve amplifies that mode to
  $10^{13}$ even though velocity and divergence are correct. The gate is recorded as
  failed with this reason; the pressure content that drives the flow is exercised
  only indirectly, through the converged velocity.
* **Withheld — time marching**: no transient integrator exists in this backend, so
  the decaying-vortex ladder (energy decay rate, temporal order) remains covered by
  the periodic MAC kernel only.
* **Viscosity range**: the frozen steady problem stiffens as $\nu\to0$; the physical
  MAC case $\nu=0.05$ ($Re=125.7$) does not converge in 25–30 Newton iterations, so
  the published study uses $\nu=1$.
* **Cross-backend gap**: at n=8 the time-marched MAC kernel reaches
  `velocity_relative_l2 = 5.05e-03` on the same physical case, i.e. ~21× smaller than
  the native hybrid operator's 1.08e-01 at $\nu=1$. The constants are not directly
  comparable (different viscosity, different formulation), but the native hybrid
  cell/face operator is markedly more diffusive at the same order — see
  `cross_backend_reference` in `study.json`.
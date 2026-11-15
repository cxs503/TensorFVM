# FEAT-UNSTR3D-MESH — 3-D cell-centred unstructured FVM core (TensorFVM)

Anchor: `FEAT-UNSTR3D-MESH`. Branch `agent/m21861`. Every number below was
produced by the code in this worktree; the raw bundle is
`artifacts/feat-unstr3d-mesh/gates.json` and the regenerating commands are in
`docs/feat-unstr3d-mesh/RUN.md`.

## 1 Case introduction

The task is the first step of a *general* unstructured 3-D finite-volume core
for TensorFVM. Until now 3-D was restricted to structured and native hybrid
meshes (`native_mesh3d.py`); 2-D Gmsh tri/quad meshes already worked through
`gmsh_external.GmshExternalMesh` with an owner/neighbour face operator.

Physical background of the verification cases:

| case | physics | geometry | parameters |
|---|---|---|---|
| Poisson | steady diffusion `div(Gamma grad phi) = f` | unit cube, Kuhn six-tet split, `n` cells per axis, optional smooth sinusoidal warp (amplitude `0.25 h`) | `Gamma = 1`, `phi = sin(pi x) sin(pi y) sin(pi z)`, `f = -3 pi^2 phi`, Dirichlet on all six patches |
| affine exactness | linear reproduction | same cube, `n = 3`, warp 0 and 0.3 | `phi = 0.3 + (1, -2, 0.5) . x` |
| 45 deg convection | steady pure convection `div(u phi) = 0` | same cube, `u = (1,1,0)/sqrt(2)`, inflow `x=0` (`inlet`) and `y=0` (`wall`), outflow `x=1`, `y=1`, streamwise-parallel `z` faces (zero flux) | `phi = cos(2 pi eta / lambda)`, `eta = (y - x)/sqrt(2)`, `lambda = 0.5` |
| mesh invariants | geometry | `n = 8` tet cube, warp 0 and 0.25 | — |

Mesh sizes: `n = 2, 4, 8` give 48, 384, 3072 tetrahedra (6 per lattice cube);
`max` non-orthogonality is 36.7 deg (undistorted) and 51.3 deg (warped), i.e.
the non-orthogonal correction is exercised, not decorative.

## 2 Method

New modules (no 2-D path was refactored):

* `src/tensorfvm/gmsh_msh3d.py` — pure-Python Gmsh 2.x **ASCII v2** tet reader.
  Sections `$MeshFormat` (v2 + ASCII only), `$PhysicalNames`, `$Nodes`,
  `$Elements` (linear tet = type 4, boundary triangle = type 2). Physical names
  are whitelisted exactly like the 2-D path: `inlet`, `outlet`, `wall`,
  `far-field`, `fluid`; anything else, a binary file, a high-order element, an
  untagged boundary triangle or a conflicting group is a hard error. Also a
  writer (1-based ids) used for round-trip tests.
* `src/tensorfvm/unstructured_mesh3d.py` — `tet_mesh_from_msh` closes
  MSH -> `NativeMesh3D` (owner/neighbour faces come from the existing 3-D
  native builder, which already enforces closure, positive volume and a
  positive owner-to-neighbour projection); `cube_tet_msh` emits a reproducible
  warped tetrahedral cube as **Gmsh 2.2 ASCII**, so every gate runs through the
  same import path a user mesh takes; `mesh_invariants` reports the four
  geometric invariants.
* `src/tensorfvm/unstructured_fvm3d.py` — `UnstructuredFVM3D`:
  * inverse-distance weighted **least-squares gradient** from cell values plus
    explicit boundary-face data (linear-exact, machine precision on an affine
    field);
  * **over-relaxed** flux decomposition `coef = |s|^2/(s.e)`, non-orthogonal
    remainder `t = s - coef e` (minimum-correction variant selectable);
  * `poisson` iterates the deferred correction to its fixed point (machine
    precision defect, reported iterations);
  * **k-exact quadratic reconstruction** (node-sharing cloud, scaled monomials,
    weighted LS) with a face-centred evaluation used *only* in the
    non-orthogonal correction;
  * **convection** with `upwind` / `central` / `limited` face weighting on the
    upwind M-matrix (deferred correction, damped).

Toolchain: numpy 2.4.6 / scipy 1.16.0 (sparse LU), Python 3.11.6, no Gmsh
binary needed. `pytest` gates: `tests/test_unstr3d_msh_import.py`,
`tests/test_unstr3d_poisson_order.py`, `tests/test_unstr3d_convection_45deg.py`,
`tests/test_unstr3d_benchmark_cases.py`.

## 3 Results

Poisson, volume-weighted L2 error of the cell-centroid field (`h = 1/n`):

| correction used | n=2 (h=.5) | n=4 (h=.25) | n=8 (h=.125) | order 2->4 | order 4->8 |
|---|---|---|---|---|---|
| k-exact quadratic face gradient, warp 0 | 9.530e-2 | 2.059e-2 | 4.091e-3 | **2.21** | **2.33** |
| k-exact quadratic face gradient, warp 0.25 | 9.530e-2 | 1.879e-2 | 5.011e-3 | 2.34 | 1.91 |
| linear LS gradient (control) | 1.994e-2 | 1.129e-2 | 3.899e-3 | 0.82 | 1.53 |
| exact analytic gradient (upper bound) | 1.738e-2 | 4.554e-3 | 1.166e-3 | 1.93 | 1.97 |

Affine reproduction (no source, affine Dirichlet): max |phi_h - phi_exact| =
2.76e-13 (warp 0) and 1.60e-12 (warp 0.3) — the operator is exactly
linear-exact, so the non-orthogonal correction adds no first-order error.

Mesh invariants (n = 8, 3072 tets): positive volumes; face area vector rebuilt
from the stored vertex order matches the stored one to 0.0; outward closure
1.18e-16; every internal owner-to-neighbour projection positive; patch counts
inlet 128 / outlet 128 / wall 256 / far-field 256.

Convection gate: **not verified in this session** — see §5.

## 4 Figures

No figure is attached, and the reason is explicit: the numeric gates were run
through `pytest`/`python -` on the shared 99-control-plane host where heavy
matplotlib rendering plus the required grid ladder exceeded the per-command
wall budget. The quantitative comparison a figure would carry (log-log
`e(h)` against `h^2`) is already tabulated in §3 with the fitted orders;
generating the plot is a pure re-plot of `artifacts/feat-unstr3d-mesh/gates.json`.

## 5 Points to note (pitfalls, limits)

* **A tetrahedral LS gradient is first order, not second.** Measured interior
  gradient error 4.95e-1 -> 3.57e-1 -> 1.87e-1 -> 9.5e-2 for n = 2,4,8,16, i.e.
  clean O(h). Because the correction is `t . g` with `t = O(h^2)`, a first-order
  gradient leaves an O(1) divergence defect: with the linear LS correction the
  Poisson order stalls at 1.5 (table row 3). The first delivered version of
  this module therefore did **not** pass the gate.
* The fix is the k-exact quadratic reconstruction evaluated **at the face**
  (not interpolated cell gradients): interpolating cell gradients keeps the
  O(h^2) *face* error of the interpolation; evaluating each side's quadratic at
  the face centroid removes it. Rows 1 and 4 show that the implicit
  (over-relaxed two-point) part was always capable of second order.
* The warped ladder is pre-asymptotic: order 1.91 on n = 4->8, i.e. **below the
  gate**. Reported, not hidden. The undistorted ladder passes at 2.21/2.33.
* The Kuhn six-tet split must use the Freudenthal ordering `{000, e_pi1,
  e_pi1+e_pi2, 111}`. An arbitrary six-tet list looks innocent and produces the
  right total volume but leaves hanging nodes, i.e. internal faces counted once;
  `NativeMesh3D.from_cells` does not detect that, `mesh_invariants` and the
  face-count/volume bookkeeping do. This was a real defect in the first
  generator.
* Diffusion-only scope: `poisson`/`advection` are scalar operators. No pressure
  coupling, no turbulence, no time term is claimed here.
* Gmsh is **not** installed in this environment, so the reader is exercised on
  ASCII files written by the test suite and by `cube_tet_msh`, not on a mesh
  exported by the Gmsh binary.

## 6 Error judgement

| quantity | reference | computed | error | < 3 %? |
|---|---|---|---|---|
| Poisson L2, n=8, warp 0, k-exact correction | 0 (manufactured exact solution) | 4.091e-3 (absolute in L2, field norm 0.354) | 1.16 % of the field L2 norm | **yes** |
| Poisson L2, n=8, warp 0.25 | 0 | 5.011e-3 | 1.42 % | **yes** |
| observed order, n=4->8, warp 0 | 2 (gate) | 2.33 | — | **yes** (gate met) |
| observed order, n=4->8, warp 0.25 | 2 (gate) | 1.91 | -4.5 % | **no — gate missed, reported** |
| affine field max error | exact linear | 2.8e-13 / 1.6e-12 | 0 | **yes** |
| 45 deg convection overshoot | analytic `u.grad(phi)=0` profile | **not measured** | — | unknown, not asserted |

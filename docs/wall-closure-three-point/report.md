# `wall_closure='three_point'` — reviving a dead closure and measuring what it is worth

*Repository:* TensorFVM · *Operator:* `src/tensorfvm/native_viscous3d.py`
(`NativeViscous3D`, the shared-face viscous traction assembled by the collocated / RANS /
SST 3-D solvers) · *Script:* `scripts/verify_three_point_wall_closure.py` ·
*Data:* `docs/wall-closure-three-point/three_point_wall_closure.json`, `docs/wall-closure-three-point/three_point_wall_closure.png` ·
*Tests:* `tests/test_native_wall_closure_three_point.py` (13 cases)

## 1. Case and motivation

`docs/native_viscous_dirichlet_order.md` records an open defect: the Dirichlet
wall traction of the native 3-D viscous operator is **first order** and the
wall-adjacent cell divergence of `div(tau(u))` is **O(1)**. The module has
carried a `wall_closure='three_point'` option documented to fix exactly this
("exact for affine fields, second order for smooth fields"), but no test, script
or document ever exercised it — the default stayed `two_point` and the claim was
unverified.

`native_viscous3d.py` is the operator behind every wall-resolved native
traction, so an unverified closure is a verification debt with a direct
consequence for wall shear, `Cf` and wall-normal resolution studies.

## 2. Method

1. **Audit the option.** Instantiate both closures on the same `n^3` box used by
   `tensorfvm.verification.native_order3d` and read `wall_second_order_faces`.
2. **Repair the closure** so that it actually engages (three defects below).
3. **Qualify it against analytic fields**, never against itself:
   * *affine field* `u = base + G.x` — the traction must be exact to round-off
     (this is the only test that catches a matrix/source inconsistency inside
     the stencil);
   * *quadratic field* `u_x = x^2` — the analytic wall normal derivative is
     `2 x n_x`, so the closure can be scored pointwise;
   * *manufactured trig field* — `truncation_study` / `face_traction_study`
     already measure the wall-cell divergence and the wall normal-derivative
     error on `n = 4, 8`;
   * *interior control* — the interior divergence one must not move at all.

## 3. Defects found (all three made the closure silently equal to `two_point`)

| # | defect | effect |
|---|---|---|
| 1 | alignment search used `align = +sig*(g_n . n_f)` and required `align >= 0.5`, i.e. it looked for an internal face whose normal points *the same way* as the outward wall normal | on a Cartesian box the best candidate is anti-parallel (`align = -1`), so **no face** was ever upgraded: `wall_second_order_faces = 0` |
| 2 | one-sided distances were taken along the **outward** normal, `delta = (x_cell - x_face).n_f < 0` | every candidate was rejected by the `delta0 > 0` guard, again with no upgraded face |
| 3 | the sparse row replacement built `drop` with 3 entries per face (`repeat(fb,3)`) while a normal-derivative row has `N` columns | `ValueError: all index and data arrays must have the same length` — unreachable only because of defects 1 and 2 |

A fourth, subtler defect appears once the closure engages: **triangulated
boundary facets** put the Dirichlet datum at the *triangle centroid*, off the
normal line through the owner centre. The plain normal-projection stencil then
is not even affine exact (measured affine traction error `4.8e-1`). The repair
carries the tangential offset `e_i = (x_i - x_face) - delta_i*u_inward` and adds
the term `w . grad_t` with `w = -(c1*e_0 + c2*e_1)`, evaluated with the
tangential part of the two-point face gradient, on **both** the matrix and the
boundary-data side.

After the repair all Dirichlet faces are upgraded (`192 / 768` at `n = 4 / 8`),
and the residual is exact for affine data again.

## 4. Results

Mesh: unit box, `n^3` hexahedra, triangulated boundary facets, `mu = 1.3`,
`warp = 0` (Cartesian) and `warp = 2` (interior-node warp of the same topology).

| closure | warp | n | upgraded faces | affine traction err | quadratic wall `du/dn` err | `l2_wall` | `l2_interior` | rel. wall `du/dn` (trig) |
|---|---|---|---|---|---|---|---|---|
| two_point | 0 | 4 | 0 | 3.7e-16 | 1.286e-1 | 3.348 | 1.2042 | 6.192e-2 |
| two_point | 0 | 8 | 0 | 2.6e-16 | 6.431e-2 | 3.342 | 0.2803 | 2.757e-2 |
| three_point | 0 | 4 | 192 | 7.1e-16 | **5.233e-2** | 3.693 | 1.2042 | 7.712e-2 |
| three_point | 0 | 8 | 768 | 3.2e-16 | **2.617e-2** | 3.379 | 0.2803 | 2.768e-2 |
| two_point | 2 | 4 | 0 | 5.1e-16 | 1.360e-1 | 3.331 | 1.1947 | 1.013e-1 |
| two_point | 2 | 8 | 0 | 4.2e-16 | 6.642e-2 | 3.129 | 0.3358 | 6.166e-2 |
| three_point | 2 | 4 | 192 | 6.0e-16 | 1.291e-1 | 3.918 | 1.1947 | 1.703e-1 |
| three_point | 2 | 8 | 768 | 7.3e-16 | 7.154e-2 | 3.333 | 0.3358 | 7.527e-2 |

Observed orders (n = 4 -> 8):

| quantity | two_point (warp 0 / 2) | three_point (warp 0 / 2) |
|---|---|---|
| wall normal derivative, trig field | 1.17 / 0.72 | 1.48 / 1.18 |
| quadratic wall normal derivative | 1.00 / 1.03 | 1.00 / 0.85 |
| wall-cell divergence `l2_wall` | 0.00 / 0.09 | 0.13 / 0.23 |
| interior divergence `l2_interior` | 2.10 / 1.83 | 2.10 / 1.83 (bit-identical) |

## 5. Interpretation

* The closure is now **real**: it engages on every Dirichlet face and stays
  affine-exact at round-off, and it leaves the interior operator bit-identical.
* Where it targets the wall normal derivative, the *quadratic* benchmark
  improves by a **factor 2.4** at both grids (5.23e-2 vs 1.29e-1 at `n = 4`,
  2.62e-2 vs 6.43e-2 at `n = 8`). This is a systematic error reduction, not a
  lucky grid.
* It does **not** reach second order (1.2-1.5): the offset term is fed by the
  two-point tangential gradient, whose wall error is `O(h)` while the weight
  `w` is `O(1)`.
* It does **not** repay the O(1) wall-cell divergence. This is consistent with
  the earlier independent attribution in `docs/native_viscous_dirichlet_order.md`
  and `tests/test_wall_cell_gradient_attribution.py`: replacing the wall normal
  derivative by its analytic value changes the wall-cell divergence by only a
  few percent, because the divergence amplifies the whole wall-face gradient
  error by `1/h` and the *internal* faces of a wall-adjacent cell keep the
  as-built gradient.
* On the warped mesh at `n = 4` the closure is *worse* than the default
  (1.70e-1 vs 1.01e-1): the second cell is chosen by best alignment, which on a
  strongly warped box is not collinear with the normal, so the one-dimensional
  stencil is fed a transverse curvature it does not model.

## 6. Error judgement against the acceptance gates

| key quantity | reference | deviation | < 3 % ? |
|---|---|---|---|
| affine traction, both closures and warps | analytic `mu s.grad` | <= 7.1e-16 absolute | **yes** |
| quadratic wall `du/dn`, `three_point`, n = 8 | analytic `2 x n_x` | 2.62e-2 absolute on a field of scale 2 (1.3 % of scale) | **yes** |
| same, `two_point`, n = 8 | analytic | 6.43e-2 (3.2 % of scale) | **no** |
| wall-cell `div(tau)`, `three_point`, n = 8 | analytic `div(tau)` | 3.379 absolute vs field scale 8.26 (41 %) | **no** |
| observed order of the wall normal derivative | expected 2 | 1.48 (warp 0) | **no** |

The closure is therefore an honest improvement, not a closure of the recorded
debt: `docs/native_viscous_dirichlet_order.md` still stands, and the default
remains `two_point` because the three-point path is measurably worse on warped
meshes and no production solver uses it yet.

## 7. Notes, limitations, reproduction

* `wall_closure='three_point'` affects the **traction** (momentum flux) path
  only; the scalar diffusion gradient keeps the two-point closure.
* The closure is only a candidate for production once the tangential gradient
  that feeds its offset term is itself second order at the wall (that is the
  next unit of work, not a tuning knob).
* Do not enable it on warped meshes without re-checking §4 row 7: the
  alignment criterion must be replaced by a least-squares stencil over the
  available neighbour centres.
* Reproduce:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python scripts/verify_three_point_wall_closure.py --levels 4 8 --warps 0 2 --plot
python -m pytest tests/test_native_wall_closure_three_point.py -q
```
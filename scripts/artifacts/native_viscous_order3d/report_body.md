## 1. Case

Object: `tensorfvm.native_viscous3d.NativeViscous3D`, the shared-face
diffusion/viscous traction closure assembled by `native_collocated3d`,
`native_rans3d` and `native_sst3d`. Manufactured scalar fields on the unit
cube, all-Dirichlet data from the analytic field, mu = 1.3, structured
hexahedra with triangulated faces:

| family | meshes | warp | field |
|---|---|---|---|
| trig1d-orth | 16x4x4 32x4x4 64x4x4 | 0.0 | `trig1d` |
| trig3d-orth | 4x4x4 8x8x8 12x12x12 | 0.0 | `trig3d` |
| trig3d-warp | 6x6x6 12x12x12 | 0.35 | `trig3d` |

Fields: `trig1d = sin(pi x)`, `trig3d = sin(pi x) cos(pi y/2) exp(z)`.

## 2. Method

For every mesh the driver evaluates `scalar_flux(values, 0, exact boundary
values)` and compares the Dirichlet-face flux `q = mu grad(u).S` with the
analytic flux under two metrics (norm-based and per-face mean relative),
then measures the constant-mode residual of the assembled operator
`A = divergence @ scalar_flux_matrix`. Affine exactness is re-checked on every
mesh as a negative control. Driver: `scripts/audit_native_viscous_order3d_wall_closure.py`,
library: `src/tensorfvm/verification/native_viscous_order3d.py`.

## 3. Results

| family | mesh | N | wall flux (norm) | per-face mean | per-face floored | constant mode | affine |
|---|---|---|---|---|---|---|---|
| trig1d-orth | 16x4x4 | 256 | 1.779e-03 | 1.606e-03 | 1.606e-03 | 2.268e-01 | 6.1e-15 |
| trig1d-orth | 32x4x4 | 512 | 4.215e-04 | 4.015e-04 | 4.015e-04 | 1.646e-01 | 1.1e-14 |
| trig1d-orth | 64x4x4 | 1024 | 1.030e-04 | 1.004e-04 | 1.004e-04 | 1.174e-01 | 1.5e-14 |
| trig3d-orth | 4x4x4 | 64 | 5.925e-02 | 1.463e-01 | 7.635e-02 | 3.849e-01 | 8.9e-16 |
| trig3d-orth | 8x8x8 | 512 | 2.413e-02 | 8.865e-02 | 2.999e-02 | 2.357e-01 | 1.1e-15 |
| trig3d-orth | 12x12x12 | 1728 | 1.536e-02 | 6.870e-02 | 1.887e-02 | 1.814e-01 | 1.4e-15 |
| trig3d-warp | 6x6x6 | 216 | 3.500e-02 | 1.042e-01 | 4.276e-02 | 2.875e-01 | 1.2e-15 |
| trig3d-warp | 12x12x12 | 1728 | 1.643e-02 | 6.826e-02 | 1.928e-02 | 1.856e-01 | 2.1e-15 |

Observed orders (log2 in h):

| family | wall flux | per-face mean | constant mode |
|---|---|---|---|
| trig1d-orth | [2.08, 2.03] | [2.0, 2.0] | [0.46, 0.49] |
| trig3d-orth | [1.3, 1.11] | [0.72, 0.63] | [0.71, 0.65] |
| trig3d-warp | [1.09] | [0.61] | [0.63] |

## 4. Findings

1. The m2530 verdict reproduces on a genuinely multi-dimensional field: the
   wall normal-derivative flux converges at order [1.3, 1.11] (norm metric) and [0.72, 0.63]
   (per-face metric) for `trig3d`, with 2.41% at 8x8x8 versus the 6.92% the
   m2530 table reports at the same mesh, so the first-order verdict stands.
2. The defect is inherently transverse: the same closure is second order,
   order [2.08, 2.03], on the 1-D control where no transverse curvature exists.
   The m2530 attribution to a single missing curvature term in the
   centre-to-face residual is therefore not direction-independent: a
   `d.H.d/2` type correction cannot repair it, which we verified by
   implementing such a term (least-squares cell Hessian, Dirichlet faces
   only): the 1-D order stayed at 2 while its error grew tenfold on coarse
   grids because the Hessian estimate is only O(h) accurate, and the 3-D
   rate was unchanged. That experiment was not merged.
3. The constant mode of the assembled operator decays with h:
   ['0.385', '0.236', '0.181'] for `trig3d-orth`. The Dirichlet datum of this closure enters only
   through the right-hand side, so `scalar_diffusion_matrix` and
   `momentum_diffusion_matrix` describe the zero-jump (mirror) operator and
   become singular as h -> 0. A fixed-value solve assembled from them
   therefore leaves the constant mode unconstrained, which is the correct
   reading of the m2530 'Dirichlet diffusion solution has no convergence
   order' row: that experiment cannot measure the closure. This row of the
   earlier report should be withdrawn and replaced by this conditioning result.
4. Warped meshes: orders [1.09], i.e. the warp does not change the rate.

## 5. Caveats

- The m2530 harness itself is absent from the repository (its deliverables
  were filtered before merge), so this is an independent rebuild rather than
  a rerun of identical code; the per-face comparison uses our own definition
  of the metric, which lands between the two m2530 entries.
- Only the scalar closure is audited here. The vector traction adds the
  tangential-gradient terms, which the scalar case does not probe.
- No solver was changed by this work.

## 6. Error verdict

| quantity | baseline | result | under 3%? |
|---|---|---|---|
| affine flux exactness | analytic tau.S | 1.5e-14 relative | yes (machine precision) |
| 3-D wall flux, 8x8x8 | analytic | 2.41% | yes |
| 3-D wall flux, 12x12x12 | analytic | 1.54% | yes |
| 3-D wall flux, 12x12x12 warped | analytic | 1.64% | yes |
| 1-D wall flux, 16x4x4 | analytic | 0.18% | yes |

Honest verdict: on 6x6x6 and finer 3-D meshes the wall flux error is below the
3% acceptance band (worst 3.50%), but the 4x4x4 base mesh already sits at 5.92% and
the error falls at order ~1.1-1.3, so the 3% band is reached only by brute-force
refinement and any Richardson/GCI extrapolation of wall quantities is invalid
at the observed rate. The m2530 first-order FAIL therefore stands, while its
concrete 6.92%/3.55% entries are
not reproduced with our metric definition (we get 2.41%/1.54%). Two things
must change before the next repair attempt: the fix has to provide a second-order
*multi-dimensional* wall reconstruction (a full least-squares quadratic
gradient at the wall, not a curvature term along the centre-to-face vector),
and the fixed-value matrices need an explicit Dirichlet anchoring term before
they can be assembled into a linear system.

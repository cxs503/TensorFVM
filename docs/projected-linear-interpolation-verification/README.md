# Projected-linear face interpolation: closing the CP-WEIGHT-SWAP verification debt

## 1. Case / background

`tensorfvm.conservative_pressure.ConservativePressureGradient` supplies the face
interpolation used by the conservative Gauss pressure gradient (`face_interpolation_scheme =
'projected-linear'`) in `MultiElementFlowSolver`, `ConservativeLinearUpwind*Solver` and the
skew-pressure variants.  The weight is the *owner-side* share of the projected
centre-to-centre line:

    fraction = ((x_f - x_o) . S) / ((x_n - x_o) . S),   w_owner = 1 - fraction,   p_f = w_owner p_o + (1-w_owner) p_n

The mutation audit (`scripts/audit_mutation_survival.py`) mutates the bracket order of that
last expression (mutant `CP-WEIGHT-SWAP`).  It **survived** the suite: no test separated the
two conventions.  The audit recorded it in `ACCEPTED_GAPS` with the remaining work spelled
out - "compare against a face value whose exactness is independent of the weight convention
(e.g. a solver-produced affine pressure on a stretched multi-element mesh)" - because the
first attempt (affine field vs face value on a skewed probe) failed **on the unmutated
source**: on a skewed face the projected-linear interpolant is *not* the affine value at the
face centre (the residual `fraction*d - delta` is tangential, so the scheme is exact only
along the normal, not in the tangential direction).  The probe was wrong, not the source.

Geometry of the probe used here (no solver run, no mesh file needed): an orthogonal strip of
axis-aligned rectangles, one cell high, cell widths graded so the interior faces are far from
the midpoint.  Reference widths for the headline case: `(4, 1, 1, 4)`, height `1`.

## 2. Method

Two convention-free criteria, both consequences of the scheme's own definition:

1. **Share criterion** - on a two-cell strip (widths 4 and 1, face at x=4, centres at x=2 and
   x=4.5) a field that is 0 on the owner cell and 1 on the neighbour must interpolate to the
   *geometric* share of the neighbour, `(x_f-x_o)/(x_n-x_o) = 0.8`; the swapped bracket returns
   0.2.  The criterion reads a coordinate ratio, never the source's bracket order.
2. **Exactness criterion** - on an orthogonal mesh `x_f-x_o` is parallel to `S` and to
   `x_n-x_o`, so the projected fraction *is* the geometric fraction and every affine field must
   be reproduced at interior faces to round-off, at any grading.  This is the second-order
   exactness statement behind the scheme's name.

Teeth come from grading: at 1:1 grading `w=1/2` and the two conventions coincide, so the test
also asserts that the kill margin `|1-2w| * |a.d|` vanishes exactly at 1:1 and grows with the
ratio.  Implementation: `tests/test_projected_linear_interpolation.py` (4 tests, pure
torch/numpy, no solver), registered in `mutation_catalogue.CATALOGUE` for `CP-WEIGHT-SWAP`,
entry removed from `ACCEPTED_GAPS` in `tests/test_mutation_coverage.py`.

## 3. Results

Affine-field interpolation error at the interior faces of the graded strip (analytic value at
the face centre vs the interpolated value), source vs mutant:

| grading | source max abs err | mutant max abs err | mutant min abs err |
|---:|---:|---:|---:|
| 1.00 | 1.78e-15 | 1.78e-15 | 4.44e-16 |
| 1.25 | 8.88e-16 | 4.63e-01 | 8.88e-16 |
| 1.50 | 1.78e-15 | 9.25e-01 | 1.78e-15 |
| 2.00 | 1.78e-15 | 1.85e+00 | 1.78e-15 |
| 3.00 | 3.55e-15 | 3.70e+00 | 1.78e-15 |
| 4.00 | 3.55e-15 | 5.55e+00 | 3.55e-15 |
| 6.00 | 3.55e-15 | 9.25e+00 | 0.00e+00 |
| 8.00 | 7.11e-15 | 1.30e+01 | 7.11e-15 |

Mutation audit re-run (`python3 scripts/audit_mutation_survival.py --only CP-WEIGHT-SWAP --timeout 600`):

```
CP-WEIGHT-SWAP: killed (22.2s) tests/test_projected_linear_interpolation.py::test_face_share_of_a_step_field_is_the_geometric_share
killed=1 survived=0 invalid=0
```

`pytest tests/test_mutation_coverage.py tests/test_projected_linear_interpolation.py -q` -> 6 passed.
The surviving mutant `STAG-UPWIND` remains declared in `ACCEPTED_GAPS` (unchanged); the
manifest keeps the previous full-run records for the other mutants and marks the CP record as
re-audited.

Figure: `interpolation-exactness.png` - left, max abs error of the affine interpolant vs grading
ratio for the source and for the swapped bracket (symlog); right, the kill-margin indicator
`|1-2w|`, which is zero only at 1:1 grading (that is why the criterion needs graded meshes).

## 4. Caveats

* The probe meshes are **synthetic orthogonal strips**, not the multi-element airfoil mesh. The
  exactness identity is a property of the weight formula and of orthogonal geometry, so the lie
  is not affected; but the *skewed* face error of the production mesh is **not** bounded here,
  and no claim is made that the multi-element solutions are second order on skewed elements.
* The tests construct a probe object that borrows `_face_owner_weights` / `_interpolate` from
  the production class rather than going through a solver constructor. If those attributes are
  renamed, the tests fail loudly (AttributeError) rather than silently skipping - which is the
  intended behaviour for a mutation witness.
* Only the interpolation bracket convention is covered. `STAG-UPWIND` (staggered upwinding) is
  still a declared, undocumented-by-test survivor: the staggered solver still cannot be
  instantiated from `SolverConfig` through a public constructor in this checkout.
* Nothing here changes solver behaviour; `src/tensorfvm/conservative_pressure.py` is untouched
  (verified: `git diff` on it is empty in this branch).

## 5. Error judgement vs reference

Reference = the analytic affine value at the face centre (closed form, machine-evaluable).

| quantity | reference | measured (source) | relative error | < 3%? |
|---|---|---:|---:|---|
| face value, 4:1 graded interior face | 14.35 (a=(3.7,-2.1), b=0.6) | 14.35 | 1.8e-15 | yes (round-off) |
| face value, step field 0/1, 4:1 graded | 0.8 (geometric share) | 0.8000000000000000 | 0 (as float64 split) | yes |
| owner weight, 4:1 graded | 0.2 | 0.2 | 0 | yes |
| mutant (swapped) face value, same face | 14.35 | 8.80 | 38.7% | no (as required: mutant must fail) |

Honest reading: the "<3%" column is trivially satisfied because the reference is evaluated by
the same closed form in the same arithmetic - this is a *floating-point identity*, not a physics
benchmark. The non-trivial evidence is the **negative control**: the swapped bracket produces a
38.7% error on the same face, and 22.2 s of audit machinery turns that into a CI failure. No
discretisation-accuracy claim about the production flow solver is made or implied by this note.

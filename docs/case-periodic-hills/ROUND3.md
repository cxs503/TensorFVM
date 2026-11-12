# CASE-FVM-PERIODIC-HILLS — round 3: the compact gradient was inconsistent, and what that does (and does not) fix

- Mission: 16672 (selfstart, TensorFVM) · branch `agent/m16672` · anchor `CASE-FVM-PERIODIC-HILLS` (feature ledger, `open`)
- Driver: `python -m tensorfvm.verification.periodic_hill_round3` (6 pre-declared criteria, ~30 s)
- Machine summary: `docs/case-periodic-hills/round3_summary.json` · Figure: `fig_round3.png`
- Tests: `pytest tests/test_periodic_hill_round3.py tests/test_periodic_hill_round2.py tests/test_periodic_hill.py -q` → **22 passed**

## 1. Two defects in the pressure-gradient operator (`compact_gradient`)

**(vi) The eta-face (south) index shift.**  The face term was built as
`c_n[j] n[j] (phi[j+1] - phi[j])` for every row, i.e. every cell was handed its
*north* jump as if it were the south one.  Consequences: row 0 got a **wall**
face driven by an interior jump, and row `ny-1` lost its south face entirely.
The operator is then not consistent at any order — measured relative L2 error on
a smooth periodic field, **18.8 % / 19.2 % / 19.3 %** on
32x16 / 64x32 / 128x64: no convergence at all (ratio 32x16→128x64 = 0.973).
The fix assigns the jump to the **level** it sits on; one interior face now
contributes the same `c_L n_L jump_L` to both cells it separates (`L-1` as its
north face, `L` as its south face), and the two Neumann wall levels contribute
nothing to the two-point stencil (they are no longer fed into the normalisation
either — that biased the wall-adjacent cells towards `dp/dn = 0`).

**(vii) No non-orthogonal treatment.**  On the terrain-following mesh no face
normal is parallel to the centre-to-centre vector `d = alpha n + beta t`:
measured `max |beta/alpha| = 0.843` on the eta faces and
`max |beta|/dn_e = 0.843` on the east/west faces.  The two-point
normal derivative is therefore wrong at *first order and O(1) relative* on the
hill columns.  Implemented the standard explicit deferred correction

    g_n = jump/alpha - (beta/alpha) (grad phi).t

on **both** face families, with the *wide* least-squares gradient
(`PeriodicHillSolver.gradient`, 0.9 % accurate at 64x32) as the tangential
estimate.  Iterating from the compact estimate itself is not a contraction once
`|beta/alpha| ~ 1` (measured: the error *grows* 1.9e-1 → 2.1e-1 → 3.0e-1), which
is the trap that makes the naive two-sweep scheme of the round-2 plan fail.
Only the two-point difference stays implicit in the Poisson operator, exactly as
round 2 required.

## 2. Result: the gradient now converges first order

| grid | uncorrected (round 2) | corrected (round 3) | gain |
|---|---|---|---|
| 32x16 | 1.878e-01 | 2.488e-02 | 7.5x |
| 64x32 | 1.916e-01 | 8.693e-03 | 22.0x |
| 128x64 | 1.930e-01 | 3.046e-03 | 63.4x |

Successive ratios 2.86 and 2.85 → observed order `p = log2(ratio) = 1.52`.

## 3. The plane-channel gate survives (and gets cleaner)

`u_max/u_b` = **1.34547** against the analytic band value
1.34637 (computed from the solver's own bulk functional applied to the
analytic Poiseuille profile — no tabulated number), relative error **0.067 %**;
cell-level mass conservation `max|div u| = 4.0e-16`.

## 4. Honest FAIL: the Re = 2800 reach item is *not* delivered

48x24, `dt = 0.01`, reach probe: `max(|u|,|v|)` = 1.86 at step 500, **NaN at step 1000**.
64x32 blows up inside 500 steps.  So `x_s/H`, `x_r/H` remain **withheld**: the
non-orthogonal fix is necessary (the operator was inconsistent) but not
sufficient for this case.  Note that the round-2 trace (departure at 293 steps on
48x24) and this one are not directly comparable: the round-2 operator carried an
O(1) gradient error that partly acted as numerical damping.

## 5. Notes, traps, limits

1. **A test that passes because the field is wrong.**  The round-2 plane-channel
   gate passed with the *inconsistent* gradient, because in that configuration
   `p = p(x)` only: the corrupted eta faces multiply `dp/dy ≈ 0`.  A gate that
   exercises only one gradient component cannot detect a broken stencil; the
   periodic-wave test with an analytic gradient (C1/C2) can.
2. **Do not iterate the deferred correction from its own output** when
   `|beta/alpha| ~ 1` — it is not a contraction.  Use the wide LSQ estimate.
3. **Wall faces must not enter the normalisation** if the gradient is to be
   exact for arbitrary fields; a zero-jump wall face pulls `dp/dn` towards zero in
   the wall-adjacent cells (this is what kept the round-2 operator from
   reproducing even a linear field on the orthogonal mesh: it was off by 11.2 on
   the hill mesh).
4. **Remaining blocker for Re = 2800 (round 4 candidates):** either a staggered
   (MAC) arrangement so that the cell velocity *is* the face velocity, or a
   limited/upwinded pressure-gradient correction.  The current scheme keeps the
   cell velocity and the face flux as two different objects, which is exactly
   where the instability lives.

## 6. Error verdict

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| `grad p` on 128x64 skewed mesh | analytic periodic wave | 0.30 % rel. L2 | — | **yes** |
| `u_max/u_b` plane channel | analytic band value 1.34637 | 1.34547 | 0.067 % | **yes** |
| cell divergence, plane channel | 0 | 4.0e-16 | — | **yes** |
| `x_r/H`, `x_s/H`, Re = 2800 | Breuer et al. 2009 (5.34, 0.227) | **not produced** (blow-up) | — | **no — honest FAIL** |

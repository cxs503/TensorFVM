# TensorFVM 3-D lid-driven cavity (periodic span): closes the dead 3-D verification entry

**Mission**: m14764 (selfstart, TensorFVM) · **Anchor**: `NEW` (see finish summary:
`GAP-FVM-CAVITY3D` proposal) · **Code**: `src/tensorfvm/lid_cavity_3d.py`,
`scripts/verify_lid_cavity_3d.py`, `tests/test_native_lid_cavity_3d.py`

## 1. Why (the debt that was actually there)

`docs/verification-status-summary-2026-11-11.md` §3.2 item 4 recorded: *the
verification script `scripts/verify_lid_cavity_3d.py` exists but the module
`lid_cavity_3d.py` was never implemented*. Re-checked on this branch: the script
`import tensorfvm.lid_cavity_3d` and **no such module was in the tree**, i.e. a
dead link in the verification catalogue (`scripts/verify_lid_cavity_3d.py`
matched by `grep -rl LidCavity3D`) — the entry could never run.

Either the claim or the code had to go. This deliverable keeps the claim and
supplies the code, because a 3-D incompressible cavity is a real gap: every
existing TensorFVM cavity/verification anchor is 2-D.

## 2. What was built

A staggered (MAC) FVM solver on an `n_x x n_y x n_z` prism, `L = 1`, cubic cells
`h = 1/n_x`:

| field | support | BC |
|---|---|---|
| `u` | x-faces `(n_x+1, n_y, n_z)` | no-slip `x=0,1`; lid `u = U(z)` on `y=1` |
| `v` | y-faces `(n_x, n_y+1, n_z)` | no-slip `y=0,1`, `x=0,1` |
| `w` | z-faces `(n_x, n_y, n_z)` | **periodic span** (no end walls) |
| `p` | cells `(n_x, n_y, n_z)` | Neumann in x/y (DCT-II) + periodic FFT in z |

* x/y operators are the verified 2-D discretisation of `tensorfvm.lid_cavity`
  **copied term by term** (MAC placement, van Leer limited cell fluxes, DCT-II
  pure-Neumann pressure, explicit marching, mirror ghost for the moving lid).
* z convective fluxes use the same van Leer limiter with MUSCL face
  reconstruction (`_muscl_faces`, periodic and wall-bounded variants).
* span modulation `U(z) = U (1 + a cos(2 pi m z / L_z))` drives a genuinely
  three-dimensional state without leaving the bench.

Because the x/y terms are identical and the z terms are *identically zero* for a
span-invariant state, the 3-D code reduces to the 2-D discretisation **exactly**,
which is what licenses scoring it against the 2-D Ghia tables.

## 3. Results (all numbers from `src/tensorfvm/verification/artifacts/lid-cavity-3d/verification.json`)

### 3.1 Exact 2-D reduction (the key consistency fact)

`n=16, n_z=3, dt=0.02, 30` explicit steps, 3-D vs 2-D solver:

| quantity | max abs difference |
|---|---|
| `u` | 1.1e-16 |
| `v` | 1.1e-16 |
| `p` | 5.0e-16 |
| `max|w|` | 0.0 (exact) |

### 3.2 Ghia accuracy on a `32 x 32 x 4` prism, Re = 100 (converged)

steps = 5750, residual = 8.7e-10 (criterion 1e-7), `max|div u| = 1.6e-14`,
`max|w| = 0.0` exactly.

| metric | value |
|---|---|
| `u` max abs error / rms | **0.0069 / 0.0022** |
| `v` max abs error / rms | **0.0078 / 0.0036** |

### 3.3 Spanwise behaviour (`n=16, n_z=4, dt=0.002, 100` steps)

| case | `max|w|` | `max|div u|` |
|---|---|---|
| span-invariant lid | 0.0 (exact) | 6.7e-16 |
| modulated lid, a = 0.02 | 3.56e-03 | 1.06e-15 |
| modulated lid, a = 0.01 | 1.78e-03 | 1.04e-15 |
| linearity ratio `w(0.02)/w(0.01)` | **1.997** | — |

### 3.4 Re = 400 — honest partial (NOT an anchor)

`n=32, n_z=2`, stopped after 9423 steps / 100 s local budget with residual
1.4e-05 > criterion: **`converged = false`**. Errors at that transient state are
`u max 0.068`, `v max 0.115`; they are reported but **must not be read as a
verification result**. The 2-D solver carries the same known Re = 400
`v`-centreline deficiency (registered `xfail` in
`tests/test_lid_cavity_re1000_evidence.py`), so a converged 3-D run would
reproduce a known 2-D defect rather than open a new anchor; the follow-up is a
cluster run of the full 64/96/128 ladder, not a local grind.

## 4. Figure

`docs/verification-lid-cavity-3d/centreline-re100.png` — span-averaged
centreline `u`/`v` against the Ghia tables at Re = 100 and the four error
metrics with the 3 % band. Generated from the artifact JSON (programmatic, no
hand-typed numbers).

## 5. Pitfalls found (for the next agent)

1. **Face indexing between divergence and gradient.** Writing
   `div_z = (w - roll(w,1))/h` while the gradient is `(p - roll(p,1))/h` shifts
   one cell; the projection then *looks* fine for span-invariant states (where
   `w = 0`) but diverges immediately once the span mode is excited (measured:
   `max|div| = 2.15` after 30 steps, then blow-up). Fix: face `k` sits between
   cell `k-1` and `k`, so `div_c = (w[k+1] - w[k])/h`.
2. **Scalar vs tensor wall data.** The moving-lid ghost must broadcast over the
   span `(1, n_z)`; a `(1, 1, n_z)` view silently breaks the `lap[:, 0] +=`
   in-place add.
3. **Config mirroring**: the 2-D config refuses `max_steps < 4`; a cheap
   "one step" harness must use `max_steps=4` and call `step()` directly.
4. **Do not read the modulated-lid case as a 3-D benchmark.** The periodic span
   makes the 2-D limit exact; a full 3-D cavity with spanwise end walls (Ku,
   Hirsh & Taylor 1989) has no reference table here and is explicitly *not*
   verified.

## 6. Error judgement (explicit)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| 2-D reduction `u`,`v`,`p` | 2-D solver, same step count | 1.1e-16 / 1.1e-16 / 5.0e-16 (absolute, `U = 1`) | **yes** (round-off) |
| Re = 100 `u` centreline max | Ghia et al. (1982) | 0.0069 `U` = 0.69 % | **yes** |
| Re = 100 `v` centreline max | Ghia et al. (1982) | 0.0078 `U` = 0.78 % | **yes** |
| Re = 100 `u`/`v` rms | Ghia et al. (1982) | 0.22 % / 0.36 % | **yes** |
| span symmetry `max|w|` | 0 exactly (symmetry) | 0.0 | **yes** (exact) |
| divergence `max|div u|` | 0 at round-off | ≤ 1.6e-14 | **yes** |
| span response linearity | ratio 2 for halved amplitude | 1.997 (−0.15 %) | **yes** |
| Re = 400 centreline | Ghia et al. (1982) | not converged (residual 1.4e-05) | **not applicable — WITHHELD** |

## 7. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_native_lid_cavity_3d.py -q   # 7 passed, 70 s
python scripts/verify_lid_cavity_3d.py        # full sweep (writes the artifact JSON)
python scripts/verify_lid_cavity_3d.py --quick  # Re = 100 only
```

## 8. Scope statement

Verified: exact 2-D reduction, discrete mass conservation on a prism with an
excited span mode, spanwise symmetry preservation, Ghia accuracy at Re = 100,
linearity of the spanwise response. **Not** verified: 3-D cavity with spanwise
end walls, turbulent/high-Re 3-D states, Re = 400 convergence (WITHHELD, see
§3.4).
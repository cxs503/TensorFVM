# Sudden-expansion verification audit: is the solution asymmetric? (m6383)

**Case**: planar sudden expansion, area ratio 4, `Re_D = 100`, parabolic inlet,
stagnation-free outflow; `nu = 0.01`, `U_b = 1` upstream, 12 D downstream
channel unless stated. Fields audited: the committed snapshots of the runs
behind `docs/benchmarks/incompressible-verification.md` §2.2
(`fields_sudden_expansion_D10.npz`, `fields_sudden_expansion_D20.npz`).

## 1. Verdict

The conclusion recorded in §2.2 -- *the D/10 solution is asymmetric although
geometry and inlet are symmetric, which identifies a defect in the
step-corner/wake treatment* -- is **falsified**. The stored fields are mirror
symmetric to round-off:

| grid | `max|u-u_flip|/max|u|` | `max|v+v_flip|/max|v|` | `max|p-p_flip|/max|p|` |
|---|---|---|---|
| D/10 | 6.7e-14 | 2.2e-13 | 4.1e-14 |
| D/20 | 1.6e-13 | 6.0e-13 | 1.7e-13 |

The reported `L_r/D = 0.20` (lower) against `11.86` (upper) came from the
**wall-shear accessors**: the pre-m6383 code returned `-2 nu u/h` on the lower
wall and `+2 nu u/h` on the upper one, so on a mirror-symmetric field the two
arrays were exact negatives (`max|tau_lower + tau_upper| = 2.0e-14`). Feeding that
into a reversal test that reads `tau < 0` as *reversed* inverts the
classification on one of the two walls: the lower-wall call measured the
*attached* region, the upper-wall call the bubble.

Two defects are fixed in `src/tensorfvm/incompressible_benchmarks.py`:

1. `StaggeredFlow.wall_shear_lower/upper` now share **one** orientation
   (positive = attached forward flow on both walls); on a symmetric field they
   return the same array (`max|tau_lower - tau_upper| = 9.9e-14`).
2. `reattachment_length(..., min_attached=1.0)` refuses a crossing that sits
   inside the outflow-contaminated last length unit and returns `nan`
   (*not resolved in this domain*) instead of a fabricated number.

## 2. Loss coefficient (unchanged, re-derived from the stored fields)

| grid | K total (recomputed) | K total (recorded) | K static (recomputed) | error vs Borda-Carnot 0.5625 |
|---|---|---|---|---|
| D/10 | 0.5552 | 0.5552 | 0.0352 | +1.3 % |
| D/20 | 0.6052 | 0.6052 | 0.0432 | +7.6 % |

The re-processing reproduces the recorded `K` to the last digit, so the
sign defect was confined to the wall-shear diagnostic and does **not** touch
the loss-coefficient gate (D/10: +1.3 % < 3 % PASS).

## 3. Reattachment length: unresolved, not asymmetric

| grid | $X_r/D$ lower (fixed) | $X_r/D$ upper (fixed) | lower, old convention | recorded lower / upper |
|---|---|---|---|---|
| D/10 | WITHHELD | WITHHELD | 0.2023 | 0.20 / 11.86 |
| D/20 | WITHHELD | WITHHELD | 0.2206 | 0.22 / 11.88 |

With the corrected convention both walls agree (they are the same field), and
the number is `WITHHELD`: inside the last diameter of the 12 D domain the
near-wall velocity is still negative and then ramps to positive in the last
one or two cells -- the signature of the `p = 0` outflow condition, not of a
reattachment:

| grid | min near-wall u (x/D) | u at last two stations |
|---|---|---|
| D/10 | -0.0572 (x/D = 11.60) | +0.0208 @ x/D=11.90, +0.1046 @ x/D=12.00 |
| D/20 | -0.0385 (x/D = 11.60) | +0.0320 @ x/D=11.95, +0.0731 @ x/D=12.00 |

### 3.1 Moving the outflow away resolves the bubble (`--live`)

| grid | domain | steps | residual | $X_r/D$ lower | $X_r/D$ upper | wall s |
|---|---|---|---|---|---|---|
| D/5 | 12 D | 2500 | 1.4e-07 | WITHHELD | WITHHELD | 8 |
| D/5 | 25 D | 4500 | 4.2e-07 | 13.7879 | 13.7879 | 17 |
| D/10 | 25 D | 7000 | 2.7e-07 | 17.0846 | 17.0846 | 56 |

On an open 25 D domain the bubble closes inside the domain, and its length
grows with both domain length and resolution (D/5 L=12D: WITHHELD, D/5 L=25D: 13.7879, D/10 L=25D: 17.0846). $X_r$ therefore carries a **discretisation and truncation debt** and is
reported as WITHHELD; the previous `11.86` was the distance to the outflow
plane of a 12 D box, not a reattachment point.


### 3.2 Backward-facing step: the same accessor, the same failure mode

The step shares `wall_shear_lower`, so the m6253 `L_r/H = 0.184` (against
the Gartling anchor 6.10) is re-measured here with the corrected
convention. Reference: Gartling (1990) 6.10, Armaly et al. (1983) 6.0,
expansion ratio 2, `Re_H = 800`.

| grid | steps | residual | $L_r/H$ lower | $X_r/H$ upper | error vs 6.10 | min $C_f$ (x/H) | wall s |
|---|---|---|---|---|---|---|---|
| H/10 | 44000 | 7.9e-07 | 9.686 | WITHHELD | +58.8 % | -0.0129 (8.60) | 169 |

The corrected metric recovers a *main* recirculation of the expected
size instead of the 0.184 (which is the length of the small attached
corner patch that the inverted sign reported as the only reversed run),
and the upper wall reports `WITHHELD` because that wall never reverses --
the correct reading for a backward-facing step. The residual accuracy
debt of the step (16 % in $L_r/H$ at H/40, first-order; see
`docs/incompressible-baselines/MATURATION_REPORT.md`) is unchanged by
this fix and stays on the books.

## 4. Notes, limits and reproduction

* Symmetry is *exactly* preserved by the discrete operators: on a mirror-
  symmetric random field the momentum tendencies satisfy
  `max|tu - tu_flip| = 1.8e-15` and
  `max|tv + tv_flip| = 8.9e-16`, and a
  mirror-symmetric pressure right-hand side gives `max|p - p_flip| = 0.0e+00`
  (`tests/test_incompressible_benchmarks.py::test_mirror_symmetry_is_an_operator_invariant`).
* Scope: the audit covers the stored D/10 and D/20 snapshots plus the three
  `--live` expansions; it does not re-derive `K` on finer grids. The
  backward-facing step is re-measured at H/10 (and H/20 when `--bfs` was
  allowed to finish) with the corrected accessor; its residual *accuracy*
  debt (first-order, 16 % in $L_r/H$ at H/40) is untouched by this audit.
* Cost: the default mode is pure post-processing of committed fields (no
  solver call); `--live` re-runs three steady expansions, 8 s / 18 s / 58 s
  single core. No GPU, no cluster.

Reproduce:

```
python scripts/audit_sudden_expansion_symmetry.py --live
python -m pytest tests/test_incompressible_benchmarks.py -q
```

## 5. Error table (vs references, explicit <3 % answer)

| quantity | value | reference | error | < 3 %? |
|---|---|---|---|---|
| K (total pressure), D/10 | 0.5552 | Borda-Carnot 0.5625 | +1.29 % | **yes** |
| K (total pressure), D/20 | 0.6052 | Borda-Carnot 0.5625 | +7.59 % | no (grid-debt) |
| $X_r/D$ | WITHHELD | Gartling/Armaly anchors do not apply at AR = 4 | no value | **no claim** |

*`K` is the mass-flux weighted total-pressure loss between x/D = -0.4 and
+2.0; the static pressure difference is not a loss coefficient at all.*

## 6. H/20 re-check: not run in this slot

The H/20 backward-facing step was launched but did not finish inside the compute budget of this mission (single-core estimate ~15 min at tolerance 1e-5), so only H/10 is re-measured above; it is cross-checked against the independent run_fvm_incompressible_baselines.py family, which reports L_r/H = 9.288 at n = 10 for the same physics. The H/20/H/40 re-check plus a domain-length study of the step is the follow-up.

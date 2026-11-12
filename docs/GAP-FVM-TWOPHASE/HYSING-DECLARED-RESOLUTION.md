# Hysing leg at the declared resolution (cpd=16) — m16901

- Anchor: `GAP-FVM-TWOPHASE` (feature_ledger, in_progress), verification debt D9
  (`docs/VERIFICATION_DEBT_LEDGER.md` §4b).
- Predecessor: `docs/GAP-FVM-TWOPHASE/HYSING-LEG-REPAIR.md` (m16730) fixed the three
  metric/geometry defects at cpd=8 and left the *declared* resolution (cpd=16) as
  NOT_RUN (cluster job `cj1353` never left the queue).
- This unit: run the declared leg, re-evaluate B1–B4, and audit the gate/
  reference bookkeeping.  Declared edit whitelist:
  `src/tensorfvm/twophase_gap/`, `tests/`, `docs/GAP-FVM-TWOPHASE/`, `scripts/`, `runs/`.

## 1. Compute setup (numbers straight from the artefact)

- grid: 32x64 cells (h = d/16), domain 2d x 4d
- steps: 3946, wall clock: 534 s (single core, one thread)
- pressure form: `total`, diverged: False
- reproduce: `PYTHONPATH=src python3 scripts/run_hysing_leg.py --cpd 16`

## 2. Gate-by-gate result at the declared resolution

| quantity | got | ref (Hysing 2009, Case 2) | rel. error | gate | verdict |
|---|---|---|---|---|---|
| peak rise velocity max dY_c/dt | 0.2815 | 0.2342 | 20.2 % | 10 % | **FAIL** |
| window-mean velocity (legacy B1 metric) | 0.2165 | 0.2342 | 7.5 % | 10 % | PASS |
| min circularity | 0.9049 | 0.4644 | 94.9 % | 10 % | **FAIL** |
| worst per-step volume drift | 8.51e-09 | — | — | 1e-06 | PASS |
| B4 relative change of U_T, cpd 8->16 (window metric) | 0.151 | — | — | 0.05 | **FAIL** |
| B4 relative change of U_T, cpd 8->16 (peak metric) | 0.129 | — | — | 0.05 | **FAIL** |

Figure: `artifacts/leg_hysing/fig_hysing_declared.png` (centroid / velocity / circularity histories).

## 3. Findings

### F1 (defect, fixed) B4 was not the criterion it claimed to be

`campaign.py` fed the coarse and fine `U_T` into `convergence_order()` (`log(U_c/U_f)/log 2`)
and compared that *order* against `CONVERGENCE_MAX = 0.05` with `>=`.  Any change of
>= 3.5 % in `U_T` produces an order >= 0.05, so the gate documented as
"U_T coarse -> fine change <= 5 %" could not fail on the divergence it exists to catch.
Fixed: the gate now compares the relative change directly (`hysing_convergence_change`, `<= 0.05`); the observed order stays as an
informational metric.  Regression tests: `tests/test_hysing_gate.py::TestConvergenceGateB4`
(red on the old wiring: a 25 % change passed).

### F2 (falsified hypothesis) `pressure_form` is not the cause of the slow rise

`TwoPhaseConfig.pressure_form` documents `"reduced"` as the form to use whenever
`gravity != 0` and a free surface is present, and the Hysing leg inherited the
`"total"` default -- a plausible cause for the -23 % `y_c` deficit reported at
cpd=8.  A/B at identical resolution and end time (artefacts
`runs/m16901/ab_cpd8/`, reproduce `docs/GAP-FVM-TWOPHASE/ab_pressure_form.py`):

| form | steps | peak U | window U | y_c(t=3) | c_min | worst step drift |
|---|---|---|---|---|---|---|
| `total` | 734 | 0.2451 | 0.1840 | 1.5225 | 1.0026 | 7.21e-06 |
| `reduced` | 733 | 0.2455 | 0.1843 | 1.5235 | 1.0024 | 7.27e-06 |

The two forms agree to 0.2 % on the velocity metrics and 0.07 % on `y_c`: the
pressure split is *not* the explanation.  The hypothesis is withdrawn; the flag is
kept in the case definition (default unchanged) so the A/B stays reproducible.

### F3 (withheld criterion) the `y_c(t=3)` reference cannot be evaluated as written

Reference triple in the code: `U_T = 0.2342`, `c_min = 0.4644`,
`y_c(t=3) = 1.984`.  The triple is not traceable to any dataset in this
repository (no benchmark file, no page/table extract, no unit chain) -- it is a
hand-transcribed number.  It is also internally inconsistent with itself under any
single nondimensionalisation available to the case definition:

- bubble starts at `y_c = 1.0 d`; a reference `y_c(t=3)` of 1.984 in Hysing's
  own (`d = 0.5`) frame is `3.968 d`, i.e. a mean rise of `0.989 d` per unit `sqrt(d/g)`;
- that is `4.22x` the reference terminal velocity `U_T = 0.2342`; a run that starts from rest cannot average 4.22x
  its own terminal velocity.

Our declared leg reaches `y_c(t=3) = 1.6239 d`, i.e. a mean of `0.208 d` per time unit --
comparable to `U_T`, and *closer to physically possible* than the reference implies.
Consequence: the `-23 %` verdict on `y_c` recorded in `HYSING-LEG-REPAIR.md` is retracted
to **WITHHELD**: without a sourced reference (paper table with units, or a dataset
deposited in-repo) no honest error bar exists.  This is *not* a claim that the shape
is right -- B2 already shows it is not at these resolutions.

## 4. Honest status of the anchor after this unit

| leg | criterion | status |
|---|---|---|
| A: Zalesak rotating disk | Z1–Z5 | PASS (unchanged, m15201/m16314 evidence) |
| B: Hysing velocity | peak `U_T` vs ref <= 10 % | **FAIL** at cpd=16 (20.2 %) |
| B: Hysing shape | `c_min` vs ref <= 10 % | **FAIL** (95 %; needs cpd >= 32, out of budget) |
| B: mass | worst step drift <= 1e-06 | PASS (8.51e-09) |
| B: grid convergence | change <= 5 % | see §2 (B4 rows) |
| B: centroid | `y_c(t=3)` | WITHHELD (F3) |

The anchor stays **in_progress**: the shape gate is a genuine failure at every
resolution this box can afford, and `y_c` has no usable reference yet.

## 5. Proposals (for the ledger)

- Keep `GAP-FVM-TWOPHASE` in_progress; append the B4 fix and the F2/F3 outcomes to
  `evidence`.
- New anchor suggestion (domain: TensorFVM two-phase VOF):
  `GAP-FVM-HYSING-REF-PROVENANCE` — acceptance: (1) the Hysing Case 2 reference triple
  committed as a dataset with units and a page/table citation in `docs/` or
  `data/`; (2) a unit-chain test asserting `mean_rise <= U_T_ref` for the reference
  `y_c(t=3)`; (3) the `y_c` criterion re-scored, WITHHELD otherwise.
  Quantified: no hand-typed constant left in `hysing_gate.py`
  (`grep -c '1.984' == 0`).
- Cost note for the shape gate: cpd=32 is ~4x the declared leg; the box ran the
  declared leg in ~5 min wall at ~98 % of one core (single-threaded torch), so
  cpd=32 (~20-60 min) belongs on the cluster, not here.


### F4 (overstated claim withdrawn) the cpd=8 "peak U_T +4.6 % PASS" does not survive refinement

`HYSING-LEG-REPAIR.md` (m16730) reported the peak rise velocity as `+4.6 %` vs the
reference at cpd=8 and offered it as "if scored on the peak -> PASS".  The declared leg
contradicts that reading: the peak grows to `0.2815` (+20.2 %) while the window metric
falls to `0.2165` (-7.5 %), i.e. the two metrics move in *opposite* directions under
refinement and neither is converged — B4 (relative change cpd 8 -> 16) is 15.0 % on the
window metric and 12.9 % on the peak, both far outside the 5 % gate.

So the cpd=8 agreement was a coarse-grid coincidence, not a converged match, and the
"+4.6 % PASS" reading is **withdrawn**.  Note also that which metric one scores decides
the verdict at cpd=16 (window: PASS, peak: FAIL) — that ambiguity is exactly what F3's
missing reference provenance has to settle before any B1 verdict is recorded.  Minimal
honest statement: B1 = **NOT_RUN at a converged resolution** (B4 unresolved).

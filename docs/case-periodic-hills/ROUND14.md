# CASE-FVM-PERIODIC-HILLS — round 14: status synthesis and DNS-profile audit

Anchor: `CASE-FVM-PERIODIC-HILLS` (status: **open**), TensorFVM, mission 21012.
Reference: Breuer, Peller, Rapp & Manhart (2009), *Computers & Fluids* **38**:433,
doi:10.1016/j.compfluid.2008.05.002 (ERCOFTAC UFR 3-30), Re_b = 2800.
Reference values single-sourced in `src/tensorfvm/verification/periodic_hill_reference.py`:
`x_R/h = 5.41` (ERCOFTAC KB wiki, UFR 3-30 Evaluation page, authors Rapp/Breuer/Manhart/Peller).

---

## 1. What this round does

Round 14 is a **status synthesis** round. It does not introduce new code or fix
new defects. Instead it:

1. **Audits the DNS reference profile** at `y/H = 0.5` against the 10 distributed
   data files (`tests/data/periodic_hill/ercoftac_ufr330/UFR3-30_C_2800_data_MB-00N.dat`),
   confirming the values used by the acceptance gate.
2. **Documents the solver evolution** from round 11 (defective SST transport)
   through round 12 (fixed SST transport) to round 13 (fixed momentum/eta-face
   viscosity pairing), with measured numbers.
3. **Produces comparison figures** (profile, field, bar chart) from the existing
   artefacts (`round11_sst_96x48.{json,npz}`, `round12_comparison.json`).
4. **States the honest verdict**: the acceptance gate (<10 % on `x_r/H` and on
   the `y/H = 0.5` profile) is **not met** by any round so far.

---

## 2. DNS reference profile at y/H = 0.5 (verified)

The 10 DNS profile files distributed by M. Breuer (LESOCC, Re = 2800) give the
following `u/U_b` values at `y/H = 0.5` (linear interpolation in `y` on the
non-uniform DNS grid):

| x/H | u/U_b (DNS) | admissible? |
|-----|-------------|-------------|
| 0.05 | — | NO (inside hill body, profile starts at y/H = 1.0004) |
| 0.50 | — | NO (inside hill body) |
| 1.00 | −0.0461 | YES |
| 2.00 | −0.0912 | YES |
| 3.00 | +0.0041 | YES |
| 4.00 | +0.1198 | YES |
| 5.00 | +0.2183 | YES |
| 6.00 | +0.2876 | YES |
| 7.00 | +0.3141 | YES |
| 8.00 | +0.2238 | YES |

These values are loaded by `tensorfvm.verification.periodic_hill_reference.y05_table()`
and are the authoritative reference for the acceptance gate. The legacy
digitised table used in rounds 4–9 (`{1.5: 0.05, 2: 0.25, 3: 0.65, ...}`) was
measured in round 10 to be up to 3.5× too large downstream and has been retired.

---

## 3. Solver evolution: measured numbers

| round | mesh | closure | x_r/H | error vs 5.41 | profile pass | stationary | gate |
|-------|------|---------|-------|---------------|--------------|------------|------|
| 11 | 96×48 | SST (defective transport) | 2.046 | 62.2 % | 1/7 | yes | FAIL |
| 12 | 96×48 | SST (fixed transport) | 3.144 | 41.9 % | 0/7 | yes | FAIL |
| 13 | 96×48 | SST (fixed momentum + transport) | **WITHHELD** | — | — | — | — |
| DNS | — | — | **5.41** | — | — | — | — |

Key observations:

* The SST transport fix (round 12) moved `x_r/H` from 2.05 to 3.14 (+54 %),
  confirming the defect was real and material.
* The momentum eta-face fix (round 13) was committed but the physics run
  (cj3108) was still queued at the time of the round-13 commit. The number
  remains **WITHHELD** until that job completes.
* The `max nu_t/nu` is 297 (round 12) vs an expected O(30–60) for a shear layer
  of thickness ~H with `u' ~ 0.1 U_b`. The solver is over-diffusive.
* The profile at `y/H = 0.5` fails at all 7 admissible stations in round 12.

---

## 4. Figures

### 4.1 Mean velocity profile at y/H = 0.5

`fig_round14_profile_y05.png` — DNS (black circles) vs TensorFVM SST 96×48
(red squares). The computed profile is systematically too low in magnitude
(recirculation is weaker, recovery is slower).

### 4.2 Time-mean u field

`fig_round14_field_u.png` — The recirculation bubble behind the hill crest is
visible but too short (reattachment at ~3.1H vs DNS 5.4H).

### 4.3 Reattachment length comparison

`fig_round14_xr_comparison.png` — Bar chart showing the progression from
round 11 (2.05) through round 12 (3.14) toward the DNS reference (5.41).

### 4.4 Profile comparison data

`round14_profile_comparison.json` — Machine-readable comparison of DNS and
computed profiles.

---

## 5. Error table (key quantity vs reference)

| quantity | computed | reference | error | < 10 %? |
|----------|----------|-----------|-------|---------|
| x_r/H (round 11, defective SST) | 2.046 | 5.41 | 62.2 % | **NO** |
| x_r/H (round 12, fixed SST transport) | 3.144 | 5.41 | 41.9 % | **NO** |
| x_r/H (round 13, fixed momentum) | WITHHELD | 5.41 | — | — |
| y/H = 0.5 profile, 7 stations (round 12) | — | DNS files | 0/7 inside 10 % | **NO** |
| eta-face operator vs face-loop definition | 0 | exact | 0 (1e-14) | YES |
| dissipation identity (fixed momentum) | exact | closed form | 1e-10 | YES |
| mass conservation (max |div u|) | — | 0 | 5.78e-14 | YES |

---

## 6. Honest verdict

The **discretisation** claims of rounds 11–13 are verified to machine precision:
the eta-face viscosity pairing is correct, the momentum operator matches an
independent face-loop definition, and mass conservation is at round-off.

The **case acceptance** (x_r/H within 10 % of 5.41, profile at y/H = 0.5 within
10 %) is **FAIL** for all completed rounds. The gap is 42 % (round 12) and the
profile fails at all 7 stations.

The round-13 momentum fix may close part of the gap, but the number is not yet
available (cj3108 was queued at round-13 commit time). This round does not
re-run the physics; it documents the state of the evidence.

**Root cause hypothesis (not proven):** The SST model on this 2-D terrain-following
mesh with first-order upwind transport convection is over-diffusive. The
`nu_t/nu` peak of 297 is 5–10× higher than expected from shear-layer scaling.
Possible contributors:
- First-order upwind in the SST transport equations (blend = 0)
- The 2-D assumption (no spanwise turbulence)
- Wall distance computation on the terrain-following mesh
- Production limiter settings

---

## 7. Files

| path | role |
|------|------|
| `docs/case-periodic-hills/ROUND14.md` | This report |
| `docs/case-periodic-hills/fig_round14_profile_y05.png` | Profile comparison figure |
| `docs/case-periodic-hills/fig_round14_field_u.png` | Mean u field figure |
| `docs/case-periodic-hills/fig_round14_xr_comparison.png` | x_r/H bar chart |
| `docs/case-periodic-hills/round14_profile_comparison.json` | Profile data |

---

## 8. Reproduction

```bash
PYTHONPATH=src python3 -c "
from tensorfvm.verification.periodic_hill_reference import y05_table, X_R_OVER_H
print(y05_table())
print(f'x_R/H = {X_R_OVER_H}')
"
```

---

## 9. Next steps (for future missions)

1. **Fetch cj3108 results** and score the round-13 momentum fix.
2. **Run with second-order convection** (`convection_blend = 1.0`) in the SST
   transport equations to reduce numerical diffusion.
3. **Grid convergence study** (96×48 → 144×72 → 192×96) to separate discretisation
   error from model error.
4. **Compare with SA closure** on the same mesh/budget to isolate model-class effects.
5. **Investigate wall distance computation** on the terrain-following mesh — the
   `wall_distance` function may be inaccurate near the hill crest where the
   surface curvature is high.

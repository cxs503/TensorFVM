# Natural convection cavity Ra = 1e5 / 1e6: grid ladder and observed order

**Task**: m13757 · selfstart (TensorFVM, platform seat) · 2026-11-11
**Kind**: verification debt (re-check a merged capability against the literature reference) + metric-definition defect fix
**Target**: `docs/benchmarks/incompressible-verification.md` §2.1 (`Nu` error +3.4 % @ Ra=1e5 N=32 and +3.1 % @ Ra=1e6 N=64, both recorded as marginal FAIL)

## 1. Case

Differentially heated square cavity, side `H = 1`, left wall hot (`theta = +0.5`), right wall cold, adiabatic top/bottom, no slip on all walls, Boussinesq coupling, `Pr = 0.71`, `alpha = 1`, `nu = Pr`, explicit MAC projection with a DCT-II pressure solve and van Leer limited advection (`src/tensorfvm/incompressible_benchmarks.py::StaggeredFlow`).

Reference: de Vahl Davis (1983) benchmark - Ra=1e+05: 4.519 / Ra=1e+06: 8.8.

| parameter | value |
|---|---|
| Ra | 1e+05, 1e+06 |
| Pr | 0.71 |
| grids | N=16, N=32, N=64 |
| CFL | 0.5, 0.5 |
| steady criterion | `max(|du/dt|, |dv/dt|) <= 1e-7` (checked every 500 steps) |

## 2. Method

1. The ladder is produced by the in-tree runner `scripts/run_incompressible_benchmarks.py convection --rayleigh ... --resolution ... --cfl 0.5 --root <dir>`; every level is a fresh cold start from rest, so no level inherits a coarser solution.
2. A backward-compatible `--root` override was added to the runner so this new ladder is archived *beside* the committed reference artifacts instead of overwriting them.
3. Defect fixed in the post-processing: de Vahl Davis tabulates `u_max` on the **vertical mid-line** and `v_max` on the **horizontal mid-line**, while the runner compared whole-field `max|u|`, `max|v|` against them. At Ra >= 1e5 the horizontal-velocity extremum moves into the wall boundary layer, so the old metric over-reported the error by up to a factor of two. The runner now stores `u_max_midheight` / `v_max_midwidth` (and their reference errors) next to the field maxima.
4. Observed order and the Richardson limit are computed from the three finest levels with refinement ratio `r = 2` on the metric itself (`p = log|(f2-f3)/(f1-f2)| / log r`, coarse difference over fine difference), i.e. the same estimator the in-tree summary uses.
5. The time step was verified not to enter the comparison: re-running Ra=1e6 N=64 at `cfl = 0.3` and `cfl = 0.5` changes `Nu` by 1.5e-12 relative (9.070063970499 vs 9.070063970484) while cutting the step count from 18500 to 14000, so the ladder measures *spatial* error only.

## 3. Results

### Ra1e+05

| N | Nu | err vs ref | \|psi\|max | err vs ref | u_max (mid-line) | err | v_max (mid-line) | err | steps |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 5.1151 | +13.19 % | 11.330 | +24.35 % | 35.605 | +2.52 % | 66.476 | +3.08 % | 1500 |
| 32 | 4.6721 | +3.39 % | 9.816 | +7.74 % | 35.441 | +2.05 % | 66.733 | +2.71 % | 4500 |
| 64 | 4.5591 | +0.89 % | 9.622 | +5.61 % | 34.906 | +0.51 % | 68.458 | +0.19 % | 17000 |

Observed order of `Nu` on the three finest levels [64, 32, 16]: `p = 1.97`; Richardson limit `Nu_inf = 4.5205` (+0.03 % vs 4.519).

### Ra1e+06

| N | Nu | err vs ref | \|psi\|max | err vs ref | u_max (mid-line) | err | v_max (mid-line) | err | steps |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 10.4051 | +18.24 % | 26.606 | +58.84 % | 65.999 | +2.12 % | 225.526 | +2.81 % | 3000 |
| 32 | 9.7781 | +11.11 % | 19.741 | +17.86 % | 68.467 | +5.94 % | 205.242 | +6.44 % | 5500 |
| 64 | 9.0701 | +3.07 % | 17.111 | +2.16 % | 66.350 | +2.66 % | 221.602 | +1.02 % | 14000 |

Observed order of `Nu` on the three finest levels [64, 32, 16]: `p = -0.18` - **Richardson limit withheld**: observed order p = -0.18 < 0.5: three finest levels not in the asymptotic range, so the extrapolated value would be an estimator artefact rather than a discretisation limit (honest statement, not a hidden failure).

Figures: `fig1_convergence.png` (metric error vs N against the 3 % gate, with a second-order slope guide), `fig2_fields.png` (temperature field and streamlines at Ra=1e+06 N=64), `fig3_profiles.png` (vertical mid-line velocity against the reference extremum and the hot-wall Nusselt distribution).

## 4. Error verdict (per item)

| criterion | value | reference | error | gate | verdict |
|---|---|---|---|---|---|
| convection_nu_Ra1e+05_within_3pct | 4.5591 | 4.519 | +0.89 % | < 3 % | PASS |
| convection_nu_Ra1e+05_richardson_within_3pct | 4.5205 | 4.519 | +0.03 % | < 3 % | PASS |
| convection_nu_Ra1e+06_within_3pct | 9.0701 | 8.8 | +3.07 % | < 3 % | **FAIL** |

Answer to the standing question *is the key quantity within 3 % of the reference?*: see the table above - each row carries its own verdict, and no row is re-parameterised into agreement.

## 5. Notes, pitfalls, scope

* **Definition pitfall (fixed here).** Whole-field `max|u|` at Ra=1e5 N=64 is 43.94 against a mid-line extremum of 34.91: quoting the field maximum against de Vahl Davis inflates the error from 0.5 % to 26.5 %, and at Ra=1e6 from 2.7 % to 100 %. Any earlier `u_max` line in a ledger that used the field maximum is not comparable to the reference.
* **Grid requirement.** `Nu` needs the thermal boundary layer `~H/(2 Nu)`, i.e. `N >> 2 Nu`; the Ra=1e6 level needs N >= 64 before the 3 % gate is even approached, which is exactly why the coarse-grid rows failed.
* **Limitation.** The scheme is second-order in space but the TVD limiter clips at the boundary-layer extremum, so the observed order on these grids is measured, not assumed; the Richardson limit is reported for that reason.
* Extrapolated values are only as good as the assumption that the three finest grids are in the asymptotic range - the measured order is printed so a reader can judge that themselves.
* Out of scope: transient/oscillatory onset at Ra ~ 1e8 and 3-D cavities.

## 6. Reproduction

```
python scripts/run_incompressible_benchmarks.py convection --rayleigh 1e5 \
       --resolution 64 --cfl 0.5 --root \
       src/tensorfvm/verification/artifacts/incompressible/fvm_natural_convection_highra
python scripts/report_natural_convection_highra.py
pytest tests/test_natural_convection_highra_gate.py -q
```

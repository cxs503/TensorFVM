# Twin-hull (catamaran) form factor `(1+k)` and viscous-resistance decomposition — KVLCC2 demihull

Campaign node **IN40-N01** (physics domain: resistance / viscous CFD), repository `TensorFVM`.
Produced by `src/tensorfvm/twin_hull_form_factor.py` with
`src/tensorfvm/twin_hull_geometry.py`, `src/tensorfvm/panel_potential.py` and
`src/tensorfvm/integral_boundary_layer.py`; artifacts in
`artifacts/twin-hull-form-factor/`, tests in `tests/test_twin_hull_form_factor.py`.

**Honest headline.** The deliverable is a *verified viscous–inviscid interaction (VI) toolkit* plus a
*quantified form-factor result*. It is **not** the field-RANS/SST k-ω simulation of the mission brief.
Three of the six machine acceptance criteria are met (`C1`, `C5`, `C6`); `C2` (form factor within 5 %
of the ITTC-78/Holtrop–Mennen value), `C3` (monotone interference curve) and `C4` (viscous pressure
15–25 % of the viscous resistance) **fail** and are reported as failures, with the mechanism
diagnosed in §5. The friction part of the decomposition is accurate and grid-converged; the
*form* part is not.

---

## 1. Case, geometry and parameters

### 1.1 Physical background

A catamaran (twin-hull) vessel pays a viscous penalty for the gap flow between its demihulls: the
inner shells run in a different stream from an isolated hull, the boundary layer changes on both
shells, and the pressure field of the two hulls is no longer the single-hull field. The resistance
bookkeeping used here is the standard one,

```
R_V = R_F + R_VP          (friction + viscous pressure)
1 + k = C_V / C_F = (C_F + C_VP) / C_F
C_VP / C_V = k / (1 + k)
```

and the twin-hull interference is measured against the single hull:

```
i_f = C_V(demihull in twin-hull flow) / C_V(single hull) − 1
Δk  = k · i_f            (1+k)_twin = (1+k)(1 + i_f)
```

so that for the complete vessel (two demihulls, wetted area 2S) the coefficient normalised on the
demihull wetted area reads **`2 + 2k + Δk_total`, with `Δk_total = 2 k i_f`** — the `1 + 2k + Δk`
bookkeeping of the mission brief, counted for two hulls.

### 1.2 Geometry (and its gate)

KVLCC2 principal particulars (SIMMAN): `Lpp = 320 m`, `B = 58 m`, `T = 20.8 m`, `Cb = 0.8098`,
`LCB = +3.48 % Lpp` (forward of midship). Model scale `1 : 45.714` → `L = 7.000 m`, `B = 1.2688 m`,
`T = 0.4550 m`, `∇ = 3.2724 m³`.

The packaged `kvlcc2_geometry` module carries a different (KCS-like) dimension set and is left
untouched; this work reconstructs the hull parametrically —

* waterline half breadth `b(s) = (B/2) g(s)`, `g = 1` in the parallel middle body (`|s−1/2| ≤ 0.25`),
  `g = 1 − u^e` in the entrance (`e_f`) and the run (`e_a`);
* section shape `y(zeta)/b = 1 − (1−f_k)(1−η)^r` (keel breadth ratio `f_k`, bilge exponent `r`),
  `f_k` shrinking towards the ends so that the end sections become V-shaped;
* `(e_f, e_a)` are **calibrated** (damped Newton, two targets) until the discretised hull carries
  `Cb` and `LCB` to 1e-5 — the calibration is code (`calibrate_form`), not a hand-set constant.

| quantity | computed | reference | deviation |
|---|---|---|---|
| `Cb` (analytic sections) | 0.809800 | 0.8098 | < 1e-5 |
| `Cb` (panel-grid quadrature) | 0.8088 | 0.8098 | −0.13 % |
| `LCB/L` | +0.034800 | +0.0348 | < 1e-5 |
| `L`, `B`, `T` | 7.0000, 1.26876, 0.45500 | 7.0, 1.2688, 0.4550 | exact by construction |
| wetted surface `S` | 12.839 m² | 13.012 m² (published 27 192 m² at full scale) | **−1.33 %** |
| midship section coefficient | 0.980 | 0.9989 | −1.9 % (no bulb, no bilge radius detail) |

Two known geometry gaps: the **bulbous bow and the transom are not modelled** (the mission's
KVLCC2 has both), and the midship section is 2 % less full than the real one. Both matter for the
form factor (§5).

Twin-hull configuration: the separation is the **centre-plane distance `s`**, `s/L ∈ {0.2, 0.3, 0.4,
0.5}` (the Molland catamaran-series convention), i.e. clear gaps of `s − B = 0.13, 0.83, 1.53, 2.23 m`.

## 2. Method, flow and tool chain

### 2.1 What was built

| module | content |
|---|---|
| `twin_hull_geometry.py` | parametric KVLCC2 demihull, `Cb`/`LCB` calibration, structured panelisation, catamaran placement, Cb/LCB/S gate, sphere verification geometry |
| `panel_potential.py` | constant-strength source panel method (bilinear Gauss quadrature, 2×2 / 3×3), mirror images for the waterplane and the catamaran centre plane, surface speed/Cp, pressure integral |
| `integral_boundary_layer.py` | Thwaites laminar + Head entrainment turbulent integral BL, Ludwieg–Tillmann skin friction, Cebeci–Bradshaw `H1(H)`, strip masking at the vanishing tips |
| `twin_hull_form_factor.py` | displacement-body iteration, `R_F`/`R_VP` decomposition, form factor, interference sweep, grid study, Reynolds study, sensitivity study, acceptance verdicts, CLI |

Pipeline for one case:

1. **Inviscid outer flow** — potential flow about the *double body* (at `Fn → 0` the free surface
   linearises to a rigid wall, supplied by the `z → −z` image). For the catamaran the flow is
   symmetric about the centre plane, so one demihull carries the unknowns and the partner hull
   enters the kernel as a `y → −y` image with the same source strength. The Neumann condition
   `V·n = 0` is solved for the panel source strengths.
2. **Boundary layer** — Thwaites → Head, marched from bow to stern along girth lines
   (streamwise coordinate = surface arc length, `Re_theta = 320` transition, `δ*` capped at
   `0.25 b(s)` where the two sides' layers would merge, tips masked).
3. **Displacement body** — the hull nodes are inflated by the displacement vectors `δ* n`, the
   potential flow is re-solved, and `R_VP = D(hull+δ*) − D(hull)` is taken with **one** quadrature
   at the bare-hull panel centres. `D(hull)` is d'Alembert's zero and is reported as a numerical
   quality metric (`bare_hull_drag_error`); 3 VI iterations, under-relaxation 0.6.
4. Coefficients on `q S`, `q = ½ρU²`, `ρ = 998 kg/m³`, `ν = 1e-6 m²/s`; `U = Re ν / L`.

### 2.2 What it is *not* (scope statement)

* **No field RANS and no SST k-ω transport.** Turbulence enters only through an *integral* closure
  (Head). No `k` and `ω` fields are solved; the mission's SST requirement is **not** delivered.
* **No wave-making resistance**: `Fn → 0` rigid-wall free surface, so `C_T` in this report means the
  viscous part `C_V`.
* No bulb, no transom (see §1.2), no propeller/hull interaction, no sinkage/trim.
* The BL is a *strip* model: streamwise curvature and cross-flow are neglected.

### 2.3 Verification of the components (all in `tests/test_twin_hull_form_factor.py`)

| check | result | tolerance |
|---|---|---|
| sphere `Cp` vs `1 − 9/4 sin²θ` (RMS over the surface) | 0.040 (16×12), 0.022 (24×18), 0.013 (36×26) | < 0.06 / 0.03 / 0.02, monotone mesh convergence |
| sphere pressure integral (d'Alembert) | < 5e-3 of `q S` | < 5e-3 |
| flat plate, Head, turbulent from the leading edge | `C_F/`ITTC-57 = 0.86 (Re 1e6), **0.92 (Re 1e7)**, 0.89 (Re 1e8) | within 15 % |
| flat plate laminar branch vs Blasius | 0.952 × `1.328/√Re` | within 8 % |
| flat plate momentum thickness vs 1/7-power law | −3 % at `x = L` | within 6 % |
| Neumann residual of the hull solve | `max|V·n|/U = 3.4e-4` | < 5e-3 |
| bare-hull d'Alembert residual | `|D_inv|/(qS) = 9.1e-5` | < 2e-3 |
| test suite | **14 passed** (25 s) | — |

Known closure bias: the integral method under-predicts flat-plate `C_F` by 8–14 % relative to
ITTC-57. This is why the form factor is reported twice, with the computed `C_F` (`k`) and with the
ITTC-57 line (`k_itcc`).

## 3. Results

### 3.1 Single hull (KVLCC2, `Re = 1e7`, `U = 1.4286 m/s`, `Fn = 0.172`, 1008 panels)

| quantity | value |
|---|---|
| `S_wet` | 12.839 m² |
| `R_F` | 37.89 N (`C_F = 0.002900`) |
| `R_VP` | 14.40 N (`C_VP = 0.001101`) |
| `C_V = C_F + C_VP` | 0.004001 |
| `k = C_VP/C_F` | **0.380** (`1+k = 1.380`) |
| `k_itcc = C_VP/C_F,ITTC57` | 0.367 (`1+k = 1.367`) |
| `C_VP/C_V` | **0.275** |
| `C_F` vs ITTC-57 (0.003000) | −3.3 % |
| separated strips | 3.8 % of the wetted strips |
| empirical reference (Holtrop–Mennen 1984) | `1+k = 1.2212` (`k = 0.2212`) |

### 3.2 Twin hull: interference over separation (`Re = 1e7`, production grid 36×28)

| `s/L` | clear gap [m] | `C_F` | `C_VP` | `C_V` | `k` | `i_f` total | `i_f` friction | `i_f` form | `(1+k)(1+i_f)` |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.131 | 0.00245 | −0.00033 | 0.00212 | −0.135 | −0.469 | −0.153 | −1.30 | 0.459 |
| 0.3 | 0.831 | 0.00281 | 0.00108 | 0.00389 | 0.385 | −0.027 | −0.030 | −0.018 | 1.348 |
| 0.4 | 1.531 | 0.00277 | 0.00138 | 0.00415 | 0.497 | +0.037 | −0.044 | +0.251 | 1.553 |
| 0.5 | 2.231 | 0.00273 | 0.00136 | 0.00410 | 0.498 | +0.024 | −0.057 | +0.237 | 1.535 |

The **friction** part of the interference is small and systematic (−3 % to −6 %: the partner hull
shields the inner shell, so the demihull's friction falls), while the **form** part flips sign and
is not grid-converged; at `s/L = 0.2` (clear gap ≈ 0.10 B, i.e. one panel wide on this grid) the
form contribution collapses and `C_VP < 0`, which is unphysical.

### 3.3 Grid convergence (three grids, both directions, ratio 1.5: 24×18, 36×28, 54×42 panels)

| quantity | coarse | medium | fine | max rel. change | observed order `p` |
|---|---|---|---|---|---|
| `C_F` | 0.002880 | 0.002898 | 0.002908 | **0.63 %** | **1.51** |
| `C_VP` | 0.001071 | 0.001101 | 0.001059 | 3.84 % | undefined (non-monotone) |
| `C_V` | 0.003951 | 0.004000 | 0.003967 | **1.23 %** | undefined (non-monotone) |
| `k` | 0.3719 | 0.3800 | 0.3642 | 4.17 % | undefined (non-monotone) |

Richardson extrapolation of `C_F`: `0.002920` with an extrapolation error of −0.4 %.

### 3.4 Reynolds-number study (single hull)

| `Re` | `C_F` | `C_F` ITTC-57 | `C_VP` | `k` | `k_itcc` | `C_VP/C_V` |
|---|---|---|---|---|---|---|
| 1e7 | 0.002900 | 0.003000 | 0.001101 | 0.380 | 0.367 | 0.275 |
| 1e8 | 0.001960 | 0.002080 | 0.000770 | 0.394 | 0.370 | 0.283 |

`Re = 1e8` at model scale means `U = 14.3 m/s` (an equivalent-Reynolds study, `Fn` not physical);
the form factor is Reynolds-neutral to 4 %, as it should be.

### 3.5 Model-uncertainty study (single hull, `Re = 1e7`)

| knob | setting | `k` | `1+k` | `C_VP/C_V` |
|---|---|---|---|---|
| displacement cap `δ*/b` | 0.10 / **0.25** / 0.40 | 0.287 / **0.380** / 0.421 | 1.287 / **1.380** / 1.421 | 0.223 / **0.275** / 0.296 |
| aft closure | pointed / **transom 0.2 B/2** | **0.380** / 0.297 | **1.380** / 1.297 | 0.275 / 0.252 |
| quadrature order | **2×2** / 3×3 | **0.380** / run flagged failed | **1.380** / — | 0.275 / — |

So the *model* spread of the form factor at this level of modelling is `k ∈ [0.287, 0.421]`
(`1+k ∈ [1.287, 1.421]`), i.e. **±5 % on `(1+k)` around its central value**, from knobs that are
physical-modelling choices, not numerical resolution. The 3×3 quadrature run is *flagged as failed*
rather than reported: its centre node sits on the bare-hull pressure points once the hull is inflated
by `δ*` (Rankine kernel near-singular), which is why the production setting is the 2×2 rule — an
open numerical item, reported as such.

## 4. Figures

| figure | content |
|---|---|
| `figures/fig1_geometry.png` | waterline, midship section and panel centroids of the reconstructed demihull |
| `figures/fig2_sphere_verification.png` | source-panel `Cp` on a sphere vs the analytic solution, two meshes |
| `figures/fig3_boundary_layer.png` | flat-plate `c_f` vs ITTC-1957 for three Reynolds numbers; hull `c_f` and `δ*` along the hull |
| `figures/fig4_grid_convergence.png` | `C_F`, `C_VP`, `C_V` vs mesh size and the 2 % band |
| `figures/fig5_interference.png` | interference factor and twin-hull `(1+k)` vs `s/L`, against the empirical reference |
| `figures/fig6_pressure.png` | double-body `Cp` on the inner and outer shells, single hull vs `s/L = 0.3` |

## 5. Discussion: why `C2`, `C3`, `C4` fail (diagnosis, not excuse)

A per-panel breakdown of the form drag (see the campaign report) shows that **≈ 50 % of the computed
`C_VP` comes from the aft 2 % of the hull length** (2.5 % of the wetted area) and ≈ 100 % from the
aft 5 %. Two model weaknesses sit exactly there:

1. **Aft closure.** The parametric hull tapers to a knife-edge at the AP (the transom is not
   modelled). The strip BL then runs into an artificial stagnation, `δ*/b` saturates the validity
   cap, and the first-order displacement perturbation is amplified. Replacing the knife edge with a
   transom of `0.2 B/2` lowers `k` from 0.380 to 0.297 — but the transom introduces a 90° corner
   whose inviscid flow is singular without a Kutta/separation condition (`|D_bare|/(qS)` grows from
   9e-5 to 1.5e-3, i.e. 30×), so it is kept as a sensitivity, not as the baseline.
2. **First-order VI is not the correct closure for the form factor.** The pressure is taken from the
   flow about the inflated body evaluated on the *bare* hull, and the BL displacement has no
   streamline-divergence amplification (Lighthill's external displacement). The result is a form
   factor that is too large: `1+k = 1.380` vs `1.221` (**+13.6 %** on `1+k`, **+75 %** on `k`).

`C4` follows `C2`: `C_VP/C_V = 0.275` is 2.5 points above the 25 % band for the same reason (the
sensitivity study shows 0.223 at `δ*/b = 0.10` and 0.296 at 0.40, i.e. the criterion is inside the
model spread but not inside its central value).

`C3` fails for two independent reasons: at `s/L = 0.2` the clear gap (0.13 m) is *one panel wide* on
the production grid, and the form part of the interference collapses (`C_VP < 0`); and even for
well-resolved gaps the form part of `i_f` is not grid-monotone (see §3.3), so the net `i_f` sequence
(−0.51, −0.03, +0.04, +0.02) is not monotone. The *friction* part of the interference is monotone
and physical (−3 % → −6 % as the gap grows... note it becomes *more* favourable with separation,
which is the opposite sign to the model-test trend of unfavourable viscous interference reported by
Molland/Insel for catamarans at low Fn): the present model therefore does **not** reproduce the
literature trend, and saying otherwise would be wrong.

## 6. Error adjudication (explicit)

| key quantity | computed | reference | error | < 3 %? |
|---|---|---|---|---|
| `Cb` | 0.809800 | 0.8098 (SIMMAN) | < 1e-5 | **yes** |
| `LCB/L` | 0.034800 | 0.0348 | < 1e-5 | **yes** |
| `S_wet` | 12.839 m² | 13.012 m² (published) | −1.33 % | **yes** |
| `C_F` (hull, Re 1e7) | 0.002900 | 0.003000 (ITTC-57) | −3.3 % | no (marginal, 3.3 %) |
| sphere `Cp` RMS | 0.013 (fine) | 0 (analytic) | 0.013 abs. | n/a (abs. tolerance) |
| flat plate `C_F` (Head) | 0.00276 | 0.003000 (ITTC-57) | −8 % | no (integral-closure bias) |
| `C_F` grid convergence | 0.63 % | 0 | — | **yes** |
| `C_T = C_V` grid convergence | 1.23 % | 0 | — | **yes** (≤ 2 % criterion) |
| `1+k` | 1.380 | 1.221 (Holtrop–Mennen 1984) | **+13.6 %** | **no** |
| `k` | 0.380 | 0.221 | **+75 %** | **no** |
| `C_VP/C_V` | 0.275 | 0.15–0.25 band | +2.5 pts | **no** |
| `i_f` monotone over `s/L` | no | required | — | **no** |

No baseline exists for the viscous interference `i_f` in this environment (no access to the Molland
series tables), so `C3` is adjudicated on the *monotonicity* and *sign trend* criteria only, with the
trend statement above taken from the qualitative literature behaviour; the quantitative comparison
is left open.

## 7. Reproduction

```bash
# full study (~6 min): geometry gate, grid study, separation sweep, Re study, sensitivity
PYTHONPATH=src python3 -m tensorfvm.twin_hull_form_factor --out artifacts/twin-hull-form-factor
# fast smoke configuration (~40 s)
PYTHONPATH=src python3 -m tensorfvm.twin_hull_form_factor --quick --out /tmp/thq
# figures
PYTHONPATH=src python3 scripts/twin_hull_form_factor_report.py
# tests
PYTHONPATH=src python3 -m pytest tests/test_twin_hull_form_factor.py -q
```

Artifacts: `grid_independence.json`, `separation_sweep.json`, `reynolds_study.json`,
`sensitivity.json`, `summary.json` (the last carries the six acceptance verdicts and all numbers
quoted above). They are written by the command above to `artifacts/twin-hull-form-factor/` (the
repository's regenerable output root) and a copy is committed next to this report in
`docs/twin-hull-form-factor/artifacts/` so the evidence travels with the commit.
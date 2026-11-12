# D1/D2 decomposition -- is the dam-break gate a model error or a measurement limit?

Auto-rendered by `docs/GAP-FVM-TWOPHASE/render_d1d2_decomposition.py` from `artifacts/d1d2_audit/` (`python -m tensorfvm.twophase_gap.dambreak_audit`).  **No verdict is changed:** D1/D2 remain honest FAIL on every grid below; this document only decomposes *why* they read what they read.

## 1. Case and parameters

Gravity-driven 2D dam break, aspect-2 column (Martin & Moyce 1952, *Phil. Trans. R. Soc. A* **244**, 312): `a = 0.05715 m` half-width, `h0 = 0.1143 m`, domain `0.6 x 0.2 m`, `rho_l/rho_g = 1000`, `mu_l = 1e-3 Pa s`, `g = 9.81 m/s^2`, PLIC geometric VOF, void free surface.  Reference: digitised experiment, `t* = t sqrt(g/a)`, `x* = (x_f - a)/a`, `h* = h/h0`.  Gates D1 (front) and D2 (height) are `max relative error <= 0.15` over `0.3 < t* <= 4.0`.

## 2. Method

The same solver leg is run over a refinement sequence.  For each grid three quantities are extracted from the leg's own history by pure functions in `tensorfvm.twophase_gap.dambreak_gate`:

* `decompose_dambreak_error` -- (i) the *time shift* that best aligns the curves (`ref(t - delta)` in x\* units), (ii) the late-window (`t* >= 2`) least-squares slope ratio `computed/reference`, (iii) a pure time-*scale* fit `ref(t/s)` as a falsification control for the reference's normalisation convention (a `sqrt(2g/a)` vs `sqrt(g/a)` clash would show up as a scale, not a shift);
* `resolvable_error_floor` -- the smallest relative error the cell-quantised observable can express, `quantum/|ref|`, and the share of window samples where that floor alone already exceeds the 15 % gate;
* the gate values themselves, copied verbatim from the leg records.

Both helpers are pure functions of the sampled curves and are unit tested against synthetic histories with a known shift / gain / scale in `tests/test_dambreak_error_decomposition.py`; a further test asserts the diagnostics cannot change a gate verdict.

## 3. Results

### 3.1 Gates (unchanged) and the decomposition

| grid | steps | D1 front | D2 height | front shift [t*] | front slope ratio | front RMS x\* (none -> shifted) | height shift [t*] | height slope ratio |
|---|---|---|---|---|---|---|---|---|
| 60x20 | 82 | 0.839 FAIL | 0.798 FAIL | 0.87 | 1.307 | 0.771 -> 0.132 | 0.77 | 2.437 |
| 120x40 | 169 | 0.988 FAIL | 0.845 FAIL | 0.82 | 1.310 | 0.766 -> 0.120 | 0.96 | 3.141 |
| 240x80 | 343 | 0.848 FAIL | 0.826 FAIL | 0.75 | 1.327 | 0.682 -> 0.117 | 0.97 | 2.705 |

D1 as a function of resolution is **not monotone** (60x20: 0.839, 120x40: 0.988, 240x80: 0.848) while the decomposed phase lag falls smoothly (0.87, 0.82, 0.75 t\*): the gate number is dominated by early-time samples, the phase lag is the physically meaningful trend.

### 3.2 Time-scale falsification control

| grid | best pure time scale | RMS at best scale | RMS at best shift |
|---|---|---|---|
| 60x20 | 1.540 | 0.289 | 0.132 |
| 120x40 | 1.530 | 0.301 | 0.120 |
| 240x80 | 1.445 | 0.282 | 0.117 |

### 3.3 Cell-quantisation floor under the 15 % gate

| grid | front quantum [x\*] | front relative floor at window start | front window samples where the floor alone violates D1 | height quantum [h\*/h0] | height samples where the floor alone violates D2 |
|---|---|---|---|---|---|
| 60x20 | 0.175 | 0.376 | 0.16 | 0.0875 | 0.84 |
| 120x40 | 0.087 | 0.192 | 0.03 | 0.0437 | 0.47 |
| 240x80 | 0.044 | 0.096 | 0.00 | 0.0219 | 0.00 |

Figures: `artifacts/d1d2_audit/figures/fig_d1d2_audit.png` (front, height, gate-vs-floor) and `fig_phase_lag.png` (RMS misfit against the time shift).

## 4. Findings

1. **The D1/D2 failure is a delayed release, not a slow wave.**  Over `t* >= 2` the computed front slope is *steeper* than the reference on every grid (ratio > 1), and a single time shift of 0.75-0.87 t\* removes ~83 % of the front RMS misfit.  The column starts to drain late, then overtakes the experimental rate.
2. **The phase lag is essentially grid independent** (0.87 -> 0.82 -> 0.75 t\* from 60x20 to 240x80), so refining the mesh will not remove it — consistent with the committed finding that 120x40 -> 240x80 only moves D1 from 0.988 to 0.848.
3. **A reference-normalisation error is not supported.**  A pure time scale fits worse than a pure shift on every grid, and the best scale is grid dependent, so a `sqrt(2g/a)`/`sqrt(g/a)` clash alone cannot explain the misfit.  (Neither diagnostic *proves* the digitised anchor is right: it is a figure read-off, ±few %.)
4. **On the coarse grids the 15 % gate is partly a measurement limit.**  The front observable moves in whole `dx` steps, so at the window start the smallest expressible relative error is 0.19 (120x40) and 0.38 (60x20) — above the gate — and the height observable is even worse: at 60x20 one `dy` step is 8.75 % of `h0`, i.e. 84 % of the window samples have a quantisation floor above 15 %.  D2 at 60x20/120x40 therefore cannot be read as a 15 % accuracy bound.
5. **D1 is non-monotone in resolution while the phase lag is monotone** — the max-over-window metric is dominated by early-time samples where one cell of front motion is tens of percent.

## 5. Round-3 plan implied by this audit (not implemented here)

1. Gate the *decomposed* quantities, not only the raw max: require `phase_lag_t_star <= 0.25` in addition to the relative-error gate, so a late release is measured directly instead of being aliased into a max-relative-error number.
2. Declare the resolution together with the gate, and require `quantum/|ref| <= gate/3` over the comparison window (240x80 is the first grid that satisfies this for the front: floor 0.096).
3. Attack the release itself: the void free-surface projection must carry the hydrostatic head into horizontal momentum at `t* < 0.5` (the late slope is already 31 % *too fast*, so a stronger driving term alone would overshoot; the target is an earlier start, not a bigger wave).

## 6. Error judgement vs the reference (explicit)

| quantity | grid | value | gate | < 3 %? |
|---|---|---|---|---|
| D1 front max rel error | 60x20 | 0.839 | 0.15 | **no** |
| D1 front max rel error | 120x40 | 0.988 | 0.15 | **no** |
| D1 front max rel error | 240x80 | 0.848 | 0.15 | **no** |
| D2 height max rel error | 60x20 | 0.798 | 0.15 | **no** |
| D2 height max rel error | 120x40 | 0.845 | 0.15 | **no** |
| D2 height max rel error | 240x80 | 0.826 | 0.15 | **no** |
| front rel error after the best time shift | 60x20 | 2.853 | — | **no** (shift-only alignment is a diagnostic, not an accepted result) |
| volume drift | all grids | 4.11e-16 | 1e-2 | yes |

Nothing here converts the honest FAIL into a PASS: the front and height of the capability are still outside 15 % of the experiment on every grid tested.

## 7. Reproduction

```bash
PYTHONPATH=src python -m tensorfvm.twophase_gap.dambreak_audit \
    --grids 60x20 120x40 --out docs/GAP-FVM-TWOPHASE/artifacts/d1d2_audit
python docs/GAP-FVM-TWOPHASE/render_d1d2_decomposition.py
PYTHONPATH=src python -m pytest tests/test_dambreak_error_decomposition.py -q
```


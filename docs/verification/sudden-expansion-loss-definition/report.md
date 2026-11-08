# Sudden-expansion loss coefficient: which K, at which station? (m6489)

**Case**: planar sudden expansion, area ratio 4, `Re_D = 100`, parabolic inlet, 12 D downstream channel, 3 D upstream channel (`nu = 0.01`, `U_in = 1`).
Pure post-processing of the committed snapshots; no solver call, no cluster.

## 1. Verdict

The loss coefficient of this case is **not a well-posed single number** in the
committed evidence chain. Three definitions -- each used somewhere in the chain --
evaluate on the *same* snapshots to:

| definition | where it is used | D/10 | D/20 | error vs Borda-Carnot 0.5625 (D/10 / D/20) |
|---|---|---|---|---|
| `K_flux` (mass-flux weighted total pressure) | `incompressible-verification.md` section 2.2, `k_total_pressure` in the raw provenance | 0.5552 | 0.6052 | +1.3 % / +7.6 % |
| `K_signed` (signed-flux mean total pressure) | control-volume energy balance | 0.2833 | 0.3127 | +49.6 % / +44.4 % |
| `K_bulk` (bulk-velocity total pressure) | `baseline-runs/expansion_n0*.json`, machine criteria `expansion.K.*` | 0.9706 | 0.9718 | +72.6 % / +72.8 % |

Max/min ratio across definitions at the same grid: **3.43x** (D/10), **3.11x** (D/20).
The recorded chains are even further apart: the doc/provenance family records
`0.5552` / `0.6052`, the baseline family records `1.0260` (n010), `1.1058` (n020), `1.1313` (n030), `0.8885` (long_n010).
So the same case carries a **PASS** (-1.3 % vs Borda-Carnot, doc section 2.2, D/10) and a
**FAIL** (101 % vs Borda-Carnot, `expansion.K.finest`, D/30) verdict inside one evidence
bundle. Neither is an accuracy statement; see 2-4.

## 2. Why the definitions disagree (quantified, not asserted)

**(a) The inlet profile.** `K_bulk` assumes the kinetic energy entering the station is
`rho U_in^2 / 2` per unit mass. The committed field carries
`<u^3> / U_bulk / q_in` = **1.332** (D/10) / **1.416** (D/20) times that, because the inlet
profile is parabolic and not yet uniform. Decomposing
`K_bulk - K_flux = (KE_in - 1) - (KE_out - 1)(U_out/U_in)^2 + (pressure-averaging)` (programmatic, `results.json:definitions.<grid>.bulk_minus_flux`):

| grid | `K_bulk - K_flux` | kinetic-energy term | share | residual (pressure averaging) |
|---|---|---|---|---|
| D/10 | +0.4154 | +0.3265 | 79 % | +0.0889 |
| D/20 | +0.3666 | +0.4067 | 111 % | -0.0401 |

So the bulk definition books the parabolic-profile kinetic energy as *loss*; the
remainder is the difference between a mass-flux weighted and an area-weighted mean
pressure. Borda-Carnot 0.5625 is a *bulk-velocity* formula derived for a **uniform**
inlet, so it is not comparable with a case driven by a parabolic profile under either
definition.

**(b) The station.** The downstream station `x/D = +2` sits *inside* the recirculation:
12.7 % (D/10) / 13.0 % (D/20) of the mass flux crossing that plane
is reversed. Weighting the mean total pressure by `|u|` and by `u` therefore give
0.555 vs 0.283 (D/10) -- the factor 2.0. Only the
signed average appears in the control-volume energy balance; the `|u|` average is the
conventional CFD post-processing choice and is the one the doc uses.

**(c) The accumulation.** `K` measured between two stations is the loss *accumulated* up
to the downstream station, and it keeps growing inside the 12 D domain:

| grid | K at x/D=2 (gate) | K at the last station | share accumulated at the gate |
|---|---|---|---|
| D/10 | 0.5552 | 0.7543 | 74 % |
| D/20 | 0.6052 | 0.8017 | 75 % |

The profile is non-decreasing apart from small local dips (D/10: 14 steps, worst -0.0103 = 1.4 % of the final value; D/20: 30 steps, worst -0.0204 = 2.5 %), i.e. no station inside this domain measures a
*converged* loss -- the recirculation is still open at the outflow (see the m6383
symmetry audit, section 3.1: `X_r/D` is WITHHELD in 12 D, 17.08 at D/10 in 25 D).

## 3. The grid pair does not converge in K either

`K_flux(D/20) - K_flux(D/10)` over `x/D in [1, 11.5]`:

* mean 9.3 %, min 9.1 %, max 10.0 %;
* at the gate station x/D = 2: **9.0 %**.

The gap is *flat* in x (figure c), i.e. a systematic resolution offset, not the
decaying difference that a converging sequence would show. Against a 5 % gate this
means: the D/10 PASS (-1.3 %) is a near-cancellation of (i) under-accumulated loss at
the gate station and (ii) the coarse grid's deficit, and the D/20 FAIL (+7.6 %) is the
same number read from the other side. **The gate discriminates the grid, not the
accuracy.**

## 4. Reproduction of the recorded values (evidence binding)

| grid | recorded `k_total_pressure` | recomputed here | relative gap |
|---|---|---|---|
| D/10 | 0.5552 | 0.5552 | 0.00 % |
| D/20 | 0.6052 | 0.6052 | 0.00 % |

The flux-weighted numbers are reproduced from the snapshots (gap < 0.01 %, i.e. the
snapshots *are* the fields behind the recorded raw JSON). The baseline-family values
(`K_bulk` recorded 1.0260 at D/10) are
**not** reproducible from these snapshots
(recomputed `K_bulk` = 0.9706), which shows the machine criteria and the doc
table are computed from different runs *and* different definitions.

## 5. Notes, limits, recommendations

* Limit: only two snapshots exist (D/10, D/20); the D/30 and 30 D-domain runs are
  recorded only as JSON, so no third grid could be audited field-by-field.
* Limit: this audit fixes no solver code. It is a measurement audit -- the fix belongs in
  the *criterion*, not in the discretisation.
* Recommendation 1 (acceptance): replace `expansion.K.finest` (|K - K_BC| / K_BC, D/30)
  by a *self-consistency* gate on the grid pair at a station where the flow is
  unidirectional and redeveloped, e.g. `|K(D/20) - K(D/10)| / K(D/20) <= 10 %` in a
  >= 30 D domain, with the definition pinned to `K_flux` and stated in the criterion text.
* Recommendation 2 (reference): if Borda-Carnot is kept, the comparison must use a
  uniform inlet (plug profile) -- otherwise the parabolic-profile kinetic-energy excess
  (1.33x `q_in`) is charged to the expansion loss.
* Recommendation 3 (documentation): the doc table (section 2.2) and the machine criteria
  (`expansion.K.*`) must name the same definition; today a reader of `results.json` sees
  FAIL 101 % where the doc section 2.2 reports PASS -1.3 % for the same case.

## 6. Error judgement (explicit)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| K_flux, D/10, gate station | 0.5552 | Borda-Carnot 0.5625 | +1.3 % | no |
| K_flux, D/20, gate station | 0.6052 | Borda-Carnot 0.5625 | +7.6 % | no |
| K_bulk, D/10, gate station | 0.9706 | Borda-Carnot 0.5625 | +72.6 % | no |
| K_flux grid pair (D/20 vs D/10) at the gate | 0.6052 | D/10 0.5552 | +9.0 % | no |
| recomputation of the recorded K_flux | 0.5552 | recorded 0.5552 | 0.00 % | yes |

Answer to the standing question: **no** quantity compared against Borda-Carnot is within
3 %, and it cannot be, because the comparison mixes a parabolic-inlet, station-truncated,
recirculation-cutting measurement with a uniform-inlet asymptotic reference. The ver-
dict for this criterion is **WITHHELD (measurement not well posed)**, not PASS and not
FAIL. The only quantity reproduced to <3 % is the recomputation itself (evidence
binding, last row).

## 7. Reproduction

```
python scripts/audit_sudden_expansion_loss_definition.py
pytest -q tests/test_sudden_expansion_loss_definition.py
```

Figure: `fig_loss_definitions.png` -- (a) accumulated K vs station for both grids and
both weightings, (b) the three definitions at the gate station, (c) the grid-pair spread
versus station.

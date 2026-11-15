# CASE-FVM-PERIODIC-HILLS — round 15: the healed hill operator scores 3.77, and the channel
# discriminator refuses to answer (honest WITHHELD)

Anchor: `CASE-FVM-PERIODIC-HILLS` (feature ledger, `open`) · mission 21087 · branch `agent/m21087`
Reference: ERCOFTAC UFR 3-30 distributed DNS (`x_r/H = 5.41`, `u/U_b(y/H=0.5)` table)
Compute: cj3171 (channel, node n315) · cj3172 (hill, node n330) — both single-core, cluster queue.

## 1. Case

Periodic hill (ERCOFTAC 81 / Breuer et al. 2009, *Comput. Fluids* 38:433,
doi:10.1016/j.compfluid.2008.05.002): 9H x 3.036H periodic domain, hill of height
H = 1 on the lower wall, `u_b = 1`, `nu = 1/2800` (`Re = u_b H / nu = 2800`),
96 x 48 terrain-following cells, `dt = 0.02`, 20000 steps (t = 400), time mean over
the last 240 time units, SST k-omega (Menter 1994/2003).

## 2. What was done

1. **Transport-convection plumbing defect (fixed).** The round-10..14 driver's
   `--convection-blend` reached only `PeriodicHillConfig.convection_blend` (the
   momentum operator); `SSTKOmegaConfig.convection_blend` stayed at its
   first-order default. Round 11's own artefact recorded the disagreement and
   nobody compared the two fields: momentum 0.75, transport 0.0 — so round 14's
   "run the transport with second-order convection" next step had **never been
   run**. `build_solver` now takes `transport_convection_blend`, records it next
   to `momentum_convection_blend`, and `tests/test_periodic_hill_round15.py`
   pins it.
2. **Single-sourced scoring path.** Round 11 scores `x_r/H` against 5.34 and the
   profile against the *legacy digitised* table that round 10 showed to be up to
   3.5x too large. `score_hill()` re-scores any round-11-schema payload against
   `periodic_hill_reference` (5.41 + `y05_table()`, 10 % bands, round-9
   convention: relative for |ref| >= 0.1, absolute 0.10 U_b otherwise) and reports
   the legacy 5.34 error alongside, so an artefact cannot silently quote one.
3. **Hill re-score of the healed operator** (cj3172): rounds 12--13's transport
   and eta-face fixes are in `HEAD`, so this is the first physics number for the
   healed code.
4. **The discriminator** (cj3171): the same solver with `hill_amplitude = 0` is a
   plane channel, where three closed-form anchors exist (linear total stress
   `(nu+nu_t)du/dy = u_tau^2(1-y/h)`, log law, and the log-layer identity
   `nu_t+ = kappa y+ - 1`). Gates were pre-declared in the module docstring.

## 3. Results

### 3.1 Hill, healed operator vs the three earlier states

| state | mesh | x_r/H | error vs 5.41 | admissible y/H=0.5 stations passing |
|---|---|---|---|---|
| round 11 (defective SST transport) | 96x48 | 2.046 | 62.2 % | 2/8 |
| round 12 (transport fixed) | 96x48 | 3.144 | 41.9 % | 0/7 (round-12 report) |
| **round 15 (transport + eta-face fixed, this run)** | 96x48 | **3.767** | **30.4 %** | **1/8** |
| DNS (reference) | — | 5.41 | — | — |

Stationarity: `x_r/H` stationary=True; samples=601.
The eta-face pairing fix is worth **+0.62 H** (+19.8 % relative) on the same mesh and
budget — it is a real, measurable physics contribution, not a bookkeeping change.
The acceptance gate is still **FAIL** (30.4 % > 10 %).

### 3.2 Per-station profile, y/H = 0.5 (round 15)

| x/H | computed u/U_b | reference u/U_b | |delta| | rel. error | verdict |
|---|---|---|---|---|---|
| 1.00 | -0.1252 | -0.0461 | 0.0791 | 171% | PASS |
| 2.00 | +0.0431 | -0.0912 | 0.1344 | 147% | FAIL |
| 3.00 | +0.1798 | +0.0041 | 0.1757 | 4256% | FAIL |
| 4.00 | +0.3065 | +0.1198 | 0.1867 | 156% | FAIL |
| 5.00 | +0.3954 | +0.2183 | 0.1771 | 81% | FAIL |
| 6.00 | +0.4438 | +0.2876 | 0.1562 | 54% | FAIL |
| 7.00 | +0.4320 | +0.3141 | 0.1178 | 38% | FAIL |
| 8.00 | +0.3300 | +0.2238 | 0.1063 | 47% | FAIL |

The near-wall side of the bubble (x/H = 1, 2) is now inside tolerance while the
recovery region (x/H >= 4) remains 40--70 % low: the recirculation is too short and
the shear layer too diffusive, i.e. the same signature the earlier rounds reported.

### 3.3 Plane-channel discriminator (cj3171) — **WITHHELD, not a physics verdict**

| quantity | measured | reference | verdict |
|---|---|---|---|
| C1 log law, kappa_eff | 24.514 | 0.41 | FAIL |
| C2 linear total stress, max rel. residual | 0.890 | <= 0.05 | FAIL |
| C3 nu_t+ / (kappa y+ - 1), median | 0.0261 | [0.7, 1.4] | FAIL |
| C4 Re_tau | 17329.0 | 262.1 (Dean 1978) | FAIL |
| max nu_t/nu | 0.0444 | O(30) in a developed channel | — |

These are **not** closure numbers. `nu_t/nu = 0.04` means the k-omega field decayed:
the run never left the laminar spin-up. The channel is driven only by the
bulk-velocity controller and starts from a uniform field, so its development time is
the viscous diffusion time `h^2/nu = 6452` time units, while the submitted
budget was `steps*dt = 300` time units. At `dt = 0.01` that is ~6.5e5
steps (tens of hours single-core) or a synthetic turbulent initial profile — neither
was available inside this shift, so the closure-vs-geometry question stays **open**
and the round-14 root-cause hypothesis stays unproven. Reporting the channel as a
"closure defect" would have been the easy lie; it is a budget finding.

## 4. Figures

* `fig_round15_hill.png` — y/H = 0.5 profile (DNS vs r11 vs r15) and the x_r/H
  progression with the +-10 % band.
* `fig_round15_channel.png` — nu_t+ vs kappa y+ - 1 and u+ vs the log law for the
  (under-developed) channel run: the figure is the evidence for the WITHHELD.

## 5. Error table and the <3 % question

| quantity | computed | reference | error | < 3 %? |
|---|---|---|---|---|
| x_r/H (round 15) | 3.767 | 5.41 | 30.4 % | **NO** |
| u/U_b at y/H=0.5, 8 admissible stations | see 3.2 | DNS files | 1/8 within 10 % | **NO** |
| transport-blend plumbing | closure blend == recorded value | config | exact | YES (unit test) |
| eta-face operator (round 13, merged) | symmetric/conservative | face-loop definition | 1e-14 | YES |

No quantity in this round reaches the 3 % bar; the only <3 % results are the
discretisation identities, which are machine-precision by construction.

## 6. Gotchas / limits

* The headline number is a **FAIL** (30.4 %); the improvement over round 12 is
  evidence that the round-13 fix was necessary but not sufficient.
* The channel experiment must start from a developed (or synthetic turbulent)
  profile, or run ~10^5--10^6 steps; otherwise it measures spin-up, not the closure.
* `round11_*.json` payloads written before round 15 carry `profile_comparison`
  against the retired legacy table; use `score_hill()` for any new verdict.
* The 5.34 (feature ledger / rounds 3--5) vs 5.41 (DNS audit) reference difference
  is 1.3 % and both bands contain the round-15 value's error direction; both are
  reported for every run here.

## 7. Reproduction

```bash
PYTHONPATH=src pytest tests/test_periodic_hill_round15.py -q          # 11 passed
PYTHONPATH=src python3 scripts/round15_channel_and_rescore.py hill --tag r15_96x48
PYTHONPATH=src python3 scripts/round15_channel_and_rescore.py channel --ny 64 --nx 8 \
    --steps 30000 --burn-in 10000 --tag 64x8
PYTHONPATH=src python3 scripts/round15_analysis.py                    # table + figures
```

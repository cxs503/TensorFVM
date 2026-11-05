# Ghia (1982) Re = 400 v-centreline: the proposal, scored both ways

Mission m3240 (reissued as m4004 after PR #82 was closed for being developed off the
current baseline). Status: **proposal registered, transcription untouched**.
Inputs: `docs/verification-lid-cavity-ladder/summary.json` (committed ladder),
`src/tensorfvm/lid_cavity_reference.py` (U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I (u at x = 0.5) and II (v at y = 0.5)),
`docs/GHIA_RE400_V_ATTRIBUTION.md` (the attribution this builds on).
Artefacts: `scripts/render_ghia_re400_v_proposal.py`, `metrics.json` and the three
figures in this directory; gate in `tests/test_ghia_re400_v_proposal.py`.

## 1. Case

Square cavity `[0, L]^2`, no-slip on three walls, lid `u = U` at `y = L`, `L = U = 1`.
Staggered MAC finite volume (`tensorfvm.lid_cavity.LidCavitySolver`), Re = U L / nu in
{100, 400, 1000}, grids n in {32, 64, 128}, CFL 0.15, steady-state tolerance
1e-07 with a 200000 step cap. Reference: U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I (u at x = 0.5) and II (v at y = 0.5). Scoring interpolates
the converged mid-plane profile to the tabulated coordinates with a Catmull-Rom cubic.

## 2. Method

1. The attribution audit proved the extraction innocent: at the disputed point the
   solver value converges at clean second order (2.00) to a Richardson limit of
   -0.38989, the interpolation error of the metric path at n = 128 is bounded by
   1.0e-03 (two orders below the gap), and the printed entry is the only corner in
   all six tables.
2. This script therefore scores every committed ladder row **twice**, from the stored
   profiles: once against `table(Re, comp)` (as printed, 7 rows) and once against
   `corrected_table(Re, comp)`, which substitutes the registered proposal
   (400/v/0.9063: -0.23827 -> -0.38989) and nothing else.
3. The proposal is `status="proposed"`, `applied=False`. It is reachable only through the
   opt-in `corrected=True` path; no verification artefact, gate or committed number in
   this repository reads it.

## 3. Results

The disputed point, per resolution (`x = 0.9063`, `y = 0.5`):

| n | solver v | printed v | delta vs printed | proposal | delta vs proposal |
|---|----------|-----------|------------------|----------|-------------------|
| 32 | -0.35284 | -0.23827 | -0.11457 | -0.38989 | +0.03705 |
| 64 | -0.38064 | -0.23827 | -0.14237 | -0.38989 | +0.00925 |
| 128 | -0.38758 | -0.23827 | -0.14931 | -0.38989 | +0.00231 |

Re = 400 v-centreline error, the only row in the ladder that moves:

| n | max abs err, as printed | <3 %? | max abs err, proposal applied | <3 %? |
|---|------------------------|-------|-------------------------------|-------|
| 32 | 0.1146 | NO | 0.0371 | NO |
| 64 | 0.1424 | NO | 0.0093 | YES |
| 128 | 0.1493 | NO | 0.0043 | YES |

All seven committed rows, both scorings (errors in units of U):

| Re | n | u err | v err (printed) | v err (proposal) | gate printed | gate proposal | max div |
|----|---|-------|-----------------|------------------|--------------|---------------|---------|
| 100 | 32 | 0.0069 | 0.0078 | 0.0078 | pass | pass | 1.4e-14 |
| 100 | 64 | 0.0045 | 0.0087 | 0.0087 | pass | pass | 9.2e-14 |
| 100 | 128 | 0.0049 | 0.0091 | 0.0091 | pass | pass | 6.8e-13 |
| 400 | 32 | 0.0684 | 0.1146 | 0.0371 | FAIL | FAIL | 7.2e-15 |
| 400 | 64 | 0.0066 | 0.1424 | 0.0093 | FAIL | pass | 3.2e-14 |
| 400 | 128 | 0.0013 | 0.1493 | 0.0043 | FAIL | pass | 2.5e-13 |
| 1000 | 64 | 0.0293 | 0.0197 | 0.0197 | pass | pass | 2.1e-14 |

Worst error over the whole ladder: **0.1493** as printed,
**0.0684** with the proposal applied. Rows failing the 3% pointwise
gate: printed **400/n32, 400/n64, 400/n128**, proposal **400/n32**.

Figures: `re400_v_disputed_point.png` (the entry, the solver sequence and the proposal),
`v_error_vs_resolution.png` (error vs n, both scorings), `all_centrelines_n128.png`
(all six centrelines at the finest committed grid).

## 4. Notes and limits

* Nothing is corrected. `table(Re, comp)` returns the transcription byte for byte;
  `self_check()` now also proves that the proposal layer touches exactly the registered
  entry, moves no extremum, and applies nothing by default (`self_check()["proposals"]`).
* The proposal is the solver's Richardson limit, not an independent measurement. That is
  a hypothesis about a printed number, and it is labelled as one: adjudicating it needs
  the printed Table II or an independent DNS at `(x = 0.9063, y = 0.5)`.
* The proposal does not move the extrema, so the relative-error criteria of
  `ghia_acceptance` are unaffected; only the pointwise v criterion changes.
* `ghia_acceptance` still fails by default and the strict xfail in
  `tests/test_lid_cavity_re1000_evidence.py` stays red. A pass is available only through
  `allow_registered_disputes=True`, which reports every ignored point back.

## 5. Error assessment against the reference

| quantity | baseline | value | <3 %? |
|----------|----------|-------|-------|
| Re = 100 u (n = 32) | Ghia Table I, as printed | max abs err 0.0069 | YES |
| Re = 100 v (n = 32) | Ghia Table II, as printed | max abs err 0.0078 | YES |
| Re = 100 divergence (n = 32) | exact (div u = 0) | 1.4e-14 | YES |
| Re = 100 u (n = 64) | Ghia Table I, as printed | max abs err 0.0045 | YES |
| Re = 100 v (n = 64) | Ghia Table II, as printed | max abs err 0.0087 | YES |
| Re = 100 divergence (n = 64) | exact (div u = 0) | 9.2e-14 | YES |
| Re = 100 u (n = 128) | Ghia Table I, as printed | max abs err 0.0049 | YES |
| Re = 100 v (n = 128) | Ghia Table II, as printed | max abs err 0.0091 | YES |
| Re = 100 divergence (n = 128) | exact (div u = 0) | 6.8e-13 | YES |
| Re = 400 u (n = 32) | Ghia Table I, as printed | max abs err 0.0684 | NO |
| Re = 400 v (n = 32) | Ghia Table II, as printed | max abs err 0.1146 | NO |
| Re = 400 v (n = 32) | Ghia Table II, proposal applied | max abs err 0.0371 | NO |
| Re = 400 divergence (n = 32) | exact (div u = 0) | 7.2e-15 | YES |
| Re = 400 u (n = 64) | Ghia Table I, as printed | max abs err 0.0066 | YES |
| Re = 400 v (n = 64) | Ghia Table II, as printed | max abs err 0.1424 | NO |
| Re = 400 v (n = 64) | Ghia Table II, proposal applied | max abs err 0.0093 | YES |
| Re = 400 divergence (n = 64) | exact (div u = 0) | 3.2e-14 | YES |
| Re = 400 u (n = 128) | Ghia Table I, as printed | max abs err 0.0013 | YES |
| Re = 400 v (n = 128) | Ghia Table II, as printed | max abs err 0.1493 | NO |
| Re = 400 v (n = 128) | Ghia Table II, proposal applied | max abs err 0.0043 | YES |
| Re = 400 divergence (n = 128) | exact (div u = 0) | 2.5e-13 | YES |
| Re = 1000 u (n = 64) | Ghia Table I, as printed | max abs err 0.0293 | YES |
| Re = 1000 v (n = 64) | Ghia Table II, as printed | max abs err 0.0197 | YES |
| Re = 1000 divergence (n = 64) | exact (div u = 0) | 2.1e-14 | YES |

Answer, stated plainly: **as printed, the Re = 400 v centreline is NOT inside 3 % at any
committed resolution (0.1493 worst, and the error grows with refinement), so no
amount of extra grid closes it.** With the proposal applied the same measurements are inside
3 % at every resolution that resolves the lid shear layer, n >= 64 (worst 0.0093);
the n = 32 row still misses on the proposal (0.0371) but that row already misses on
u as printed (0.0684), i.e. it is a resolution limit and not a reference question. Both
columns are printed side by side on purpose: the second one is conditional on a correction
nobody has approved, and quoting it alone is exactly the failure mode this artefact exists to
prevent.

## Reproduce

```bash
PYTHONPATH=src python scripts/render_ghia_re400_v_proposal.py     # regenerates this file
PYTHONPATH=src python -m pytest tests/test_ghia_re400_v_proposal.py -q
```

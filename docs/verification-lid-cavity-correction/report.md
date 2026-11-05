# Ghia (1982) lid-driven cavity, Re = 400 v-centreline: the reference edit is **rejected**

**Mission:** m4105 (round 2 of PR #126, superseding the round-1 claim of "corrected transcription")
**Status:** ⚠️ REJECTED — the published entry stays as printed; the discrepancy stays registered; CI restored.

This note replaces the round-1 conclusion of mission m3240 / m3884
(`V_CENTRELINE[400.0][0.9063]: -0.23827 -> -0.38760`, described there as a
"transcription error"). The edit is **not** adopted on `main`, for the three
reasons below; the numbers are regenerated, not transcribed, by
`scripts/audit_m4105_reference_edit_rejection.py` and
`scripts/m4105_ci_remediation_report.py`
(`docs/m4105-ci-remediation/reference_edit_audit.json`, `docs/m4105-ci-remediation/metrics.json`).

## 1. Policy the edit violated

`src/tensorfvm/lid_cavity_reference.py` states its own contract next to the
register:

> A dispute is *not* a licence to edit the transcription: the published number
> stays exactly as printed and any gate that ignores it must say so out loud.

The register entry for `(400.0, "v", 0.9063)` also carries the adjudication
requirement "needs the printed Table II or an independent DNS; offline copies of
the paper are not available in this checkout". Neither artefact exists in this
checkout, so the entry cannot be adjudicated — only registered.

## 2. The proposed value is the solver's own extrapolation (circular verification)

| Quantity | Value |
|---|---|
| Published transcription kept in the table | -0.23827 |
| Rejected replacement (round 1) | -0.38760 |
| Solver at n = 128 | -0.38758 |
| Registered Richardson limit (order 2.00) | -0.38989 |

The replacement is the solver's Richardson limit (0.6 % away from it) and is
numerically indistinguishable from the solver's own n = 128 value (2e-5 apart).
Scoring the solver against it would make the verification circular: the metric
would compare the solution with its own extrapolation, and the "inside 3 %"
statement would be true by construction. That is exactly the failure mode the
repository's gate exists to prevent.

## 3. The consistency gate fails on the edit (3 tests), not on the published table

With the published table the audit is clean: exactly one corner is flagged in all
six tables (`400/v/0.9063` plus the neighbour whose secant is flattened by it),
and it is the registered one. With the replacement applied in memory the audit
reports `flagged = []` and `registered_flagged = []` — the register's fingerprint
disappears, so `tests/test_ghia_reference_consistency.py` fails on

* `test_the_only_corner_in_all_six_tables_is_the_registered_dispute` (`audit["ok"]` false),
* `test_the_gap_survives_extrapolation_while_its_neighbours_do_not`
  (gap 1.8e-5 vs 5 x the worst neighbour gap 0.0215),
* `test_the_metric_path_cannot_manufacture_the_gap` (gap/bound = 0.04 vs the 100x requirement).

The published table satisfies all three: gap 0.1493 (62.7 % of the tabulated
value) = **149 x** the metric-path interpolation error bound (1.0e-3) and
**35 x** the worst neighbour gap — the gap is a property of the transcribed
number, not of the extraction path. That is why the debt is registered instead of
being edited away.

## 4. What stands

* `V_CENTRELINE[400.0][0.9063] = -0.23827` (as published; unchanged).
* `DISPUTES[(400.0, "v", 0.9063)]` remains the open-question record; the round-1
  `"resolution"` field asserting the fix is removed.
* The strict `xfail` in `tests/test_lid_cavity_re1000_evidence.py` stays: the
  Re = 400 v-centreline row still exceeds the 3 % band, and the suite says so out
  loud instead of hiding it (see `docs/m4105-ci-remediation/metrics_table.md`: Re = 400
  v max|err| = 0.1146 / 0.1424 / 0.1493 at n = 32 / 64 / 128, while u is
  0.0684 / 0.0066 / 0.0013).
* Closing the debt needs one of: the printed Table II, an independent
  (DNS / spectrally converged) v-profile at Re = 400, or the original digitised
  data. Until then the honest statement is "published value not reproducible at
  any resolution we can afford; extraction path exonerated".

## 5. Figures

`docs/m4105-ci-remediation/reference_edit_audit.png` (left: Re = 400 v-centreline from the
committed ladder against the published points, with the rejected replacement
marked; right: v(x = 0.9063) versus 1/n^2 converging to the registered limit, both
candidate reference values drawn as horizontal guides). The round-1 images in
this directory illustrated the rejected proposal and were removed to keep the tree
free of figures that contradict the accepted data.

## 6. Error assessment (honest)

| Quantity | Reference | Error | < 3 % ? |
|---|---|---|---|
| Re = 100 u / v centreline | Ghia Table I / II | 0.0045-0.0069 / 0.0078-0.0091 | YES |
| Re = 400 u centreline | Ghia Table I | 0.0013 (n = 128) | YES |
| Re = 400 v centreline | Ghia Table II | 0.1493 (n = 128), grows with refinement | **NO** |
| Re = 1000 u / v centreline (n = 64) | Ghia Table I / II | 0.0293 / 0.0197 | YES (borderline for u) |
| divergence, all rows | exact | <= 6.8e-13 | YES |

The Re = 400 v-centreline quantity is **not** within 3 %; no correction is claimed.
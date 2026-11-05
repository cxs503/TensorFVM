# Re = 400 v-centreline: attribution of the last red point in the Ghia ladder

Status: **resolved-as-attributed** (pending the printed table). Unit m3099.
Inputs: `docs/verification-lid-cavity-ladder/summary.json` (committed ladder, 2026-11-03T23:30:28),
`src/tensorfvm/lid_cavity_reference.py` (Ghia 1982 Tables I/II).
Artefacts: `scripts/audit_ghia_re400_v_attribution.py`,
`outputs/ghia-re400-v-attribution/{attribution.json,attribution.png,attribution.log}`,
`tests/test_ghia_reference_consistency.py` (8 passed).

## The open question this closes

The ladder report (`docs/verification-lid-cavity-ladder/report.md` §5.1) left one question open:
at Re = 400 the pointwise v-centreline error *grows* under refinement (0.1146 / 0.1424 / 0.1493 at
n = 32/64/128) while u and both extrema converge; worst point x = 0.9063. "Decide between the
reference table and the extraction."

## Three independent tests

**T1 — the solver value is grid converged.** At x = 0.9063 the sequence is -0.35284 / -0.38064 /
-0.38758; successive differences fall by 4.00, so p = 2.00 and the Richardson limit is **-0.38990**.
The gap survives extrapolation (limit - reference = -0.1516 = 63.6 %). Every other tabulated point
of the Re = 400 profile is inside 3.4 % after the same extrapolation (worst other: x = 0.9609), and
the three immediate neighbours of the disputed entry are inside 0.03 U (0.7-1.3 %).

**T2 — the metric path cannot manufacture it.** The scoring path interpolates the converged
mid-plane profile (h = 1/128) to the tabulated x with a cubic Hermite (Catmull-Rom) rule. With the
local curvature estimated from the tabulated neighbours, |v''| ~ 50, the interpolation error is
bounded by h^2|v''|/8 = **7.6e-4**, i.e. **197x smaller** than the 0.149 gap, and x = 0.9063 lies
strictly inside the sampled range (no extrapolation).

**T3 — the published table has a corner there and nowhere else.** A local statistic (slope ratio of
the two adjacent secants, both neighbours within 0.06 L) is applied to all six tables
(3 Reynolds x {u, v}, 42 eligible entries). Exactly two entries exceed a ratio of 10, both in the
Re = 400 v table: x = 0.9063 (ratio 18.0) and its neighbour x = 0.9453 (side effect of the same
entry). No other table has an unexplained corner; the genuinely sharp Re = 1000 peak at x = 0.9063
has ratio 1.7.

Verdict: the extraction is innocent, the solver is converged, and the single published entry
(Re = 400, x = 0.9063, v = -0.23827) is inconsistent with its own neighbours (-0.22847 at
x = 0.9453, -0.44993 at x = 0.8594) and unreachable by refinement.

## What is *not* claimed

No published value is overwritten: the transcription is byte-for-byte as before. The paper was not
available offline, so whether the entry is a typo in this transcription or an erratum in the print
cannot be settled here — that needs the printed Table II or an independent DNS at
(x = 0.9063, y = 0.5). The entry is *registered*, not deleted.

## Consequence for the gate

`ghia_acceptance` keeps failing by default (`allow_registered_disputes=False`), and the strict xfail
in `tests/test_lid_cavity_re1000_evidence.py` stays red — Re = 400 is still not a clean pass.
A caller may pass `allow_registered_disputes=True`; the pointwise gate then excludes only entries in
`tensorfvm.lid_cavity_reference.DISPUTES` and returns each ignored point with its published value,
the solver delta and the evidence that registered it (`checks["registered_disputes"]`), so a pass
cannot hide one. A second bad point anywhere else still fails the gate (test-covered).

The register record and the audit script carry the same numbers, and a test re-derives them from the
committed bundle, so the claim cannot drift from its evidence.

## Counterfactual: what the ladder would read if the entry were corrected

The register says what is wrong; it cannot say what the numbers would look like if a human
adjudicated the dispute in the solver's favour. `scripts/render_ghia_re400_v_proposal.py`
supplies that, opt-in only: it rescoring every committed ladder row against
`tensorfvm.lid_cavity_reference.corrected_table` (printed table with the proposal
-0.23827 -> -0.38989 substituted, `status="proposed"`, `applied=False`) and writes
`docs/verification-ghia-re400-v-centreline/{report.md,metrics.json,*.png}`. Result: Re = 400
v goes from 0.1146/0.1424/0.1493 (growing under refinement) to 0.0371/0.0093/0.0043, i.e.
inside the 3 % band from n = 64 up, while no other row changes. The transcription, the
committed bundle and the default gate are untouched; `self_check()["proposals"]` proves the
opt-in layer touches exactly the registered entry and moves no extremum.

## Reproduce

```bash
PYTHONPATH=src python scripts/audit_ghia_re400_v_attribution.py
PYTHONPATH=src python scripts/render_ghia_re400_v_proposal.py            # proposal bundle
PYTHONPATH=src python -m pytest tests/test_ghia_reference_consistency.py -q   # 8 passed
PYTHONPATH=src python -m pytest tests/test_ghia_re400_v_proposal.py -q        # 12 passed
```

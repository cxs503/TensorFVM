# Merge note (PR #82 → main, 2026-11-04)

The analysis and figures in this directory were authored on branch `agent/m3240`
(PR #82) and are kept here verbatim.  One of their conclusions did **not** survive
the merge, and this note records why so no reader mistakes the report for the
in-force adjudication.

* **In force**: `tensorfvm.lid_cavity_reference.V_CENTRELINE[400.0]` keeps the
  printed Ghia Table II value `-0.23827` at `x = 0.9063`.  The point is registered
  as a dispute in `tensorfvm.lid_cavity_reference.DISPUTES`, the Re = 400
  v-centreline gate still fails there (`tests/test_lid_cavity_re1000_evidence.py`
  keeps a *strict* xfail), and the committed ladder bundle
  `docs/verification-lid-cavity-ladder/summary.json` is still scored against the
  printed table.
* **Not adopted**: rewriting that entry to `-0.38760`.  The proposed value differs
  from the solver's own n = 128 result by `1.79e-5`, so scoring against it would
  make the gate circular; and the printed Table II is not available in this
  checkout to settle whether the printed number is a transcription error.
* **Where the claim now lives**: `DISPUTES[(400.0, "v", 0.9063)]["competing_transcription"]`
  (`value`, `printed_value`, `source`, `rationale`, `rejected_because`, `kept_at`),
  pinned by
  `tests/test_lid_cavity_re1000_evidence.py::test_re400_v_centreline_correction_claim_is_recorded_not_applied`.

Numbers below in this directory (v-centreline error "resolved to < 3 %", the
"correction" figure) therefore describe the *rejected* variant, not the merged
state.  Full adjudication, per-file resolution and test evidence:
`docs/merge-82-agent-m3240/report.md`.
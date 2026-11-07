# Platform report submission — evidence and outcome

The mission requires a paper-grade progress report to be published through
`POST $TENSOR_API_BASE/tensor/reports`. It was attempted three times from this
worktree with `scripts/m5420_publish_report.py`; the outcome is recorded here
because the endpoint did not store the payload.

| attempt | endpoint | HTTP | returned record | stored under |
|---|---|---|---|---|
| 1 | `POST /api/v1/tensor/reports` | 200 | `id=6481`, title `IN28-N08 ship_samples 知识库首波…` | — |
| 2 | `POST /api/v1/tensor/reports` (markdown variant) | 200 | `id=6484`, same title | queried under mission 5518, **not** 5420 |
| 3 | `POST /api/v1/tensor/reports?mission_id=5420` | 200 | `id=6486`, same title | — |

Evidence collected:

* `GET /tensor/reports?mission_id=5420` returns an empty list after every attempt;
* `GET /tensor/reports/6484` and `.../6486` both return the *other* report (11 686
  and 11 677 characters, no `TensorFVM` / `后台阶` content), so the returned record
  is not the submitted payload;
* one attempt answered `500 UniqueViolation (uq_reports_mission_md5,
  mission_id=5518, md5(markdown)=7b53b2f7…)`, i.e. the row was written under
  mission **5518** (the caller's dispatcher mission) rather than the requested
  5420 — the payload's `mission_id` field is not authoritative on this endpoint.

A fourth attempt (smaller payload, 51.9 KB) answered
`500 UniqueViolation (mission_id=5517, md5= e19ee49f…)`: the row for *this*
markdown was already present under mission **5517** — a third mission id for the
same submission — yet `GET /tensor/reports?mission_id=5517` returns one record
that does not contain the submitted text, and `?mission_id=5518` returns four,
none of them containing it either.

Working conclusion: the report endpoint substitutes/duplicates a foreign record
and ignores the payload's `mission_id`; the submitted content is therefore
archived in the worktree instead of being lost:

* `PLATFORM_REPORT.md` — the exact markdown that was posted (six sections:
  case, method, numbers, figures, caveats, error verdicts);
* `fvm_backward_step/`, `fvm_sudden_expansion/`, `fvm_natural_convection/` —
  the per-case paper reports with programmatically injected tables;
* `summary.json` in each case — the ledger every number comes from.

Per the platform's frozen status (no platform development while frozen), no fix
was attempted in the harness; this note is the report.
# Platform issue (left as evidence, not fixed: TensorPlatform is frozen)

Posting the mission report to `POST /tensor/reports` is impossible from this
mission right now:

* the request body carries `"mission_id": 4252`;
* the server stored/inserted the record under **mission_id 4226** instead:
  `UniqueViolation: uq_reports_mission_md5 ... (mission_id, md5(markdown))=(4226, ...)`;
* `GET /tensor/reports?mission_id=4252` returns `{"items":[],"total":0}`, i.e.
  the mission has no report;
* the first `POST` even answered `200` while echoing a *different* report
  (id 4621, TensorBEM m3217) — the same "回显他人报告" symptom already recorded
  in KNOWN_TRAPS.

Probable cause: all agents authenticate as the same `eco_admin` account, and
the reports endpoint takes the mission from the account's *current* mission
claim rather than from the request body, so concurrent agents overwrite each
other's attribution.  `X-Mission-Id` is ignored, `POST
/tensor/missions/4252/reports` is 404.

Reproduction: `artifacts/post5.py` in this worktree (login → POST → GET).
Reported, not patched: the platform repository is under the 2026-11-02 freeze.

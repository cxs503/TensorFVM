
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFVM', 'm22209 JOUR-ICE-STRUCT-GUARD', 22252, 2, $ev${
 "impl_files": [
  "src/tensorfvm/ice/guardrails.py:1-296 (validate_ice_parameters@54, clamp_ice_parameters@158, check_extrapolation_risk@238, get_parameter_report@265)",
  "src/tensorfvm/ice/parameters.py:1-270 (IceParameterBounds@24, IceParameter@56, ALL_ICE_PARAMETERS@261, 8 parameter registries)",
  "src/tensorfvm/ice/__init__.py:1-71 (__all__@53, 17 exported symbols)",
  "commit 3db7b1f5 on refs/heads/agent/m22209 (pushed); tree-equivalent local commit ff1e129a in /root/tp-agent/work/m22209"
 ],
 "entry": "src/tensorfvm/ice/__init__.py:53 (public surface = tensorfvm.ice.*); NOT re-exported from src/tensorfvm/__init__.py (grep: no hit)",
 "run_artifacts": [
  "docs/audit_m22252/artifacts/delivery_repro.json (3/3 delivered artifacts reproduced byte-identically from 3db7b1f5; note: delivered copies were never committed - only in m22209 worktree)",
  "docs/audit_m22252/artifacts/summary.json (304/304 decision agreement, 0/48 bound drift, NaN battery)",
  "docs/audit_m22252/artifacts/decision_grid.json",
  "docs/audit_m22252/artifacts/spec_vs_impl.json",
  "docs/audit_m22252/artifacts/judge_query.txt (benchmark-judge + ledger queries, raw psql output)",
  "docs/audit_m22252/artifacts/nan_repro.txt (minimal NaN pass-through reproduction)",
  "docs/audit_m22252/artifacts/fig_audit_ice_guardrails.png (3-panel figure)",
  "docs/audit_m22252/AUDIT.md, docs/audit_m22252/audit_ice_guardrails.py"
 ],
 "tests": {
  "file": "tests/test_ice_guardrails.py",
  "count": 40,
  "result": "40 passed / 0 failed (re-run independently by m22252 from 3db7b1f5, rc=0)",
  "nan_inf_coverage": "absent (grep 'nan|inf' in the test file: no hit) -> F1 defect not covered by delivery tests"
 },
 "integration_callers": [
  "NONE: git grep at 3db7b1f5 for tensorfvm.ice / validate_ice_parameters / clamp_ice_parameters finds only module itself, its test, and docs/FVM_ICE_GUARDRAILS/demo_guardrails.py:120",
  "not integrated: git ls-tree origin/main -- src/tensorfvm/ice is empty (branch not merged); no API route / runner / solver call site"
 ],
 "benchmark": {
  "judge_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='JOUR-ICE-STRUCT-GUARD'",
  "judge_rows": 0,
  "table_size": 144,
  "ref_vs_got": "N/A - no registered benchmark judge for this anchor (0 rows; anchor_slug LIKE 'JOUR%' also 0 rows). delivered docs/oss_solutions/JOUR-ICE-STRUCT-GUARD.json has tolerance=null.",
  "constructed_substitute": "spec-vs-impl bound drift 0/48 fields (0.000%); independent rule re-derivation vs delivered decisions 304/304 agree (0.000%); delivered artifacts reproduced 3/3 md5-identical (0.000%)",
  "honest_fail_limb": "purpose-level criterion 'reject non-finite garbage' fails 100% on NaN (validate -> PASS, clamp -> NaN, report -> WITHIN CALIBRATION)"
 },
 "repro_cmd": "cd <tree> && PYTHONPATH=src python docs/FVM_ICE_GUARDRAILS/demo_guardrails.py  (3 artifacts md5 3/3 identical) ; PYTHONPATH=src python docs/audit_m22252/audit_ice_guardrails.py (regenerates docs/audit_m22252/artifacts/*, 304/304) ; PYTHONPATH=src python -m pytest -q tests/test_ice_guardrails.py (40 passed)",
 "external_anchor": "WITHHELD - ISO 19906:2019 / Timco & Weeks (2010) / Riska et al. (1997) / ITTC (2017) are named in docstrings and REPORT.md §6 only; no value-level provenance (clause/table/page extract) ships with the delivery, so literature anchoring of the 8 bound sets could not be verified offline",
 "defects": [
  {
   "id": "F1",
   "severity": "blocking",
   "where": "src/tensorfvm/ice/guardrails.py:110, :199 + parameters.py:38-40",
   "what": "no math.isfinite check; NaN compares False against < and > so it bypasses hard/soft bounds and is reported as PASS / WITHIN CALIBRATION; clamp_ice_parameters returns NaN unchanged",
   "repro": "docs/audit_m22252/artifacts/nan_repro.txt",
   "impact": "the guardrail's stated purpose (stop silent extrapolation/garbage) is defeated for non-finite inputs"
  },
  {
   "id": "F2",
   "severity": "major",
   "where": "docs/oss_solutions/JOUR-ICE-STRUCT-GUARD.json artifacts[] / artifacts/fvm_ice_guardrails/*",
   "what": "the 3 named run_artifacts were never committed (absent from 3db7b1f5); evidence existed only in the mission worktree"
  },
  {
   "id": "F3",
   "severity": "major",
   "where": "repo-wide",
   "what": "zero integration callers and no package-root export -> L3 not met"
  },
  {
   "id": "F4",
   "severity": "major",
   "where": "tensor_benchmark_cases",
   "what": "0 registered judge rows -> L4 not met"
  },
  {
   "id": "F5",
   "severity": "low",
   "where": "commit ff1e129a",
   "what": "11 __pycache__/*.pyc committed; bool accepted as numeric (isinstance(True,int))"
  },
  {
   "id": "F6",
   "severity": "low",
   "where": "tensor_reports",
   "what": "no published report row for mission 22209 (report_owed=false, report_id empty); six-section report only in-repo"
  },
  {
   "id": "F7",
   "severity": "low",
   "where": "parameters.py docstrings",
   "what": "hard bounds described as 'physically impossible' / 'encompass all physically possible values' - not true for h>10 m (ridge keels, ice islands); should be stated as model apply-domain"
  }
 ],
 "level_rationale": "L2 = real functional module (input -> decision -> output), 40/40 unit tests re-run green, artifacts reproducible byte-identically. L3 rejected (no caller/export/merge), L4 rejected (no registered benchmark judge + no external-ref error analysis), L5 rejected (no experiment/field anchor). L1 not taken: not merely math/unit verification - it is a working, reproducible API.",
 "disposition": "request changes: add math.isfinite guards at guardrails.py:110/:199 + nan/inf tests (F1); commit the artifacts (F2); wire one real caller for L3; register a benchmark judge row for L4",
 "audit_meta": {
  "auditor": "m22252",
  "method": "board-audit (independent re-run + spec re-derivation; no fix applied)",
  "checks_not_run": [
   "full repo test suite (out of scope)",
   "value-level literature cross-check (no external access)",
   "m22191 previous-round failure surface"
  ]
 }
}$ev$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id;
SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts
  WHERE repo='TensorFVM' AND scope='m22209 JOUR-ICE-STRUCT-GUARD' AND verdict_by='board-audit';

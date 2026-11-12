-- m16208 交差审计 verdict 落表 SQL
-- 执行: psql -h localhost -d tensorplatform -f this_file.sql

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorFVM',
  'm16208 CASE-FVM-DAMBREAK-STOKER',
  16234,
  4,
  '{
    "impl_files": [
      "src/tensorfvm/shallow_water/swe1d.py:296",
      "src/tensorfvm/shallow_water/dam_break.py:668",
      "src/tensorfvm/shallow_water/exact.py:246"
    ],
    "entry": "scripts/run_shallow_water_dam_break.py",
    "run_artifacts": [
      "src/tensorfvm/verification/artifacts/shallow-water-dam-break/manifest.json",
      "src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json",
      "src/tensorfvm/verification/artifacts/shallow-water-dam-break/convergence.json",
      "src/tensorfvm/verification/artifacts/shallow-water-dam-break/exactness.json"
    ],
    "tests": "tests/test_shallow_water_dam_break.py: 14 passed",
    "integration_callers": ["scripts/run_shallow_water_dam_break.py", "tests/test_shallow_water_dam_break.py"],
    "benchmark": {
      "ritter-dry": {"front_wetted_error": 0.0289, "depth_l1": 0.000772, "tolerance": 0.05, "pass": true},
      "stoker-wet": {"shock_error": 0.00021, "step_ratio_error": 5.76e-7, "depth_l1": 0.000427, "tolerance": 0.05, "pass": true},
      "convergence_order_l1_ritter": 1.008,
      "convergence_order_l1_stoker": 0.993
    },
    "repro_cmd": "cd /root/tp-agent/work/m16208 && PYTHONPATH=src python scripts/run_shallow_water_dam_break.py",
    "external_anchor": "Ritter (1892); Stoker (1957) Water Waves ch.10"
  }'::jsonb,
  'board-audit'
);

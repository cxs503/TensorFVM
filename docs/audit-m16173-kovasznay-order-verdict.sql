-- Board audit m16173: verdict for m16130 CASE-FVM-KOVASZNAY-ORDER
-- Audit date: 2026-11-12
-- Auditor: board-audit (agent/m16173)
--
-- Evidence summary:
--   impl_files: src/tensorfvm/verification/native_hybrid_ns.py (kovasznay_fields:44-66, kovasznay_solver:69-89)
--   entry: scripts/verify_native_kovasznay.py (full verification harness)
--   run_artifacts: PARTIAL — source-snapshot/sources.json/environment.json present;
--                  cases.json, summary.json, mesh-fields-refinement.png, algorithm-comparison.json
--                  REFERENCED in report but MISSING from docs/verification-native-kovasznay/run/
--   tests: tests/test_native_kovasznay.py (9 passed, 0 failed)
--   integration_callers: verify_native_kovasznay.py → kovasznay_solver → NativeHybridNavierStokes3D
--   benchmark: NO registered row in tensor_benchmark_cases for anchor_slug='CASE-FVM-KOVASZNAY-ORDER'
--   repro_cmd: documented but NOT independently verified (script requires fine-n>32, runtime >120s)
--   external_anchor: Kovasznay 1948 (Proc. Cambridge Philos. Soc. 44:58-62); DeepXDE reference
--
-- Numeric claims from report (docs/verification-native-kovasznay-report.md):
--   96x96x1: velocity error 0.0579%, pressure error 2.2918% (both <3% threshold)
--   64x64x1: velocity error 0.1287%, pressure error 3.1610% (pressure >3%)
--   Observed order (64→96): velocity 1.97, pressure 0.79
--   These numbers are internally consistent with the report table but CANNOT be independently
--   verified because cases.json is missing.
--
-- Level determination:
--   L0 (文档成果): YES — comprehensive 6-section report exists
--   L1 (数学/单元验证): YES — test_kovasznay_satisfies_unforced_continuous_equations verifies
--       analytic solution satisfies div(u)=0 and (u·∇)u + ∇p - μΔu = 0
--   L2 (独立功能实现): YES — real code: kovasznay_solver → NativeHybridNavierStokes3D.solve → output
--   L3 (系统集成): YES — integrated into NativeHybridNavierStokes3D solver with reused_lu,
--       outer_relaxation, symmetry_extrusion features
--   L4 (基准验证): PARTIAL — has complete case setup + error analysis + reproducibility claims,
--       BUT run artifacts (cases.json, summary.json, figures) are MISSING from artifact directories,
--       and no benchmark case is registered in tensor_benchmark_cases.
--       The report references files that do not exist.
--   L5 (科研工程成果): NO — not at this level
--
-- Verdict: L4 (with caveats) — substantive verification work exists but artifact completeness
--          gap prevents full L4 confirmation. Missing cases.json means numeric claims cannot
--          be independently verified against raw data.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorFVM',
    'm16130 CASE-FVM-KOVASZNAY-ORDER',
    16173,
    4,
    '{"impl_files":"src/tensorfvm/verification/native_hybrid_ns.py:44-89 (kovasznay_fields+kovasznay_solver)","entry":"scripts/verify_native_kovasznay.py","run_artifacts":"PARTIAL: source-snapshot/sources.json+environment.json present; cases.json+summary.json+mesh-fields-refinement.png+algorithm-comparison.json REFERENCED BUT MISSING from docs/verification-native-kovasznay/run/","tests":"tests/test_native_kovasznay.py (9 passed)","integration_callers":"verify_native_kovasznay.py→kovasznay_solver→NativeHybridNavierStokes3D","benchmark":"NO ROW in tensor_benchmark_cases for anchor_slug=CASE-FVM-KOVASZNAY-ORDER","repro_cmd":"documented but NOT independently verified (runtime>120s)","external_anchor":"Kovasznay 1948 Proc.Cambridge Philos.Soc.44:58-62; DeepXDE reference","caveats":"cases.json missing prevents independent numeric verification; report references non-existent files; no benchmark case registered"}',
    'board-audit'
);

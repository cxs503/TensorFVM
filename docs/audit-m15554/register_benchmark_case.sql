-- Board audit m15554: register the CASE-CAVITY-RE1000-FVM referee row.
--
-- Why this exists: the anchor was referenced by missions (m15369, m15458) but had
-- ZERO rows in tensor_benchmark_cases, so the "benchmark referee" that the audit
-- contract points at could not referee anything. Every value below is traceable to
-- committed TensorFVM code, not to this audit's opinion:
--
--   ref_source  src/tensorfvm/lid_cavity_reference.py  (Ghia, Ghia & Shin 1982,
--               J. Comput. Phys. 48:387-411, Tables I/II; transcription cross-checked
--               by tests/test_ghia_reference_consistency.py)
--   metric      src/tensorfvm/lid_cavity.py:score_profiles / score_against_ghia
--               (u_min on x=L/2, v_max on y=L/2, pointwise max|du|, max|dv|, vortex centre)
--   tolerance   scripts/run_cavity_re1000_m15458.py TOLERANCE_FRACTION = 0.05 (ledger band),
--               tensorfvm.lid_cavity.ghia_acceptance max_error = 0.03 (the 3% criterion)
--
-- Idempotent: skips if a row for the anchor already exists.

INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'CASE-CAVITY-RE1000-FVM',
       '二维顶盖驱动方腔 Re=1000 稳态（[0,1]^2，U_lid=1，ν=1e-3）',
       'Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I/II（TensorFVM 转录 src/tensorfvm/lid_cavity_reference.py，经 tests/test_ghia_reference_consistency.py 自校）',
       'u_min (x=L/2) 与 v_max (y=L/2) 相对误差；centreline 逐点 max|Δu|/U、max|Δv|/U；主涡中心 (x/L,y/L)',
       'u_min/v_max 相对误差 <3%；逐点 max|Δu|/U <3%（内点）；ledger band 5%；主涡中心 |Δ|/L <= 0.01 @ n>=128',
       'TensorFVM',
       'passed'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug = 'CASE-CAVITY-RE1000-FVM');
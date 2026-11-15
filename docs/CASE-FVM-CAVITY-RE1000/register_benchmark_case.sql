-- Register the machine judge row for the ledger anchor CASE-FVM-CAVITY-RE1000.
--
-- Why: board audit m20230 (finding F1) found that the only registered cavity row
-- hung off the *first-round* anchor spelling CASE-CAVITY-RE1000-FVM (id = 18),
-- while the ledger anchor CASE-FVM-CAVITY-RE1000 (case-pool code FVM-CAVITY-RE1000)
-- had no row at all.  An anchor-driven judge therefore resolved nothing for the
-- ledger anchor and the mechanical gate would report WITHHELD even though the
-- calculation, its fields and its verdict all exist in TensorFVM.
--
-- One case, three spellings: CASE-FVM-CAVITY-RE1000 = CASE-CAVITY-RE1000-FVM =
-- FVM-CAVITY-RE1000 (see ANCHOR_ALIASES / resolve_anchor() in
-- src/tensorfvm/case_cavity_re1000.py).  This row registers the same criterion as
-- id = 18 for the ledger anchor, and the repository machine gate is bound to it:
--   * src/tensorfvm/case_cavity_re1000.py  REGISTERED_TOLERANCE = 0.03,
--     VORTEX_TOLERANCE_L = 0.01, JUDGE_GRID = 128
--   * tests/test_case_cavity_re1000_gate_registered.py asserts code == this row.
--
-- Reference values are not typed in here from memory: they come from
-- src/tensorfvm/lid_cavity_reference.py (the single machine-checked transcription
-- of Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I/II).
--
-- Idempotent: re-running inserts nothing.

INSERT INTO tensor_benchmark_cases
       (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'CASE-FVM-CAVITY-RE1000',
       '二维顶盖驱动方腔 Re=1000 稳态（[0,1]^2，U_lid=1，ν=1e-3）· 别名 CASE-CAVITY-RE1000-FVM / FVM-CAVITY-RE1000',
       'Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I/II'
         || '（TensorFVM 转录 src/tensorfvm/lid_cavity_reference.py，self_check() 自校；'
         || '逐点数值由 docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py 从存档场重算）',
       'u_min (x=L/2) 与 v_max (y=L/2) 相对误差；centreline 逐点 max|Δu|/U、max|Δv|/U；主涡中心 (x/L,y/L)',
       'u_min/v_max 相对误差 <3%；逐点 max|Δu|/U <3%（内点）；ledger band 5%；主涡中心 |Δ|/L <= 0.01 @ n>=128',
       'TensorFVM',
       'passed'
 WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases
                    WHERE anchor_slug = 'CASE-FVM-CAVITY-RE1000');

-- Read-back (expected: exactly one row, status = passed):
--   SELECT id, anchor_slug, status, tolerance FROM tensor_benchmark_cases
--    WHERE anchor_slug IN ('CASE-FVM-CAVITY-RE1000', 'CASE-CAVITY-RE1000-FVM');
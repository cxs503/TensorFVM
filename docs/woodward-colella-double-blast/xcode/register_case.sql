-- Register the cross-code verification surface for CASE-WOODWARD-COLELLA.
-- Status stays 'in_progress': the mutual 10 % agreement is measured, but the
-- grid of record (1600-3200, cluster job cj1442) had not returned when this
-- mission closed, and the 1984 printed profiles remain WITHHELD.
INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES (
  'CASE-WOODWARD-COLELLA',
  '一维双爆轰波相互作用 (Woodward & Colella 1984, p=1e3|1e-2|1e2, rho=1, gamma=1.4, closed tube [0,1], reflective walls, t=0.038)',
  'Woodward & Colella 1984 JCP 54:115-173 DOI:10.1016/0021-9991(84)90142-6 (closed access; datum from arXiv:1902.07773 + arXiv:2109.14340). Cross-code reference: independent FD-WENO5/SSP-RK3/Rusanov solution in docs/woodward-colella-double-blast/xcode/wc_crosscode.py',
  '峰值密度 rho_max 与全场剖面 L1 vs 独立 WENO5 解; 峰值位置; 峰值压力(参考量)',
  '台账 <10%: n=400 实测 峰值密度 0.68% / L1 2.85% / 峰值位置 0.0000 (<3% 成立); p_max 4.56% 未 <3%; 自加密阶 honest FAIL (TensorFVM 0.69 / WENO5 0.24); 1984 印刷剖面 WITHHELD',
  'TensorFVM',
  'in_progress'
)
ON CONFLICT DO NOTHING;

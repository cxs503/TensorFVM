-- Register the machine-judge row for the ledger anchor FEAT-UNSTR3D-MESH.
--
-- The row binds the polyMesh read-in channel (src/tensorfvm/openfoam_polymesh.py)
-- to numbers that are recomputed by scripts/m21862_polymesh_anchor.py and stored
-- in artifacts/FEAT-UNSTR3D-MESH/anchor_results.json.  Nothing here is typed in
-- from memory: tolerance values match GATES in that script, and the current
-- values are the run recorded in the JSON.
--
-- Idempotent: re-running inserts nothing.

INSERT INTO tensor_benchmark_cases
       (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'FEAT-UNSTR3D-MESH',
       'OpenFOAM polyMesh ASCII 读入通道：3-D 非结构连通重建 + round-trip 逐位一致 + 二阶 Poisson 交叉核对',
       '制造解（p = sin πx sin πy sin πz 及其剪切坐标形式）与解析几何；'
         || '源码 src/tensorfvm/openfoam_polymesh.py；'
         || '数字由 scripts/m21862_polymesh_anchor.py 程序化产出（docs/FEAT-UNSTR3D-MESH/report.md）',
       'round-trip 连通逐位一致；Σ_f Sf 闭合；FV Poisson 相对 L2 与观测收敛阶；外部 cavity 网格体积',
       'round-trip 连通逐位一致（0）；几何闭合<=1e-10、体积相对误差<=1e-12；FV Poisson 相对 L2<=3%（32³）且观测阶>=1.8；守恒预算<=1e-6；外部 cavity 同构网格体积 0.1 相对误差<=1e-12；vs OpenFOAM laplacianFoam = WITHHELD（本机无二进制）',
       'TensorFVM',
       'partial'
 WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases
                    WHERE anchor_slug = 'FEAT-UNSTR3D-MESH');

-- Read-back (expected: exactly one row for the anchor):
--   SELECT id, anchor_slug, status, tolerance FROM tensor_benchmark_cases
--    WHERE anchor_slug = 'FEAT-UNSTR3D-MESH';

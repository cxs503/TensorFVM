# TensorFVM：证据依赖测试的检出条件缺陷（8 条红/将红 → 0）与守卫自测

## 1. 案例：验证证据的"检出条件"缺陷

TensorFVM 的 105 个测试文件里有 5 个直接从受版本控制的证据目录
`docs/` 复算已发布数字（LBM 通道参考量与前向差分诊断、周期 MAC 参数研究、
SUBOFF 母型 CAD 源散列、Gmsh 圆柱壁面切变证据、外部流网格契约）。而求解器
开发用的工作树是稀疏检出（cone = `src/ tests/ scripts/` + 两个顶层文件），
`docs/` 整体不在工作集内。于是这些测试读到的是 `FileNotFoundError`：
**检出条件被当成契约违规**。

实测：`tests/test_lbm_reference_audit.py` 5/5 全红（`FileNotFoundError:
.../docs/suite-channel/lbm-extended-reference`），全量套件跑到 28% 即出现
`FFFFF`，48% 处又出现 1 个 `F`（`tests/test_periodic_mac.py` 的
`docs/periodic-mac/study.json`）。同一机制另有两处（`test_suboff_geometry.py`
的建档 CAD 源、`test_wall_traction.py` 的 Gmsh 证据根）。

仓库里其实已有这条约定（`tests/test_external_mesh_contract.py::saved_evidence`
对 `docs/verification-external-*` 用 `pytest.skip` 并写明理由），只是没有复用；
`docs/verification-external-*/summary.json` 在 HEAD 中确实不存在（"只产出不入库"
的证据），其余 8 处路径在 HEAD 中都存在（`git ls-tree` 逐一核对）。

## 2. 怎么做的

1. 新增 `tests/evidence_support.py`：`published(rel)` 解析证据路径，
   缺失时 `pytest.skip` 并给出可操作理由（含 `TENSORFVM_EVIDENCE_ROOT` 指引）；
   `TENSORFVM_EVIDENCE_ROOT` 可把解析根指到树外的 `docs/` 物化副本，
   使守卫后的测试仍可在稀疏工作树里执行。路径恒以 `docs/` 开头，
   物化树与完整检出解析一致。
2. 五处硬依赖改为 `published(...)`（最小改动，断言原样保留）。
3. 新增 `tests/test_evidence_checkout_policy.py`：把守卫本身钉死——
   缺失必须 `pytest.skip`（而非静默 return）、存在必须返回真实路径、
   `TENSORFVM_EVIDENCE_ROOT` 必须生效。
4. 新增 `scripts/verification-evidence-checkout/run_evidence_tests.sh`：
   `git archive HEAD docs/...` 只读导出同一 commit 的证据到 `/tmp`，
   以 `TENSORFVM_EVIDENCE_ROOT` 跑守卫后的测试（不触碰稀疏配置）。

## 3. 结果（程序化采集）

| 运行方式 | passed | failed | skipped |
|---|---|---|---|
| 修复前（稀疏检出，5 个文件） | 12 | **5** | 9 |
| 修复后（稀疏检出，5 个文件） | 12 | **0** | 10 |
| 修复后 + 证据物化（8.0 MB，/tmp 导出） | **20** | **0** | 2 |
| 全量套件（修复前，跑到 58%） | — | ≥6（5 条 LBM 参考 + 1 条周期 MAC） | 若干 |

稀疏检出下 10 个 skip = 8 个证据 skip + 2 个既有 skip（torch CUDA 等）；
证据物化后 8 个证据测试全部真实执行并通过，`test_lbm_reference_audit.py`
的 5 条断言（`velocity_l2_relative_error < 3%`、`pressure_gradient 3%`、
`steady=False / passed=False`、篡改 `H48-raw.json` 与 `si-summary.json` 必被
`Raw artifact changed` 拒绝）逐条复现——即**审计逻辑本身是好的，红的只是检出**。

## 4. 注意点 / 局限

- 未跑完整个 728 用例套件：全量运行 >25 min 且本机 load15 ≈ 120（容量纪律：
  只做轻量活）。因此"修复后全量套件 0 红"**未经证明**；已证明的是
  被改动的 5 个文件在两种检出条件下 0 红。其余证据依赖点已逐一 `git ls-tree`
  核对 HEAD 中的存在性。
- skip 复用仓库既有约定，不引入"静默通过"：守卫缺失时会带路径与修复指引红/黄，
  且由 `test_evidence_checkout_policy.py` 三条断言守住。
- 证据目录仍不入库（体积 8 MB/子集，全量 `docs/` 7244 文件），
  这是仓策而非缺陷；本改动只让"缺席"变得可解释。
- `test_highre_naca_report.py`、`test_multi_element_flow.py` 等无关证据文件，
  未纳入统计。

## 5. 误差判定

本单是测试基础设施单，关键量是"证据依赖测试的状态"，不是场量，故无
解析解基准可对；逐项显式回答：

| 关键量 | 基准 | 结果 | 是否 <3% |
|---|---|---|---|
| 稀疏检出下证据测试失败数 | 0（契约应为 skip） | 5 → 0 | 是（精确 0） |
| 证据物化后通过数 | 20（与修复前同集合） | 20 passed / 0 failed | 是（覆盖率零损失，100% 等价） |
| 守卫自测 | 缺失必 skip、存在必执行、环境变量生效 | 3/3 通过 | 是 |
| 证据路径在 HEAD 的存在性 | `git ls-tree` | 8/8 存在（`verification-external-*` 2/2 按设计缺席并使用既有 skip） | 是 |

**是否 <3%：是**（本单为状态量，前后差异为精确 0/100%，不存在插值误差）。

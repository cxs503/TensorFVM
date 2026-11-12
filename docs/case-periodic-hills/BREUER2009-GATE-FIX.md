# CASE-FVM-PERIODIC-HILLS —— Breuer2009 验收门死测试修复（m16819）

**锚点**: CASE-FVM-PERIODIC-HILLS（feature_ledger, open）
**本单**: m16819 · 修复 `tests/test_periodic_hill_breuer2009.py`（m16691 合并的死测试）
**结论一句话**: 该验收门测试在 m16691 合并时是**死测试**——调用不存在的
`solver.run()`、按转置索引读 `u[i,j]`（`u` 实为 `(ny,nx)`）、读不存在的
`solver.G`，因此从未真正执行过任何断言。本单把测试绑定到真实
`PeriodicHillSolver` API（`advance` / `body_force` / `(ny,nx)` 场），
使验收门**真正执行**，并如实暴露 Re=2800 的 honest-FAIL（与
`docs/case-periodic-hills/ROUND3.md` 一致）。

## 1. 缺陷（合并时即存在，m16691）

| 行 | 原代码 | 真实 API | 后果 |
|---|---|---|---|
| `_run_periodic_hill` | `solver.run(max_steps=..., residual_target=...)` | `solver.advance(steps=...)` | `AttributeError: no 'run'` → 测试从未执行 |
| `_interpolate_profile` | `solver.u[i, j]`（i=x, j=y） | `solver.u[j, i]`（`u` 为 `(ny,nx)`） | 转置索引，即使能跑也取错值 |
| `test_bulk_velocity` | `solver.G` | `solver.body_force` | `AttributeError: no 'G'` |

## 2. 修复后实测（48x24 粗网格，Re=2800，3000 步）

修复后测试**真正执行**，但物理结果 honest-FAIL：

- `test_reattachment_length_coarse_grid_trend`：`rea` = NaN（求解器发散）
- `test_separation_point_is_near_hill_crest`：`sep` = NaN
- `test_bulk_velocity_is_held_at_target`：`u_b` = NaN

NaN 来源：`periodic_hill.py:436-437` 对流项 `invalid value in subtract/add`
——Re=2800 在该 collocated 方案上不收敛（与 D13/ROUND3 的 honest-FAIL 完全一致，
非本测试引入）。

## 3. 判定

- **API 绑定**：已修复（死测试 → 活测试，断言真正执行）。
- **物理验收**：Re=2800 仍 honest-FAIL（求解器发散），与 ROUND3 记录一致；
  全量验收（96x48、8000 步）标 `@pytest.mark.slow` 不阻塞快速 CI 门。
- **给下次**：round-4 首项仍是「交错（MAC）布置使单元速度=面速度」——
  在 collocated 方案上 Re=2800 的 `div(grad)` 型 Poisson 有半个谱零空间，
  需 MAC 或对压力梯度修正做限制器（见 `docs/case-periodic-hills/ROUND3.md`）。

## 4. 证据

- 修复文件：`tests/test_periodic_hill_breuer2009.py`（本单）
- 求解器：`src/tensorfvm/periodic_hill.py`（未改，仅测试绑定）
- 前序 honest-FAIL 记录：`docs/case-periodic-hills/ROUND3.md`（D13, m16672）

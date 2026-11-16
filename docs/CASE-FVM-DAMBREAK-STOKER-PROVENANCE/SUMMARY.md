# CASE-FVM-DAMBREAK-STOKER · 算例池条目的溯源与可复现性审计

生成的工件：`docs/case-fvm-dambreak-stoker/case_pool_entry.json` 对 `src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json`。本文件所有数字由
`scripts/audit_dambreak_pool_provenance.py` 注入，无手抄。

生成时间：`2026-11-16T06:48:40Z` · 用时 `6.8 s`

## 判定汇总

| 编号 | 问的问题 | 结果 |
|---|---|---|
| `P1_reproduces` | P1 存档数字是否还能从当前源码复现 | **PASS** |
| `P2_entry_is_derived` | P2 条目是否真是工件的派生（无手改/无单位滑移） | **PASS** |
| `P3_entry_is_self_contained` | P3 能否只凭条目重跑（输入是否记全） | **FAIL** |
| `P4_verdicts_follow_from_rows` | P4 判定是否由条目自己的行推得 | **PASS** |
| `P5_entry_is_generated` | P5 条目是否与生成器逐字节一致 | **PASS** |
| `P5_case_is_frozen` | P5 该算例是否被仓内 staleness 冻结网覆盖 | **FAIL** |
| `P6_auditor_has_power` | P6 审计器自身是否可失败（5 个注入探针） | **PASS** |

## P1 复现（当前源码 vs 存档工件）

| 算例 | n | 存档步数 | 重跑步数 | 最差判据相对差 | 该判据 |
|---|---:|---:|---:|---:|---|
| `ritter-dry` | 1600 | 1513 | 1513 | 0.000e+00 | `tv_no_growth<0.5%` |
| `stoker-wet` | 1600 | 918 | 918 | 0.000e+00 | `tv_no_growth<0.5%` |

容差 `1e-12`（串行确定性求解器，超出即是源码改了物理）。

存档 `discretisation` 相对当前 `run_case` 缺失的键（**答案**）:
* `ritter-dry`: ['dry_depth']
* `stoker-wet`: ['dry_depth']

## P2 条目 vs 工件重派生

判据重派生 15 条：drift 0 条、未映射 0 条。

## P3 条目的输入完备性

| 算例 | 已记录 | 缺失 |
|---|---|---|
| `ritter-dry` | cfl, dx, gravity, grid, h0, h1, order, t_star, x_dam, x_max, x_min | dry_depth |
| `stoker-wet` | cfl, dx, gravity, grid, h0, h1, order, t_star, x_dam, x_max, x_min | dry_depth |

## P4 判定代数与约束判据

自洽性问题 0 条。

`accepted_grid_min` = **400**（条目自己给出的）；若只用 primary 判据则为 **400**。

| 网格 n | 算例 | 失效判据 | 是否 primary |
|---:|---|---|---|
| 100 | `ritter-dry` | `front_wetted<5%` | **否（诊断量）** |
| 100 | `stoker-wet` | `monotone(no_extrema)` | 是 |
| 200 | `stoker-wet` | `monotone(no_extrema)` | 是 |

## P5 溯源

* 生成器逐字节一致：`True`（14411 / 14411 字节）
* staleness 审计覆盖的算例数：53；命中 dam-break 的算例：**无**
* 未冻结名字（unfrozen）共 18，其中含 shallow-water：False

## P6 审计器的可失败性（注入探针）

| 注入 | 应对检查 | 是否被发现 |
|---|---|---|
| artifact front_fan_error +10% | P1 reproduction | 是 |
| entry front_fan value +10% (hand edit) | P2 derivation | 是 |
| one criteria pass flag flipped | P4 verdict algebra | 是 |
| x_dam blanked in the entry | P3 config completeness | 是 |
| accepted_grid_min doubled | P4 verdict algebra | 是 |


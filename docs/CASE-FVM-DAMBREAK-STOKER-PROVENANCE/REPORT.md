# CASE-FVM-DAMBREAK-STOKER · 算例池条目的溯源与可复现性审计

**锚点**: `CASE-FVM-DAMBREAK-STOKER`（算例池·浅水溃坝 Ritter/Stoker 精确解）
**席位**: tp-ver-4（verifier 席，m23069 selfstart）
**性质**: 不出新的物理数字。审的是**证据链本身**：锚点的物理论证已由 15+ 张单覆盖
（精确解 / 1-D 六门 / 2-D / 斜置 / hires / 长时程 / 自相似 / oracle 独立性 / 配置包络）。
本单回答的是引用这些数字的人必须问、但**没有任何一张旧单问过**的溯源问题。

---

## ① 案例介绍（审计对象）

| 对象 | 路径 | 内容 |
|---|---|---|
| 池条目 | `docs/case-fvm-dambreak-stoker/case_pool_entry.json` | 六门判据、网格账、honest-FAIL 账（14411 字节） |
| 存档工件 | `src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json` | 池点 n=1600 的原始读数 |
| 生成器 | `src/tensorfvm/verification/case_pool_dambreak.py::build_entry` | 条目 → 工件是纯派生 |
| 求解器 | `shallow_water.swe1d` + `dam_break`：HLL + MUSCL(minmod) + SSP-RK3，CFL 0.4 | 1-D，串行确定性 |
| 冻结网 | `docs/verification-archived-source-staleness/staleness.json` | 53 例 × 1378 条源码绑定 |

判据口径不变：Ritter 1892 / Stoker 1957 精确解，`t* = 2`，容差 5 %。

## ② 怎么做的

新增 `scripts/audit_dambreak_pool_provenance.py`（7 项机械检查，数字全部程序化注入）：

* **P1 复现**：按条目声明的配置（n/CFL/order/t*）用**当前源码**重跑每个算例，把每条判据
  从原始读数重派生后与存档比对（容差 1e-12，串行确定性求解器）。
* **P2 派生**：判据值由脚本**独立重算**（不调用 `_criteria`，避免用生成器审计生成器），
  校验条目值、比较符、阈值、单位换算（分数→%，`tv_ratio−1`→%，比值型取绝对值）。
* **P3 完备性**：12 个能推动被判数字的输入（n/dx/CFL/order/t*/x_min/x_max/x_dam/h0/h1/g/干深下限）
  是否都能从条目读出 —— 决定「第三方能否只凭条目重跑」。
* **P4 判定代数**：`pass` 是否等于全部判据、`passing_grids`/`accepted_grid_min`/`honest_fail`
  是否由条目自己的网格行推得；并列出**约束判据**及其 primary 身份。
* **P5 溯源**：条目是否与生成器**逐字节**一致；该算例是否被仓内 staleness 冻结网覆盖。
* **P6 审计器自身的判决力**：5 个注入探针（工件数值 +10 %、条目手改 +10 %、翻转 pass 标志、
  抹掉 x_dam、accepted_grid_min 翻倍）必须逐一报警。

## ③ 计算结果（`summary.json` 程序化注入）

| 检查 | 结果 | 证据 |
|---|---|---|
| P1 复现 | **PASS** | ritter-dry / stoker-wet 最差判据相对差 **0.0**（逐位相同）；步数 1513 / 918 完全一致 |
| P2 条目派生 | **PASS** | 15 条判据全部由工件重派生，drift 0、未映射 0 |
| P3 输入完备 | **FAIL** | 修前缺 `x_min/x_max/x_dam/dry_depth`；修后仅缺 `dry_depth` |
| P4 判定代数 | **PASS** | 自洽性问题 0；`accepted_grid_min = 400` 与 primary-only 口径**一致** |
| P5 生成器一致 | **PASS** | 条目 14411 字节 vs 生成器 14411 字节，逐字节相同（无手改） |
| P5 冻结网覆盖 | **FAIL** | 该算例**未登记**在 staleness 审计中（18 个未冻结名字之一） |
| P6 审计器可失败 | **PASS** | 5/5 探针被对应检查抓到 |

约束判据（把池门卡在 n=400 以下的失效）：

| n | 算例 | 失效判据 | primary |
|---:|---|---|---|
| 100 | ritter-dry | `front_wetted<5%` | **否（池条目自己标为 diagnostic only）** |
| 100 | stoker-wet | `monotone(no_extrema)` | 是 |
| 200 | stoker-wet | `monotone(no_extrema)` | 是 |

## ④ 结果图

`fig_provenance.png`：(a) P1 两算例的最差判据相对差（对数轴，**0** 落在轴下限）+ 1e-12 容差线；
(b) 七项溯源判定 1/0 条形。图由脚本生成，无手绘。

## ⑤ 注意点（局限 / 适用范围）

1. **P1 的 0.0 是"同一台机器 + 同一 BLAS + 串行"下的逐位复现**，不是跨架构可移植性证明；
   跨架构可达性未测（not_run）。
2. **P2 的独立性有限**：判据的*原始读数*仍来自 `dam_break` 的估计子，本单只独立重做了
   「读数 → 判据值」这一步（单位换算 / 比较符 / 聚合），不能替代 oracle 独立（那是 D29 的活）。
3. **P3 的 `dry_depth` 仍缺**：存档 `discretisation` 是 `run_case` 加入该键**之前**生成的。
   源码已记录该键（`dam_break.run_case`），故**下次按驱动重生成工件即自动补齐**；
   本单未重生成工件（会改动 `wall_time_s` 等无关字节，收益 < 风险），故如实记 WITHELD。
4. **P5 冻结网缺口的后果**：锚点工件所在的 `docs/shallow-water-dam-break/` 没有
   `manifest.json`，所以「生产源码改过而工件没重跑」这类漂移**不会被 `tests/test_source_staleness.py` 抓到**。
   本次 P1 恰好证明当前没有这种漂移，但机制上仍是盲区，已在欠账账本登记。
5. **P6 探针只证明"检查有牙"，不证明判据选得对**。判据本身的口径敏感性见 `AUDIT.md` F1–F11、
   `docs/case-fvm-dambreak-stoker-config-envelope`。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 <3 %」）

| 量 | 值 | 基准 | 是否 <3 % |
|---|---:|---|---|
| 干床波前 `front_fan` | 0.0173 % | Ritter 解析 | 是 |
| 湿床水跃位置 | 0.0210 % | Stoker 解析 | 是 |
| 静水区水位阶跃 `h2/h1` | 5.8e-05 % | Stoker 解析 | 是 |
| 存档 vs 重跑（全部判据） | **0.0**（逐位） | 同源码重算 | 是（严格 0） |
| 条目 vs 生成器 | 0 字节差 | 生成器输出 | 是 |

**是否 < 3 %：全部是**（最大 0.0210 %）；复现差为严格 0。

### 附带修掉的一个 main 上的红测

`tests/test_source_staleness.py::test_live_tree_ledger_reproduces_and_ratchet_is_respected`
在本单开工时是**失败**的：账本冻结的普查 `fresh=1276 / changed=101`，实测 `1253 / 124`
—— 24 个本仓合并在不重生成账本的情况下改了 23 条被绑定源码。**关键性质**：
`unregistered_stale(report, known) == []`，即没有任何**算例**新变味（18 个既有 stale 案例
仍是那 18 个），棘轮（debt may not grow）未被突破，坏的只是普查快照。已按脚本
`scripts/audit_archived_source_staleness.py --update-ledger` 复位，`test_source_staleness.py`
与 `test_case_pool_dambreak_stoker.py` 现 19/19 通过。

---
*工件*: `docs/CASE-FVM-DAMBREAK-STOKER-PROVENANCE/{summary.json,SUMMARY.md,fig_provenance.png}`
*复现*: `PYTHONPATH=src python scripts/audit_dambreak_pool_provenance.py`
*测试*: `tests/test_dambreak_pool_provenance.py`

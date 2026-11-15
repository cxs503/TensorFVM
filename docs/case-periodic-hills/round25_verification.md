# CASE-FVM-PERIODIC-HILLS · round 25 — 网格收敛阶梯的「稳定时间步」补完

## ① 案例与验证欠账

`CASE-FVM-PERIODIC-HILLS`（ERCOFTAC UFR 3-30 / periodic hill，Breuer et al. 2009
*Computers & Fluids* 38:433，DOI 10.1016/j.compfluid.2008.05.002）。判据是时均底壁剪切
`τ_w/ρ` 的零点给出的再附长度 `x_r/H`，参考值 **5.41**（本仓权威口径
`periodic_hill_reference.X_R_OVER_H`），容差 10 %。几何 L=9H 周期、H=1、槽顶 3.036H；
Re=`U_b·H/ν`=2800（U_b=1）；闭包=无（laminar，`ν_t≡0`）。

**验证欠账（round 22/23 遗留）**：round 22 预注册了一个无闭包 `Re=2800` 的网格收敛阶梯
（`96x48 → 144x72 → 192x96 → 240x120`），并把三个加密档（B/C/D）按**粗网格协议 dt**
（0.0133 / 0.01 / 0.008）递交。三档全部在启动瞬态内发散（`diverged_at_step`=508/538/652，
`samples=0`），于是预注册的归因规则返回 `inconclusive`——**阶梯从未被真正测到**。
round 23 的廉价探针（`scripts/round23_divergence_probe.py`）回答了 round 22 留下的问题：
爆掉是**随网格加密而收紧的时间步稳定性极限**，不是网格固有失效——固定网格下把 dt 减半/四分之一
`max|u|` 就保持有界：

| 网格 | 粗 dt（发散） | 稳定 dt（有界） | 稳定 CFL |
|---|---|---|---|
| 144x72 | 0.0133（step 508 爆） | **0.00665**（max\|u\|≈1.6） | ~0.16 |
| 192x96 | 0.01（step 538 爆） | **0.0025**（max\|u\|≈1.7） | ~0.08 |
| 240x120 | 0.008（step 652 爆） | **0.004**（同 144x72 的 CFL） | ~0.16 |

## ② 怎么做的（round 25 补完）

新增 `scripts/round25_stable_ladder.py`（驱动）与 `scripts/round25_ladder_analysis.py`（分析）：

* **驱动**：把三个加密档按 round-23 探针证明的**稳定 dt** 重跑，其余旋钮与 round-20 协议完全一致
  （地形跟随网格族、动量对流混合 0.75、输运混合 0.0、统计窗 `t∈[160,400]`、认证 round-17 壁剪算子），
  并写 `round25_laminar_Re2800_<tag>` 工件——**不覆盖** round-23 的历史记录（发散的
  `round20_laminar_*` 档保留原样）。
* **分析**：从存盘**时均壁剪**用认证算子重算每一档 `x_r/H`（不读 payload 派生字段），
  把四个实测档喂给预注册归因规则 `periodic_hill_grid_ladder.attribute`，产出 `round25_table.json`。
* **测试**（`tests/test_periodic_hill_round25.py`，5 条）：钉住①稳定 dt 必须等于 round-23 探针
  证明值（0.00665/0.0025/0.004）；②每档必须到达物理窗 `t∈[160,400]`；③分析必须读 `round25_*`
  工件（与 round-20 隔离）；④计算仍在排队时缺档必须优雅跳过不崩；⑤基线档（round-20 96x48）单独
  在场时分析能出表。

## ③ 计算结果（待集群回执）

三个加密档已递交集群（`cj4136`=144x72@dt0.00665、`cj4137`=192x96@dt0.0025、
`cj4140`=240x120@dt0.004），排队中。回执后 `round25_ladder_analysis.py` 会产出
`round25_table.json` 的实测 `x_r/H` 阶梯与归因判定。**本单交付的是「补完基础设施 + 稳定 dt 证据链」**，
实测数字由回执后的分析脚本落盘（见 `docs/case-periodic-hills/round25_table.json`）。

## ④ 误差判定（当前基线）

当前唯一实测档（round-20 96x48，dt=0.02）`x_r/H`=8.26，对 5.41 误差 **+52.6%**（>10% 带，FAIL）。
round 23 已钉住该档对 dt 不敏感（0.42%）。加密档是否把 `x_r/H` 拉向 5.41（网格收敛）还是保持
~8（系统性过预测），是 round 25 回执要回答的问题——**这正是把「inconclusive」变成「measured」的
验证欠账**。

## ⑤ 注意点 / 局限

* 本单在 load15≈154（≥100）的容量纪律下只做**轻量**（文档/测试/分析）活；重算力（60000–160000 步）
  全部递交集群，不压本机。
* round-23 的 `test_refined_rungs_are_unusable` 等历史测试仍钉 round-23 的「发散」状态（读
  `round20_laminar_*` 档），与 round-25 的 `round25_*` 档**互不覆盖**——两轮记录并存，不互相改写。
* 若回执显示加密档仍不收敛到 5.41，则结论是「系统性过预测」（求解器/闭包问题），需另立
  `CASE-FVM-PERIODIC-HILLS` 的闭包/数值方案子条目——本单不预设该结论。

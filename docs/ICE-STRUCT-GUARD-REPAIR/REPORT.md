# 冰区越域护栏：从 schema 层进入物理计算入口（JOUR-ICE-STRUCT-GUARD / m22342）

## ① 案例介绍

**物理背景。** 船舶在层冰中航行时，总阻力按 Lindqvist (1989) 分解为挤压、弯曲、浸没、碎冰拖曳四项。
该公式的每一项都是经验/半经验拟合，只在有限的冰厚、冰强度、航速、船体主尺度范围内有效。

**锚点故障（记录于 2026-11-09 平台构建）。** 冰域输入 `h = 50 m` 与 `σ_c = 100 MPa` 被静默外推，
产出 2450 MN 级与 143.9 MN 的"阻力"；而当时的护栏只在 schema 层（Pydantic 类型检查），
12 个输入中有 7 个可越域而不被拦截。同一锚点记录 `GET /ice-uq/ships` 返回 HTTP 500
（`KeyError 'std_resistance_kN'`）。

**本轮复测（MOCK 排除）。** 本次对该端点实测返回 HTTP 200，三家船的响应中
`reference_std_resistance_kN` 仍为 `null`、`geometry_consistent` 均为 `false`
（位移与 L·B·T·Cb 隐含位移偏差 −29.3 % / +24.8 % / +13.2 %）——即 KeyError 的直接触发条件
（空值）仍在，只是当前构建不再 500。**该端点属平台侧（非 TensorFVM 仓），本单不改它，
只作为"平台侧残余缺口"如实登记。**

**参数表（12 个输入，程序化从注册表渲染）。** 见 `SPEC.md` §3。

## ② 怎么做的

1. **修 NaN 漏洞（m22209 的 F1 阻塞缺陷）。** `is_hard_violation` 用 `<`/`>` 比较，NaN 与任何数比较恒
   False，护栏把 NaN 标成 "TYPICAL" 放行。现：`parameters.is_hard_violation` 先判 `math.isfinite`；
   `guardrails` 的 validate / clamp / risk / report 四路统一走一个 `_finite_value` 强转，
   `nan` 一律 `fail` 且**不回吐**，`±inf` 判拒绝、clamp 时落到已知方向的硬边界，`bool` 显式拒绝
   （`isinstance(True, int)` 会把 `True` 静默变成 1.0 m 冰厚）。
2. **把护栏扩到完整输入集（8 → 12）。** 新增 4 个船体参数（`ship_length`/`ship_beam`/`ship_draft`/
   `ship_block_coefficient`）三级边界，注册 `ICE_RESISTANCE_INPUTS`，并保持
   `typical ⊆ soft ⊆ hard` 嵌套不变式（12/12 成立，见 G1 覆盖）。
3. **护栏进入物理入口，而非停在 schema。** 新增 `tensorfvm/ice/load.py`：Lindqvist 层冰阻力
   （`lindqvist_terms` / `level_ice_resistance_kN`）与**受护入口** `guarded_level_ice_resistance`，
   默认 `mode='reject'`：任何越域或非有限输入都拿不到 `resistance_kN`（返回 `None` + 完整护栏结论）；
   `mode='clamp'` 是唯一让越域数值进入公式的通道，且必然带 `extrapolation_risk` 标记。
   `require_guarded_level_ice_resistance` 提供抛异常式硬前置条件。
4. **接入真实求解路径（集成调用方 2 处）。**
   (a) `tensorfvm/ice_channel.py`（冰下槽道 FVM/1-D 求解器）新增 `ice_lid_guard`，在 `run_all` 主流程里
   于任何流体求解之前校验冰盖状态，不可能状态直接抛 `IceParameterDomainError`，护栏结论写入
   `summary.json`；(b) `tensorfvm/__init__.py` 导出 `ice` 子包，`pyproject.toml` 注册
   `tensorfvm-ice` 控制台入口，`python -m tensorfvm.ice` 为正式 CLI 入口。
5. **判据机检（不靠自声明）。** 新增 `tensorfvm/ice/benchmark.py`：
   G1 用**独立转录的规范表 + 独立实现的判定规则**（不调用实现代码）在 180 个边界探针上比对；
   G2/G3 非有限值围堵率；G4 越域输入在载荷算术前的拦截率（24 次注入）；
   G5 域内输入直通（护栏不得扰动结果）；G6 锚点历史量级复现（结论 WITHHELD，见 ⑥）。
   `G1` 的规范表是与 `parameters.py` **分开维护的第二份拷贝**，因此该行真正度量的是"规范↔代码漂移"。

## ③ 计算结果

**A. 判据行（`docs/ICE-STRUCT-GUARD-REPAIR/artifacts/benchmark.json`）**

| 行 | 参考 | 实测 | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| G1 | oracle over 180 boundary probes | 180/180 agree | 0.0000 % | 3.00 % | PASS |
| G2 | 1.0 | 1.0 | 0.0000 % | 0.00 % | PASS |
| G3 | 1.0 | 1.0 | 0.0000 % | 0.00 % | PASS |
| G4 | 1.0 | 1.0 | 0.0000 % | 0.00 % | PASS |
| G5 | guarded == unguarded | max rel diff 0.000e+00 | 0.000e+00 | 1e-12 | PASS |
| G6 | 2450 MN (h=50 m) / 143.9 MN (sigma_c=100 MPa) as recorded 2026-11-09 | max rel deviation 0.637 | 6.372e-01 | 3e-02 | **WITHHELD** |

**B. 冰厚扫描：受护 vs 未受护（`detail.thickness_sweep`）**

| h [m] | 未受护 R [MN] | 受护 R [MN] |
|---|---|---|
| 0.05 | 0.3 | 0.3 |
| 0.1 | 0.5 | 0.5 |
| 0.3 | 1.6 | 1.6 |
| 0.5 | 2.8 | 2.8 |
| 1 | 6.4 | 6.4 |
| 2 | 15.9 | 15.9 |
| 3 | 28.5 | 28.5 |
| 5 | 61.9 | 61.9 |
| 8 | 132.0 | blocked |
| 10 | 191.7 | blocked |
| 20 | 640.1 | blocked |
| 50 | 3,428.9 | blocked |

**C. 锚点历史量级复现（`detail.g6_legacy`）**

| 用例 | 锚点记录 | 仓内复现 | 相对偏差 | 是否被拦截 |
|---|---|---|---|---|
| h_50m | 2450 MN | 3,428.9 MN | 39.95 % | yes |
| sigma_c_100MPa | 143.9 MN | 235.6 MN | 63.72 % | yes |

**D. 集成调用方的物理正确性（`docs/ICE-STRUCT-GUARD-REPAIR/channel_metrics.json`）**

| 量 | 值 |
|---|---|
| slip 1-D 速度剖面 L2（vs 解析） | 0.0228 % |
| no-slip FVM 速度剖面 L2（vs 解析，72x12） | 0.6346 % |
| no-slip dp/dx 相对误差 | 1.3690 % |
| `run_all` 全部判据 | PASS |

**E. 单测。** `tests/test_ice_guard_pipeline.py`（新，126 例参数化展开）+ 既有
`tests/test_ice_guardrails.py` + `tests/test_ice_channel.py`：**181 passed / 0 failed**。
`ruff check src/tensorfvm/ice src/tensorfvm/ice_channel.py src/tensorfvm/__init__.py tests/test_ice_guard_pipeline.py tests/test_ice_guardrails.py`：**All checks passed**。

## ④ 结果图

* `docs/ICE-STRUCT-GUARD-REPAIR/fig_guard_gate.png` — 左：G1–G5 各行"实测误差 vs 声明容差"（对数轴，
  程序化注入）；右：三个域内船例的受护阻力（G5 直通，护栏零扰动）。
* `docs/ICE-STRUCT-GUARD-REPAIR/fig_ice_channel_physics.png` — 左：冰下槽道 slip 解析剖面 vs 1-D FV 数值剖面
  （集成调用方在护栏内仍解出正确物理）；右：逐点速度相对误差，3 % 带为参考线。
* `docs/ICE-STRUCT-GUARD-REPAIR/artifacts/fig_ice_guard_benchmark.png` — 12 参数三级边界带 + 围堵率 + 冰厚扫描（h=50 m 处 10³ MN）。

**关于"场量云图"的说明：** 护栏是参数域校验器，本身不产生场解，故其判据图不是云图；作为替代，
集成调用方（冰下槽道求解器）的解析解对照曲线（第二张图）承担了"与解析解比较"这一职责，
护栏自身的判据则在第一张图与 ③-A 表中逐行给出。两者不等价，此处显式声明。

## ⑤ 注意点（坑与局限）

1. **G6 是诚实 FAIL/WITHHELD。** 仓内复现的 Lindqvist 值（3,428.9 MN /
   235.6 MN）与锚点记录的 2450 / 143.9 MN 差
   40.0 % / 63.7 %：产生那两个数的平台构建不在本仓，
   公式版本（速度乘子、弯曲项、浸没项长度尺度）在 m1720 前后不同，因此**量级不复现**。
   可确证的只有失效模式本身：两个输入都被拦截，且未受护值均在 10³ MN 级（远超任何实船量级）。
2. **阻力公式精度不作声明。** 本模块用未标定 Lindqvist；同公式在平台侧对三条实船试验均值的
   留一交叉验证误差约 −23 %～+17 %（超出 15 % 带），故阻力精度为 WITHHELD，本仓 benchmark 只评护栏。
   公式中 `0.3 L`（浸没项长度尺度）与 `0.1`（拖曳系数）两常数无法溯源到原文，已在 `load.py` 注释中登记。
3. **边界语义。** `h` 的硬上界 10 m 是"船-层冰模型的适用上限"，不是物理不可能（冰脊龙骨、冰川舌远超 10 m）；
   已改写 docstring，避免"physically impossible"的过强表述（m22252 审计 F8）。
4. **静态边界。** 不做温度/盐度依赖边界，也不做相关性检查（厚冰 + 低强度这类组合仍会放行）；
   `bow_angle_deg` 不在 12 参数集内，由载荷入口显式传入。
5. **平台侧残余缺口未修。** `GET /ice-uq/ships` 现返回 200，但 `reference_std_resistance_kN` 仍为 `null`、
   三条船 `geometry_consistent` 全为 `false`；该文件不在本仓，按"只改工作树内文件"的纪律未动。
6. **clamp 语义的边界。** `clamp` 只对有限值/`±inf` 有意义；`nan` 被拒绝而不是"钳到某边界"，
   因为"NaN 在哪一侧"没有答案。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准（ref） | 实测（got） | 误差 | <3 %? |
|---|---|---|---|---|
| G1 三级判定（180 个边界探针） | 独立规范表 + 独立规则 | 实现判定 | 0.0000 % | **是** |
| G2 NaN 围堵率 | 100 % | 100.0 % | 0 pp | **是** |
| G3 ±inf 围堵率（含 clamp 落界有限性） | 100 % | 100.0 % | 0 pp | **是** |
| G4 越域输入在载荷算术前拦截率 | 100 % | 100.0 % | 0 pp | **是** |
| G5 域内直通（护栏对载荷的扰动） | 0 | 0.000e+00 | 0 % | **是** |
| 冰下槽道 slip 速度剖面 L2 | 解析解 | 0.0228 % | — | **是** |
| 冰下槽道 no-slip dp/dx | 解析解 −12μU/H² | 1.3690 % | — | **是** |
| G6 越域载荷量级复现（h=50 m） | 2450 MN（2026-11-09 平台构建） | 3,428.9 MN | 40.0 % | **否（WITHHELD）** |
| G6 越域载荷量级复现（σ_c=100 MPa） | 143.9 MN（同上） | 235.6 MN | 63.7 % | **否（WITHHELD）** |

**结论：** 护栏自身的目的性判据（越域/非有限输入不得进入物理算术）全部 0 偏差达标（远严于 3 %）；
集成调用方的物理量对解析解误差 0.023 %–1.37 %，亦 <3 %。
**但**锚点记录的两个越域载荷量级无法在仓内复现（39.9 % / 63.7 %），该肢诚实记 WITHHELD，不粉饰。

## 复现

```
PYTHONPATH=src python -m tensorfvm.ice --out docs/ICE-STRUCT-GUARD-REPAIR/artifacts
PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_figures.py
PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_report.py
PYTHONPATH=src python -m pytest tests/test_ice_guard_pipeline.py tests/test_ice_guardrails.py tests/test_ice_channel.py -q
```

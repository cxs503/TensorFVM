# CASE-FVM-BLASIUS-PLATE — 平板层流边界层摩阻系数 / 形状因子 vs Blasius 相似解

算例池锚点：`CASE-FVM-BLASIUS-PLATE`（第 1 轮·任务榜·算例池，mission 16182，仓 `TensorFVM`）
判据：`Re_x = 1e5` 处局部摩阻系数 `c_f` 与形状因子 `H = δ*/θ` 相对 Blasius 相似解误差 **< 3 %**。

本篇为生成式报告：文中每个数字都来自 `runs/blasius-plate/*.json`（由
`scripts/run_case_blasius_plate.py` 写出）或 `runs/blasius-plate/fvm-partial-traces.json`
（由 `scripts/plot_case_blasius_plate.py` 从运行日志解析），无手抄。

---

## ① 问题 + 参考值出处

**物理问题。** 零压梯度光滑平板，不可压缩，从前缘起层流。以 `U∞` 为参考速度、板长 `L`
为参考长度，`Re_L = U∞ L / ν = 1e5`；比较站位取后缘所在列（`Re_x ≈ Re_L`）。
层流边界层相似解给出

```
c_f = 0.664 / sqrt(Re_x),   δ*/x = 1.7208 / sqrt(Re_x),
θ/x  = 0.664  / sqrt(Re_x),  H = δ*/θ = 2.5916
```

**参考值来源（文献，非本仓输出）。**

| 量 | 参考值 | 出处 |
|---|---|---|
| `c_f√Re_x` | 0.664 | Blasius, H. (1908), *Grenzschichten in Flüssigkeiten mit kleiner Reibung*, **Z. Math. Phys. 56, 1–37**；Schlichting, *Boundary-Layer Theory*, 7th ed., **§7, Table 7.1** |
| `δ*/x·√Re_x` | 1.7208 | 同上 Table 7.1 |
| `θ/x·√Re_x` | 0.664 | 同上 Table 7.1 |
| `H` | 2.5916 (= 1.7208/0.664) | 同上 Table 7.1 |
| `f''(0)` | 0.332057 | 同上（Howarth 1938 表值） |
| 次级 DOI（算例单给出） | `10.1002/zamm.19800600419` | 阻力工程手册语境 |

**参考值自审（防转录错误）。** `case_blasius_plate.similarity_crosscheck()` 在本仓独立
求解 Blasius 方程 `f''' + ff''/2 = 0`（RK4 + Newton 打靶），得到
`2 f''(0) = 0.664115`（与表值 0.664 差 **0.017 %**），积分剖面得
`δ*√Re = 1.720790`、`θ√Re = 0.664112`、`H = 2.59112`（与表值 2.5916 差 **0.019 %**）。
即所用文献数字可由第一性原理复现到 0.02 %，不是凭记忆填入。

## ② 方法与实现

新增 `src/tensorfvm/case_blasius_plate.py`（算例定义 + 两条测量通道 + 判据）与
`scripts/run_case_blasius_plate.py`（CLI，写机器可读工件），另加
`tests/test_case_blasius_plate.py`（7 个测试）。

两条测量通道：

* **march（经济档，本次给出判据结论）**：`tensorfvm.rans_ship_resistance.march_boundary_layer`
  （薄剪切层边界层推进，`model='laminar'`，`trip_x=None`，后向 Euler + 二阶壁法向），
  即 `docs/RANS_verification.md` 中 C1/C8 锚所用同一设备。
* **fvm（旗舰档，本次 WITHHELD）**：`tensorfvm.body_fitted.BodyFittedSolver`（二维不可压缩
  SIMPLE，`mesh_type='flat-plate'` 拉伸贴体网格，层流闭合）。局部壁面剪应力取**求解器自身**
  无滑移面力重构（逐面调用 `_surface_force`），其逐面和等于整板阻力（工件中
  `force_sum_closure = 0`），因此比较的是 FVM 离散误差而非后处理模型。

剖面处理：`boundary_layer_integrals()` 用分段线性（梯形）从壁面 `u=0` 积分整个给定剖面；
`u_e` 默认取最外节点（局部外缘速度）。**截断到 `δ99` 会让 θ 偏低约 0.9 %**（测试
`test_boundary_layer_integrals_on_exact_blasius_profile` 把该偏差钉在 0.5 % 内），故默认不截断。

## ③ 参数表（无量纲数完整）

| 名称 | 符号 | 值 |
|---|---|---|
| 雷诺数（板长） | `Re_L = U∞ L/ν` | 1.0e5 |
| 比较站位雷诺数 | `Re_x` | 1.0e5（march）/ 9.79e4 ~ 1.00e5（FVM 后缘列） |
| 运动黏度 | `ν = U∞ L/Re_L` | 1.0e-5 |
| 入流速度 / 密度 | `U∞ / ρ` | 1.0 / 1.0 |
| 域高 | `H_dom` | 0.15 (= 9.5 δ99) |
| FVM 壁法向拉伸 | `a` | 4.0（首格高 2.8e-4 ~ 3.1e-4） |
| march 壁法向节点 / 站位 | `ny` / `n_x` | 256 / 401（基准），128/512 与 101/1601（敏感性） |
| march 拉伸 / 域高 | `a` / `Y(L)` | 8.0 / 0.0923 (= 5.8 δ99) |
| 判据 | `|Δ|/ref` | < 3 % |

## ④ 对照表（ref vs got + 误差打勾叉）

**主表（march，基准配置 ny=256、401 站位）** — `runs/blasius-plate/march.json`

| 量 | ref (Blasius) | got (TensorFVM march) | 相对误差 | 判据 <3 % |
|---|---|---|---|---|
| 局部 `c_f` @Re_x=1e5 | 2.09975e-3 | 2.08492e-3 | **−0.71 %** | ✅ |
| 形状因子 `H` | 2.5916 | 2.57945 | **−0.47 %** | ✅ |
| `δ*` | 5.44165e-3 | 5.50656e-3 | +1.19 % | （诊断量） |
| `θ` | 2.09975e-3 | 2.13478e-3 | +1.67 % | （诊断量） |

δ*/θ 单量各自偏 1.2 %/1.7 %，但比值 H 偏 0.47 %：推进法的误差在 δ*、θ 上是同号的
外缘速度/积分定义差异，取比值后大部分抵消——这正是 `H` 作为相似性判据更稳健的原因。

**FVM 旗舰档** — 本轮 **WITHHELD**（未收敛，见 ⑥）。已完成的迭代数、`c_f` 与残差轨迹在
`runs/blasius-plate/fvm-partial-traces.json`：`fvm-relaxation` 第 75 步
`c_f = 7.66e-3`（仍为 Blasius 值的 3.6 倍）、`momentum = 4.4e-3`；`fvm-baseline` 第 75 步
`c_f = 1.35e-2`、`momentum = 8.95e-3`。**不把未收敛的数值当结论**，故不给 FVM 打勾/叉。

## ⑤ 收敛性与参数敏感性（≥2 点）

图 3（`figures/fig3_march_sensitivity.png`）与 `runs/blasius-plate/march-sensitivity.json`：

| 变化 | `c_f` 误差 | `H` 误差 | 相对基准变化 `Δc_f` / `ΔH` |
|---|---|---|---|
| `ny = 128` | −1.81 % | −0.83 % | −1.11 % / −0.37 % |
| `ny = 256`（基准） | −0.71 % | −0.47 % | — |
| `ny = 512` | **−0.15 %** | −0.30 % | +0.56 % / +0.17 % |
| 站位 101 | −0.99 % | −0.71 % | −0.83 % / −0.24 % |
| 站位 1601 | −0.54 % | −0.41 % | +0.17 % / +0.06 % |

* **壁法向收敛（点 1）**：`ny` 128→512 时 `c_f` 误差 −1.81 %→−0.15 %，单调趋零，说明
  −0.71 %（基准）主要是离散误差而非模型误差，且外推极限落在判据内。
* **流向收敛（点 2）**：101→1601 站位把 `c_f` 误差从 −0.99 % 收窄到 −0.54 %，量级远小于 3 %。
* **流向前缘误差带**：图 1 右图显示 `Re_x ≲ 1e3` 处误差可达 9.5 %（推进法起步段），
  `Re_x ≥ 1e4` 后稳定在 −0.15 %。本算例的比较站位 `Re_x = 1e5` 不在该带内。
* **剖面相似性（图 2）**：`η ≤ 4` 内 `u/U_e` 与 Blasius 的最大偏差 **0.24 %**。
* FVM 档的伪时间/松弛敏感性（`fvm-baseline` `Δt=0.02, α_u=0.3` vs `fvm-relaxation`
  `Δt=0.05, α_u=0.5`）在同样 75 步下把 `c_f` 从 1.35e-2 推到 7.66e-3，说明外层迭代的
  伪时间步是收敛速率主控参数，但不改变「仍未收敛」的事实。

## ⑥ 失败模式、工件与生态摩擦

**FVM 旗舰档为何 WITHHELD（honest，不粉饰）。**
* 单个 SIMPLE 外层迭代实测 **4.4 ~ 8.5 s**（`nx×ny = 64×56`，3 584 格，2 线程，99 共享节点
  负载 135~209 的实测值）。剖析显示耗时几乎全部来自压力修正内层迭代：
  `_correct()` 对 `flat-plate` 硬编码 `threshold_floor=1e-8, relative_tolerance=1e-8`，
  每步约 **1 341 次**内层 `operator` 调用。
* 按当前残差衰减率外推，收敛到 `tolerance=1e-5` 需 ~1 000 次外层迭代 ⇒ **每题 ~1.2–2.5 h**；
  本单时间窗内做不完 4 个网格级（基线/松弛/流向/壁法向）。
* 集群递交（`cj1256/cj1257/cj1258`）自 10:20 起 queued，至 10:25 仍在队首附近
  （`queued_ahead ≈ 428`），更早的 `cj1230~cj1248`（09:05–09:52 递交）同样未派发——
  调度器**在观测窗口内没有任何作业被派发**，双节点 `n315/n330` 自报 health 正常、load 37/35。
  故 FVM 档不能靠集群在本轮补齐。

**生态摩擦（如实登记，供平台/仓库侧改进）**

| # | 摩擦 | 证据 | 影响 |
|---|---|---|---|
| F1 | 集群调度停滞：递交 1 h 内无作业派发 | `run_cluster ps`：`cj1230`(09:05)…`cj1257` 全 `queued`；`nodes` 显示 n315/n330 ok、load 37/35 | 重计算出口不可用，只能压 99 控制面或降级 |
| F2 | 99 控制面高负载（load 135~209 / 128 核） | `uptime` 10:25 实测 | 单迭代耗时随负载波动 ~2× |
| F3 | `flat-plate` 压力内层容差硬编码，无配置出口 | `body_fitted.py:_correct()` `threshold_floor=1e-8 if flat_plate` | 该网格上 1 341 内层迭代/步，是唯一的耗时来源；SKIP |
| F4 | TensorPlatform 知识引擎离线 | `GET /tensor/knowledge/search` → `engine_offline`（`127.0.0.1:8021` 拒连） | 开工三查中的 knowledge_base/oss_solutions 两查无法执行 |
| F5 | 能力语义工件缺失 | `GET /tensor/capabilities/semantics` → 503 `semantic_layers.json` 缺失 | 无法在平台上核对 TensorFVM 能力等级 |

**工件清单**

| 路径 | 内容 |
|---|---|
| `src/tensorfvm/case_blasius_plate.py` | 算例定义、参考值、相似解自审、两条测量通道、判据 |
| `scripts/run_case_blasius_plate.py` | CLI：`--configs march\|march-sensitivity\|fvm-*`，写 JSON 工件 |
| `scripts/plot_case_blasius_plate.py` | 由工件与运行日志出图 + 解析 FVM 部分轨迹 |
| `tests/test_case_blasius_plate.py` | 7 测试（参考值/自审/剖面积分/判据/两条通道） |
| `runs/blasius-plate/march.json` | march 基准结果（c_f、H、δ*、θ、残差） |
| `runs/blasius-plate/march-sensitivity.json` | ny/站位敏感性 5 点 |
| `runs/blasius-plate/fvm-partial-traces.json` | FVM 已完成的迭代轨迹与成本证据 |
| `runs/blasius-plate/summary.json` | 合并摘要（含文献自审） |
| `docs/case-blasius-plate/figures/fig1..fig4` | 局部 c_f 对比/剖面/敏感性/FVM 收敛轨迹 |

**复现（一键）**

```bash
# 经济档 + 敏感性 + 全部图（~1 min）
PYTHONPATH=src python3 scripts/run_case_blasius_plate.py --out runs/blasius-plate \
    --configs march,march-sensitivity && PYTHONPATH=src python3 scripts/plot_case_blasius_plate.py
# 测试
PYTHONPATH=src python3 -m pytest tests/test_case_blasius_plate.py -q
# 旗舰档（需 ~1.2–2.5 h/网格级；建议集群）
PYTHONPATH=src python3 scripts/run_case_blasius_plate.py --out runs/blasius-plate \
    --configs fvm-relaxation --max-iterations 1200 --threads 8
```

**结论（判据逐项）**

| 目标量 @Re_x=1e5 | 误差 | <3 % ? |
|---|---|---|
| 局部 `c_f`（march，ny=256） | −0.71 % | **是** |
| `H`（march，ny=256） | −0.47 % | **是** |
| 局部 `c_f`（march，ny=512，收敛后） | −0.15 % | **是** |
| 局部 `c_f` / `H`（2-D FVM） | 未收敛 | **WITHHELD**（不声明达标） |

适用边界：结论只覆盖零压梯度、层流、不可压缩平板；FVM 档需先解决 ⑥ 的收敛成本问题，
或在集群恢复派发后补跑。
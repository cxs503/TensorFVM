# CASE-FVM-DAMBREAK-STOKER — 收敛阶审计：算例池里的 `order: 2` 到底证明了什么

**锚点**：`CASE-FVM-DAMBREAK-STOKER`（算例池·浅水溃坝 Ritter/Stoker 精确解）
**仓**：TensorFVM · **分支**：`agent/m21759` · **求解器一行未改**
**新增**：`src/tensorfvm/shallow_water/order_audit.py`（审计模块）、
`scripts/run_shallow_water_order_audit.py`（工件生成）、
`tests/test_shallow_water_order_audit.py`（20 例回归门，9.8 s，纯 numpy）
**工件**：`src/tensorfvm/verification/artifacts/shallow-water-order-audit/{summary.json,entry_claim.json}`

---

## ① 案例介绍（物理背景 / 参数表）

一维浅水方程（静水压、平底、无摩阻）的溃坝黎曼问题，两个闭式解：

| 参数 | Ritter 干床 | Stoker 湿床 | 出处 |
|---|---|---|---|
| 上游深度 h0 | 1.0 m | 1.0 m | 总账锚点 |
| 下游深度 h1 | 0.0（干） | 0.5 m | 同上 |
| 重力 g | 9.81 m/s² | 9.81 m/s² | 同上 |
| 计算域 | x ∈ [-4, 6] m，坝 x=0 | 同 | `shallow_water.dam_break` |
| 分级时刻 | t* = 2（t√(g/h0)） | 同 | 总账容差串 |
| 格式 | MUSCL/minmod + HLL + SSP-RK2，CFL 0.4 | 同 | `shallow_water.swe1d` |
| 分级量（总账） | x_f(t)、水位阶跃 h2/h1、波形单调 | 同 | 5 % 容差 |
| 本单新增的量 | **扇区内部（解析窗口内）误差的观测阶** | 同 | 本报告 |

**本单要回答的问题**：算例池条目 `docs/case-fvm-dambreak-stoker/case_pool_entry.json`
写着 `solver.order = 2`，而仓内**所有**已合并的收敛证据都是 ≈1.0（深度 L1：1.011 / 0.988；
`x_f(t)` 序列 0.68–0.73）。已合并报告的解释是「TVD 格式含间断解时理论上一阶」——这是关于**格式**
的一句话，它没有说**这个算例里到底是哪一个奇异点**占住了误差，也没有任何已提交实验能否证它。
总账要求的不是再算一遍 5 %，而是把这个 `order` 声明变成**可测、可失败**的东西。

## ② 怎么做的（方法 / 流程 / 关键设置）

### 2.1 把「算子阶」和「数据阶」拆开

一个观测阶只能被两件事限制：**算子**和**数据/解**。三个实验把两者分开：

| 实验 | 做法 | 测的是 |
|---|---|---|
| `smooth_bump_study` | 静止水域中的高斯隆起（A=0.1 m, σ=2 m），演到 t=1 s，在同一族网格上与**自身**做 3 网格 Richardson（细网格按格平均限制到粗网格） | **算子阶**（解始终光滑，无参考解、无手抄数） |
| `fan_interior_study(init='step')` | 物理溃坝：**间断**初始水深 | 被锚定的那一类行为 |
| `fan_interior_study(init='smooth')` | 用 t*=0.5 的**闭式解重启**（干床数据连续；湿床的闭式解本身已含涌波） | 数据正则性 |

### 2.2 分级窗口来自解析解，不含拍脑袋数字

`ritter_fan_window` / `stoker_fan_window` 从闭式解取扇区边界（Ritter：ξ∈[-c0, 2c0]；
Stoker：ξ∈[-c0, u2-c2]，`stoker_region_bounds` 给出），两端各裁掉
`FAN_INTERIOR_FRACTION = 0.15 ×` 扇区宽度，把干床前锋、涌波、两个折点全部挡在窗外。
所以窗内测的是**内部算子**，不是折点的分辨率。

### 2.3 O(dx) 判定：`err/dx` 应网格无关

若内部误差真的是一阶自相似极限，则把误差除以 dx、对 ξ 作图应当逐网格重合。
`collapse_finest_relative` = 最细两档归一化剖面的最大相对差；判据 `≤ 5 %`
（实测干床 1.0 %、湿床 0.6 %，即 5–8 倍余量）。

### 2.4 门禁与「门禁必须能失败」

`_verdict()` 里 6 条 gate + 1 条 diagnostic；`audit_entry_claim()` 把条目里的
`order` 与实测光滑区阶做绑定，顺序错配即 `OVERSTATED` / `UNDERSTATED`
（测试用 order=3 / order=1 两条反例证明它可以失败），无值则 `WITHHELD`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 观测阶与 O(dx) 塌缩

| study | ladder (n) | observed order p | err/dx collapse (finest) | gradable |
|---|---|---:|---:|---|
| ritter-dry/step | 100-3200 | 1.00 | 1.0 % | yes |
| ritter-dry/smooth | 100-3200 | 1.98 | 101.1 % | yes |
| stoker-wet/step | 100-3200 | 1.04 | 0.6 % | yes |
| stoker-wet/smooth | 100-3200 | 2.70 | 512.1 % | **no (non-monotone)** |
| smooth bump (self-conv.) | 200-3200 | 2.04 | — | yes |

光滑隆起自收敛：最细两对 Richardson 阶 **2.04**
（最小二乘拟合 1.99）——**算子在光滑区确为二阶**。

### 3.2 逐网格内部误差

| study | n | dx | interior L1 | interior Linf | graded cells |
|---|---:|---:|---:|---:|---:|
| ritter-dry/step | 100 | 0.10000 | 6.663e-03 | 1.932e-02 | 42 |
| ritter-dry/step | 200 | 0.05000 | 2.741e-03 | 4.338e-03 | 84 |
| ritter-dry/step | 400 | 0.02500 | 1.343e-03 | 1.816e-03 | 168 |
| ritter-dry/step | 800 | 0.01250 | 6.714e-04 | 9.090e-04 | 336 |
| ritter-dry/step | 1600 | 0.00625 | 3.356e-04 | 4.548e-04 | 672 |
| ritter-dry/step | 3200 | 0.00313 | 1.678e-04 | 2.274e-04 | 1344 |
| ritter-dry/smooth | 100 | 0.10000 | 3.728e-04 | 9.137e-04 | 42 |
| ritter-dry/smooth | 200 | 0.05000 | 8.000e-05 | 1.605e-04 | 84 |
| ritter-dry/smooth | 400 | 0.02500 | 1.994e-05 | 4.188e-05 | 168 |
| ritter-dry/smooth | 800 | 0.01250 | 5.078e-06 | 1.074e-05 | 336 |
| ritter-dry/smooth | 1600 | 0.00625 | 1.285e-06 | 2.720e-06 | 672 |
| ritter-dry/smooth | 3200 | 0.00313 | 3.240e-07 | 6.826e-07 | 1344 |
| stoker-wet/step | 100 | 0.10000 | 8.844e-03 | 1.936e-02 | 7 |
| stoker-wet/step | 200 | 0.05000 | 3.496e-03 | 9.569e-03 | 12 |
| stoker-wet/step | 400 | 0.02500 | 1.720e-03 | 4.978e-03 | 25 |
| stoker-wet/step | 800 | 0.01250 | 8.052e-04 | 2.540e-03 | 49 |
| stoker-wet/step | 1600 | 0.00625 | 4.071e-04 | 1.291e-03 | 99 |
| stoker-wet/step | 3200 | 0.00313 | 2.069e-04 | 6.714e-04 | 199 |
| stoker-wet/smooth | 100 | 0.10000 | 3.351e-03 | 9.406e-03 | 7 |
| stoker-wet/smooth | 200 | 0.05000 | 3.464e-03 | 6.195e-03 | 12 |
| stoker-wet/smooth | 400 | 0.02500 | 2.352e-03 | 4.798e-03 | 25 |
| stoker-wet/smooth | 800 | 0.01250 | 5.096e-04 | 2.188e-03 | 49 |
| stoker-wet/smooth | 1600 | 0.00625 | 5.544e-05 | 4.850e-04 | 99 |
| stoker-wet/smooth | 3200 | 0.00313 | 2.408e-06 | 4.684e-05 | 199 |

### 3.3 门禁裁决

| gate | measured | verdict | why |
|---|---:|---|---|
| `dry_bed_error_is_odx` | 0.0101 | **PASS** | the dry-bed interior error must be an O(dx) self-similar limit (err/dx collapse <= 0.05 on the finest pair) |
| `dry_bed_first_order_from_step` | 1.0001 | **PASS** | the anchored dry-bed case started from the dam discontinuity must be first order (that is the observed behaviour being explained) |
| `dry_bed_operator_recovers_order` | 1.9777 | **PASS** | the same dry-bed solution restarted from continuous data must recover ~2: proof that the loss of order belongs to the data, not the operator |
| `smooth_regime_order` | 2.0353 | **PASS** | Gaussian-bump self-convergence must reproduce the advertised order 2 within 0.3 |
| `wet_bed_error_is_odx` | 0.0061 | **PASS** | the wet-bed interior error must also be an O(dx) self-similar limit |
| `wet_bed_first_order_intrinsic` | 1.0393 | **PASS** | the wet bed owns a travelling bore, so it must stay first order even though the graded window excludes the bore |
| `wet_bed_restart_not_gradable` | 2.7035 | diagnostic | the wet-bed restart sequence must be reported as non-gradable (the closed form already contains the bore; no order is quoted) |

`all_gates_pass = True`。

### 3.4 与算例池条目的绑定

| 项 | 值 |
|---|---|
| 条目声明 `solver.order` | 2 |
| 实测光滑区阶 | 2.04 |
| 判定 | **SUPPORTED** |
| 需要写进条目的适用范围 | the advertised order describes the *smooth-regime* operator (measured 2.04); the anchored dam-break itself converges at ritter-dry/smooth: 1.98, ritter-dry/step: 1.00, stoker-wet/smooth: 2.70, stoker-wet/step: 1.04, because the dry bed starts from a discontinuous dam and the wet bed carries a bore, so no order should be quoted for the graded case |
| 对验收的影响 | the pool grades x_f(t), the still-water step and monotonicity at a fixed grid with 5 %; convergence order is not a gate, so this finding does not change the acceptance verdict |

## ④ 结果图

- `fig_profiles.png`：t*=2 的深度剖面（数值 vs 闭式解），灰带=被分级的扇区内部窗口。
- `fig_convergence.png`：log-log 误差–dx，四个实验 + 光滑隆起自收敛，附 slope 1 / slope 2 参考线。
- `fig_odx_collapse.png`：Ritter 算例的 `(h_num-h_exact)/dx` vs ξ，n=400/800/1600 三条曲线重合
  → 内部误差是**一阶自相似极限**，不是估计子假象。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **干床失去二阶，账在数据上，不在格式上**。同一族解、同一网格、同一终点时刻，唯一改动是
   初始数据由「坝上间断」换成「t*=0.5 连续闭式解」：观测阶 1.00 → 1.98。所以已合并报告里
   「TVD 含间断一阶」这句话在本算例上应当说成**「间断初值 + 干湿界面」的一阶启动瞬态**。
2. **湿床不能用重启正则化**。闭式解在 t*>0 时**本身含涌波**，重启并不移除奇异点；实测重启序列
   **非单调**（-0.05 / 0.56 / 2.21 / 3.20），因此该研究 `gradable = False`，**不引用任何阶**
   （诚实 WITHHELD，不拿 2.70 这种最细两对的均值充数）。湿床的一阶属于**解**，不是启动瞬态。
3. **粗网格的阶不可用**。n=100 那一对的 `err/dx` 塌缩差 1.19–1.46（远大于 5 %），因为下一阶修正
   在小范围内不可忽略；判据只用最细一对。这也是为什么窗内格数要报出来（Stoker 在 n=100 只有 7 格）。
4. **结论不改变 5 % 验收**。池子判的是固定网格上的 x_f(t)、水位阶跃、单调性；收敛阶不是门禁。
   本审计的产出是「`order: 2` 的适用范围」，不是「算例不合格」。
5. **不宣称这点是普适结论**。一阶全局 L1 对含间断的守恒律是经典结果（LeVeque 2002 ch.8）；
   本单做的是**定位到本算例的哪一个奇异点**，并证明算子本身无责。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

本单的分级量**没有物理实验基准**，基准是**闭式解析解**（Ritter/Stoker，以及光滑隆起的自收敛），
因此逐项对照如下：

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---|---|
| 光滑区观测阶 | 设计阶 2 | 2.04 | **否**（偏差 3.5 %） |
| 干床 `err/dx` 塌缩 | 0（一阶自相似极限） | 1.0 % | 否（1.0 % > 3 %），但远低于自定的 5 % 判据；作为「一阶极限」的证据是够的，作为「二阶极限」的证据则明确不成立 |
| 湿床 `err/dx` 塌缩 | 0 | 0.6 % | 是 |
| 干床内部观测阶 | 解析期望：间断初值一阶 | 1.00 | 是（距 1 偏差 0.0 %） |
| 干床重启观测阶 | 解析期望：二阶 | 1.98 | 是（距 2 偏差 2.2 %） |
| 湿床重启阶 | ——（非单调，不可判） | 不引用 | N/A（`gradable = False`，如实 WITHHELD） |

**诚实结论**：锚点验收量（x_f(t)、h2/h1、单调性，<5 %）不受本单影响，仍由
`case_pool_entry.json` 中已分级证据支撑；本单只把「`order: 2`」这一声明从**无证据**变成
**有证据且带适用范围**，并指出已合并报告里对一阶来源的解释需要按 ⑤.1 更正。

---

*本文件数字由 `docs/case-fvm-dambreak-order-audit/make_report.py` 从
`artifacts/shallow-water-order-audit/summary.json` 程序化注入，零手抄。*

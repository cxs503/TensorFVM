# CASE-WOODWARD-COLELLA — 算例池注册与验收契约（机器可检）

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用；`feature_ledger` `open`）
**验收（总账）**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`
**仓**：TensorFVM · **分支**：`agent/m20318` · **求解器：一行未改**
**新增代码**：`src/tensorfvm/compressible/wc_case_pool.py`（算例池入口）+ `tests/test_compressible_wc_case_pool.py`（8 例回归门）

**本单要回答的问题**（先于动手）：前七份报告已经把「算得对不对」写到很细，但总账契约一直只存在于
**散文**里——指标算术在一个模块、数字在一个 json、裁决在一串报告；`<10%` 容差要靠人读表比对。
「算例池」这个字（`算例池|…`）要求的恰恰是**可被机器裁决的入池记录**。所以本单做三件事：

1. 把 `<10%` 契约落成**逐条判据**（量 / 比较符 / 阈值 / 单位 / 是否主判据 / 排除理由），单一 CLI 出
   PASS/FAIL 与退出码，**所有数值由已存工件重算，零手抄**；
2. 用它对**全部已存网格 × 两套参考表**跑一遍，得到「入池最小网格」这一此前无人给出的量；
3. 顺手做了一遍**跨报告账目审计**（agent 技能 adversarial-review）：把每份报告里引用的数字与工件对齐。
   **审计抓到 4 处实质偏差**（见 §⑤-1…4），其中两处改变已发表结论的措辞。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

Woodward & Colella (1984, *JCP* **54**, 115–173) 的一维「两束相互作用爆轰波」：封闭管 x∈[0,1]，
两端反射壁，初始静止 ρ=1、u=0，压力三段台阶；两束激波+接触+稀疏相向传播，t≈0.028 在管中部对撞，
形成强干涉波系（近真空浅槽 + 碰撞区密度尖峰），积分至 t=0.038。**无解析解**，基准是文献数值解。

| 参数 | 值 | 出处 / 落盘 |
|---|---|---|
| 域 | x∈[0,1]，1D（同时有 y 周期导热身的 2D 路径） | W&C 1984 |
| 初始 (ρ,u,p) 左 | (1, 0, 1000) on x≤0.1 | arXiv:1902.07773 §5.1.4；arXiv:2109.14340 Eq.(37) |
| 初始 中 | (1, 0, 0.01) on 0.1<x<0.9 | 同上 |
| 初始 右 | (1, 0, 100) on x≥0.9 | 同上 |
| γ / t_end / 壁面 | 1.4 / 0.038 / 两端反射 | 同上 |
| 格式 | MUSCL-Hancock + MC 限幅 + HLLC + SSP-RK，CFL 0.4 | `compressible/cases.double_blast` |
| 基准曲线（主） | 文献 Fig.21「Reference」= pure RPJ @3200 网格，PDF 矢量路径数字化，y 轴锚在**刻度线** | `reference/reference_density_calibrated.csv` |
| 基准曲线（备，仅溯源） | 同一数字化的**原始**标定（y 轴锚在刻度**标签字形中心**，系统偏低 +0.022017 密度单位） | `reference/reference_density.csv` |
| 比较窗 / 容差 | x∈[0.4, 1.0]（文献绘图窗） / 10 % | 总账 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 2.1 入池契约的结构

`wc_case_pool.CRITERIA` 是唯一真源（6 行，每行含 `quantity / comparator / threshold / unit /
primary / excluded_reason`）。阈值**不新造**：`POOL_TOLERANCE = wc_reference.TOLERANCE_PERCENT = 10.0`，
位置类容差 `POSITION_TOLERANCE = 0.02`（沿用既有比较器），因此池记录不可能与已合并的比较代码漂移。
指标算术**全部复用** `wc_reference.metrics_from_profiles`（同窗口、同阵面检测器），模块只是把它的输出
映射到逐条判据并给出裁决。

### 2.2 三条硬约束（都在测试里）

* **零手抄**：池记录里每个 `value` 都由 `load_profile(nx)`（仓内 `.npz`）+ 参考表重算；
  测试把 nx=800 的峰值误差**钉在** `metrics_calibrated_nx800_hllc.json` 上（1e-9）。
* **门能失败**：`nx=200/400` 必须 FAIL（见 §④ 表），否则「PASS」是构造出来的。
* **容差不放宽**：契约断言容差 == 10.0，且入池网格是**推导**出来的（`minimum_passing_grid()`），不是声明。

### 2.3 工具链

| 文件 | 作用 |
|---|---|
| `src/tensorfvm/compressible/wc_case_pool.py` | 入池入口：`case_pool_entry()` / `evaluate_grid()` / `ladder()` / `minimum_passing_grid()` / `calibration_effect()` / `audit_committed_metrics()`；CLI 三种模式 |
| `tests/test_compressible_wc_case_pool.py` | 8 例回归门（契约完整性 / 参考表一致性 / 工件可复算 / 池网格 PASS / 粗网格 FAIL / 最小网格推导 / 标定效应 / CLI 退出码），5.6 s，纯 numpy，**不起求解器** |
| `docs/woodward-colella-double-blast/audit/wc_pool_figure.py` | 四栏图，只读工件 |
| `docs/woodward-colella-double-blast/case_pool_entry.json` | 入池记录（程序化生成） |

```bash
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool            # 池网格裁决（退出码 0/1）
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool --ladder   # 全阶梯
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool --audit    # 工件 vs 已提交 metrics json
PYTHONPATH=src python -m pytest tests/test_compressible_wc_case_pool.py -q
```

### 2.4 独立复现（不是读旧 json）

为排除「工件是过期的」，本单重跑了一次 nx=400（单核 29.9 s）：**与已存剖面逐点相同**
（`max|Δρ| = 0.0`，峰值均 5.649813015225584）。故 §③ 的所有数字都建立在真实求解器输出上。

## ③ 计算结果（数字表，程序化注入零手抄）

### 3.1 入池记录（`case_pool_entry.json`，池网格 nx=800，主参考=标定表）

| 判据 | 量 | 值 | 比较符 | 阈值 | 主判据 | 通过 |
|---|---|---|---|---|---|---|
| peak_density | 窗内 ρ_max | **2.031 %** | lt | 10 % | ✅ | ✅ |
| peak_position | ρ_max 的 x | **0.002674** | lt | 0.02 | ✅ | ✅ |
| l1_profile | ∫\|Δρ\|dx/∫ρ_ref dx | **2.009 %** | lt | 10 % | ✅ | ✅ |
| window_mass | ∫ρdx over [0.4,1] | **0.0199 %** | lt | 10 % | ✅ | ✅ |
| faces | 四个最陡阵面 max\|Δx\| | **0.001178** | lt | 0.02 | ✅ | ✅ |
| low_plateau | 窗内 ρ_min（浅槽） | 0.179 % | lt | 10 % | ❌ 信息项 | ✅ |
| **裁决** | | **PASS** | | | | **入池最小网格 = 800** |

### 3.2 全阶梯 × 两套参考（`--ladder`，全部由 .npz 重算）

| nx | 峰值误差〔标定〕 | 峰值误差〔原始〕 | L1〔标定〕 | 质量积分〔标定/原始〕 | 浅槽〔标定/原始〕 | 裁决〔标定〕 |
|---|---|---|---|---|---|---|
| 200 | 20.755 % | 20.478 % | 14.211 % | 0.160 / 1.268 % | 0.593 / 13.337 % | **FAIL**（峰值/峰位/L1） |
| 400 | **10.577 %** | **10.265 %** | 6.919 % | 0.062 / 1.367 % | 0.040 / 13.968 % | **FAIL**（仅峰值） |
| 800 | 2.031 % | 1.688 % | 2.009 % | 0.020 / 1.410 % | 0.179 / 14.217 % | PASS |
| 1600 | 2.342 % | 2.699 % | 1.861 % | 0.000 / 1.430 % | 0.305 / 14.361 % | PASS |
| 3200 | 2.998 % | 3.358 % | 3.079 % | 0.009 / 1.440 % | 0.381 / 14.448 % | PASS |

### 3.3 工件审计（`--audit`）

5 个网格 × 2 套参考 = **10 组**，用 `.npz` 原始剖面重算峰值/L1/质量积分，与已提交
`metrics_calibrated_nx*_hllc.json` 逐一比对：**最大绝对差 = 0.0（10/10 复现）**。
这既证明了入池记录没有手抄，也证明了历史上那批 json 确实来自这批剖面。

### 3.4 标定效应的量化（`calibration_effect()`）

质量积分误差：原始参考下 **1.27–1.44 %（平走）** → 标定参考下 **0.00–0.16 %（近零）**；
浅槽误差：**13.3–14.4 %** → **0.04–0.59 %**。两者都**不随网格变化**，是常量基准偏差的指纹。

## ④ 结果图

`docs/woodward-colella-double-blast/audit/fig_wc_pool_acceptance.png`（四栏，全部由工件生成）：

* (a) 池网格密度剖面 vs 两套参考表（可见两参考表之间的常量平移）；
* (b) 峰值误差—Δx 曲线，两套参考 + 10 % 容差线：**nx<800 双侧越线**，nx=800 最优（2.03 %），
  再加密回升（2.34 / 3.00 %）——即基准自身分辨力地板，与 `-nx3200-asymptotic.md` 一致；
* (c) 质量积分与浅槽误差柱状（原始 vs 标定，对数轴）：1.4 % / 13–14 % 的落差整体塌缩；
* (d) nx=800 各判据的「值/阈值」余量（灰=信息项不设门）。

## ⑤ 注意点（含本单审计抓到的 4 处实质偏差）

**审计方法**：把每份报告里引用的数字与 `reference/*.json` + `.npz` 逐一对齐；凡不能被工件复现的数字，
标为缺陷并追到根因。修改一律**加勘误**、不改历史结论的语气。

1. **【已更正】ρ_max 被抄错**。`final-verdict`（2 处）与 `plateau-diagnosis`（1 处）把 nx=800 的
   密度峰值写成 **6.4024**；工件值是 **6.1898**（`wc_blast_profile_nx800.npz` 与
   `metrics_calibrated_nx800_hllc.json` 一致，且本单 nx=400 重跑逐点相同）。
   6.4024 = 6.2961 + 0.1063 恰好是「参考 − 偏差」被写成「参考 + 偏差」，即**符号笔误**；
   百分比 1.69 % 因此是对的，只有绝对值错。物理含义随之改变：TensorFVM 的峰值是**低于**参考
   （数值扩散削峰），而不是高于。已在原报告加勘误段。
2. **【已更正】「nx≥400 全部 PASS」不成立**。`round2-xval` 的这句话在**峰值判据**上被本单证伪：
   nx=400 峰值误差 10.265 %（原始）/ 10.577 %（标定），**两套参考都 >10 %**。
   诚实表述是：**总账 `<10 %` 在 nx≥800 成立，nx=400 差 0.3–0.6 个百分点**（其余判据在 nx=400 已通过）。
   入池最小网格因此**推导为 800**，此前报告默认的 400 是错的。
3. **【解释更正】窗质量积分的 1.4 % 不是「与参考解的系统性差异」**。`final-verdict` 写
   「1.3–1.4 % 平走 ⇒ 已收敛到与参考解的系统性差异」；本单给出更朴素的成因：它就是
   **参考表的常量偏置**（标定后 0.02 %，全阶梯 ≤0.16 %）。原话不算错（确是系统性差异），
   但把它归给「参考解自身精度」会掩盖可修的标定问题。
4. **【诊断边界收窄】浅槽 `low_plateau` 的「病态」判定只在偏置参考下成立**。标定参考下它在**每一档**
   都 <0.6 %。本单**仍不把它设为主判据**，理由改成「小分母量的条件数问题」（阈值不变、面积分与
   质量积分已覆盖该区），而不是「它不达标」。排除理由已写进契约字段，可被测试断言。
5. **基准性质**：RPJ@3200 本身是数值解（非真解）。`<10 %` 的含义是「与已发表数值解一致」。
   数字化只覆盖 x∈[0.4,1.0]；x<0.4 的反射波区无文献参考——本单**没有**也不能补这一块。
6. **峰值误差随加密回升**（2.03 → 2.34 → 3.00 %）不是求解器退步：`-nx3200-asymptotic.md` 已把
   数值解自身误差预算量到 `U_num ≤ 0.14 %`，回升来自与「自身分辨力有限的已发表曲线」逐点比较的地板。
7. **[未做] 二维路径未入池**。2D 一致性锚（y 方向周期/反射壁的 1D 精确重现）仍只在
   `tests/test_compressible_double_blast.py` 里，未进池记录；把它也做成判据需要一次 2D 跑（本机 load15>70，
   容量纪律下未做）。

## ⑥ 误差判定（逐项 vs 基准，池网格 nx=800，主参考=标定表；显式回答「<3%?」）

| 关键量 | TensorFVM | 基准 | 误差 | <3 %? | <10 %? | 纳入主判据 |
|---|---|---|---|---|---|---|
| 密度峰值 ρ_max | **6.1898** | 6.3181 | **2.031 %** | ✅ 是 | ✅ | ✅ |
| 峰值位置 x(ρ_max) | 0.780625 | 0.777951 | 0.002674（绝对，占窗宽 0.45 %） | ✅ 是 | ✅（容差 0.02） | ✅ |
| 剖面 L1 | — | — | **2.009 %** | ✅ 是 | ✅ | ✅ |
| 窗质量积分 ∫ρdx | — | — | **0.0199 %** | ✅ 是 | ✅ | ✅ |
| 四阵面位置 | — | — | **0.001178**（绝对，最大占窗宽 0.20 %） | ✅ 是 | ✅（容差 0.02） | ✅ |
| 近真空浅槽 ρ_min | 0.1787 | 0.1784 | **0.179 %** | ✅ 是 | ✅ | ❌（信息项） |
| （对照）nx=3200 峰值 | 6.5075 | 6.3181 | 2.998 % | ⚠️ **否（≈3.00 %，压线外）** | ✅ | — |

**裁决**：池网格 nx=800 上**五项主判据全部 <3 %**（最大 2.031 %），总账 `<10 %` 满足；
入池最小网格 **800**（nx=400 仅峰值一项 10.58 % 越线，且**两套参考都越线**）。
诚实边界：`<3 %` 在 nx=3200 **不再成立**（2.998 %，基准自身地板），不得把 800 档的 3 % 结论外推到更细网格。

### 复现命令与证据指针

```bash
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool --nx 800 --json docs/woodward-colella-double-blast/case_pool_entry.json
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool --ladder
PYTHONPATH=src python -m tensorfvm.compressible.wc_case_pool --audit
PYTHONPATH=src python -m pytest tests/test_compressible_wc_case_pool.py -q     # 8 passed, 5.6 s
PYTHONPATH=src python docs/woodward-colella-double-blast/audit/wc_pool_figure.py
```

* 入池记录：`docs/woodward-colella-double-blast/case_pool_entry.json`
* 图：`docs/woodward-colella-double-blast/audit/fig_wc_pool_acceptance.png`
* 剖面工件：`artifacts/wc_blast_profile_nx{400,800}.npz`、`ladder/data/ladder_profile_nx{200,1600,3200}_hllc.npz`
* 参考表：`reference/reference_density_calibrated.csv`（主）、`reference_density.csv`（溯源）
* 前序报告：`-final-verdict.md`、`-calibration-bias.md`、`-nx3200-asymptotic.md`、`-round2-xval.md`、`-plateau-diagnosis.md`
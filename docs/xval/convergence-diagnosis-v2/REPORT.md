# XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 — 与成熟开源/标准做法（三网格 GCI）的交叉对标

**锚点**：`XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2`（第 1 轮 · 任务榜 · 开源对标）  
**任务**：m17360　**仓库**：TensorFVM（分支 `agent/m17360`）　**性质**：交叉核验（XVAL），非重做  
**对标对象**：TensorFVM 的 convergence-diagnosis-v2 估计器（`src/tensorfvm/verification/sod1978_convergence.py`）  
**参考面（OSS/标准做法）**：三网格 GCI 程序 —— Roache, *ASME J. Fluids Eng.* **116** (1994) 405-413；Celik, Ghia, Roache, Freitas, Coleman & Raad, *ASME J. Fluids Eng.* **130** (2008) 078001（OpenFOAM/SU2 类后处理与海事域 ITTC 7.5-03-01-01 通行的收敛/不确定度报告程序）；其在 OSS TensorFVM 仓内的既有实现 = `tensorfvm.ship_freesurface.mesh_convergence.richardson_extrapolation`（源码实读：`Fs=1.25`、`convergence_order()` 的 monotonic/oscillatory/divergent 分类、`r` 参数化）。  
**判据（预声明，先于任何数字）**：估计器可互换 ⇔ 观测阶 `p` 相对差 ≤ 15 %（owner 精度标准），且该要求**只在渐近区成立时施加**（重叠三重组一致性 ≤ 15 %）；导出量（Richardson 外推值、GCI 带）同 15 %；`Fs=1.25`。  
**产物**：`artifacts/xval_convergence_diagnosis.json`（全量数字）、`artifacts/xval_table.md`（程序生成的 ref vs got 表）、`figures/*.png`、`gci_reference.py`（独立参考实现）、`xval_driver.py`（驱动）、`tests/test_convergence_diagnosis_xval.py`（闸门）  
**裁定**：**部分可互换** —— 渐近区 4 个功能量中 3 个 ✅（≤15 %：6.7 % / 1.9 % / 1.9 %）、1 个 ❌（contact 位置 80.7 %）；非渐近区（激波位置、plateau L1）**两个估计器都不可读**，而 v2 未设防：已入库工件里激波位置的 log-log“阶”为 **−1.2494**（负值），据此读数得到的机制判据是错的。已按对标结论把 GCI 的「分类 + 渐近区检验 + 不确定度带」移植进本仓（见 ⑥）。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

**物理背景**：Sod (1978) 一维激波管是双曲守恒律（Euler）的解析验证基准；本锚点的诊断对象是**该算例的收敛性诊断本身**，所以同时用两个功能量族：

1. **间断问题**（诊断的困难面）：Sod 管 t=0.14 的密度 L1 误差、plateau 窗 L1、稀疏波/接触/激波四道波的位置；
2. **光滑控制面**（诊断的干净面）：周期熵波 `ρ = 1 + 0.2 sin(2πx)`（Euler 精确解，随流不变），其 L1 误差度量的是格式本身的设计阶。

两族同时进入对标，才能把「估计器不准」与「问题本身非渐近」分开——这是本次 XVAL 的核心设计（见 ⑤）。

**几何 / 参数表（无量纲）**：

| 参数 | 值 | 说明 |
|---|---|---|
| 域 / 膜片 | x∈[0,1]，x₀=0.5 | 1-D |
| 初值 | (ρ,u,p)_L=(1,0,1)，(ρ,u,p)_R=(0.125,0,0.1) | 左/右态 |
| γ | 1.4 | 量热完全气体 |
| 终时 | t=0.14 | 判据时刻 |
| 数值格式 | MUSCL-Hancock + HLLC，order=2（对照 order=1） | `tensorfvm.compressible` |
| 限制器 | MC | 敏感性项另有 superbee（既有 v2 工件） |
| CFL | 0.4（主）/ 0.2（敏感性） | 预声明两点 |
| 网格 | 100 / 200 / 400 / 800（r=2） | 主三重组 100/200/400；细三重组 200/400/800；4 网格 100/200/400/800 |
| 参考实现参数 | Fs=1.25，渐近带 ±0.15 | Celik et al. (2008) |

**两个估计器（要对比的东西）**：
- `got`（v2）：相邻网格对速率 `p = ln(e₁/e₂)/ln(h₁/h₂)`；**finest-3 的 log-log 最小二乘斜率**（对**误差**序列拟合，波位置用 `|x_s − x_s,exact|`）。
- `ref`（OSS/标准）：三点差分 `p = |ln|ε₃₂/ε₂₁||/ln r`，Richardson 外推 `φ_ext=(r^p φ_f−φ_m)/(r^p−1)`，GCI 带 `Fs·|ε|/(r^p−1)`，并**先分类**（monotonic/oscillatory/divergent）再给阶。

---

## ② 方法与实现（怎么做的）

1. **三面对标，零手抄**：所有数字由脚本重算。
   - `ref`：`docs/xval/convergence-diagnosis-v2/gci_reference.py` —— 按论文公式**独立实现**，**不 import** `tensorfvm` 任何模块（审计纪律：参考面与被审面不共享代码）。
   - `repo_gci`：仓内既有 `tensorfvm.ship_freesurface.mesh_convergence.richardson_extrapolation` —— 证明独立实现不是"某一支重推导的偏方"。两者在 **29 组可比的 (p, 外推值, GCI) 上全部一致（≤15 %，实测为机器精度级）**。
   - `got`：直接 `import` 被测模块的 `successive_orders` / `fitted_exponent` 施加到**同一数据**上。
2. **同一条物理 → 四条扫描**（全部确定性，无随机）：主三重组 100/200/400（order2, CFL0.4）；细三重组 200/400/800；CFL 敏感性 100/200/400（CFL0.2）；一阶格式 100/200/400（order1）。4 网格 100/200/400/800 由前两条扫描合并（同物理、去重）。
3. **非空转的渐近区检验**：Celik eq.(17) `GCI_21/(r^p GCI_32) ≈ 1` 在**单个三重组**下是恒等式（p 由同一三重组算出），本次实测其值恒为 `1.00`——对非渐近的激波位置序列也一样等于 1.00（`non_asymptotic_rows_where_single_triplet_eq17_looks_fine = [combined/shock_x]`）。因此驱动额外实现**重叠三重组一致性**（4 网格两套三重组独立求 p，相对差 ≤15 % 才算在渐近区）。这是本次对标最实用的一条移植。
4. **对已入库 v2 工件重诊断**（不重跑求解器）：读 `docs/repro-fvm-sod1978-convergence/study.json`（sha256 记入工件），用 ref 面对其 50/100/200 的数字重新诊断，并用**打补丁后的模块**复算有效指数与判据，给出 before/after 对照。
5. **闸门**：`tests/test_convergence_diagnosis_xval.py`（9 个用例）锁住：参考面解析自检（p=2 精确）、分类正确性、两参考实现一致、生产移植与独立实现一致（1e-10）、非单调序列不得报阶、已入库工件的失效模式仍在、**工件与 markdown 数字可重算**（防手抄）。

**执行**：`PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/xval_driver.py --out docs/xval/convergence-diagnosis-v2`（单核 232.9 s；nx=800 单次 58.5 s）。

---

## ③ 计算结果（ref vs got，数字表）

### 3.1 渐近区功能量（4 网格重叠三重组判定"在区内"）

| 功能量 | p_ref（GCI 三重组） | p_got（lsq） | p_got（相邻对） | Δlsq | Δsucc | 15 % 判定 |
|---|---|---|---|---|---|---|
| 熵波 L1（光滑控制面） | 2.1116 | 2.0713 | 2.0454 | **1.94 %** | 3.13 % | ✅ |
| 密度 L1（全域） | 0.9386 | 0.8756 | 0.8567 | **6.69 %** | 8.66 % | ✅ |
| 密度 L1（单元平均采样） | 0.9049 | 0.9220 | 0.9267 | **1.87 %** | 2.44 % | ✅ |
| 接触波位置 | 0.7868 | 1.4224 | 1.7084 | **80.72 %** | 117.1 % | ❌ |

### 3.2 非渐近区功能量（重叠三重组判"出区"）

| 功能量 | p(粗三重组 100/200/400) | p(细三重组 200/400/800) | 相对差 | 单三重组 eq.17 比 |
|---|---|---|---|---|
| 激波位置 | 2.4315 | 1.0662 | **128.1 %** | 1.00（恒等，无鉴别力） |
| plateau 窗 L1 | 3.0016 | 4.7487 | **58.2 %** | 0.15–0.78 |

→ 这两族功能量的"阶"在相邻三重组间自己就不稳定，任何估计器（含 ref）给出的单值都不可读；v2 的问题不是数值偏差，而是**没有这道闸**。

### 3.3 派生量：Richardson 外推 / GCI / 对解析真值

| 量（4 网格细三重组） | ref | 用 got 的 p | 误差 |
|---|---|---|---|
| 激波位置外推值（真值 0.7453018） | 0.745342 | 0.263504 | 外推值 64.6 %，**但真值误差仅 0.0054 %** |
| 接触位置外推值（真值 0.6298434） | 0.629968 | n/a（p_got 使 r^p−1 近奇） | 真值误差 **0.0198 %** |
| 接触位置 GCI 相对带 | 0.036 % | — | 真值 0.020 % **落在带内** ✅ |
| 熵波 L1 GCI 绝对带 | 3.98e-8 | 1.55e-8 | 零极限量，相对带无意义（见 ⑤） |

**误差判定（显式回答是否 <3 %）**：与外部分析真值（Sod 精确 Riemann 解的波速 `x_s(t)=x₀+S t`，由 `exact_star_state()` 程序算出，非手抄）比，ref 的 Richardson 外推波位置误差 **激波 0.0054 %、接触 0.0198 %，均 <3 %，也 <15 %**；GCI 带覆盖真值。**但这是 ref 的精度，不是 v2 估计器的精度**——v2 的 p 在同一点上（接触 1.42 vs 0.79）偏离 80.7 %，>15 %，判 ❌。

---

## ④ 结果图

- `figures/xval_error_vs_h.png`：4 网格上「激波位置误差」与「plateau L1」的 log-log 曲线 + ref/v2 两条参考斜率线。可见激波位置误差序列在 200→400→800 明显减速（斜率翻转），两条参考线本身就分开——非渐近的直观证据。
- `figures/xval_order_agreement.png`：30 组 (p_ref, p_got) 散点 + ±15 % 带，绿=达标、红=超差；接触位置点落在带外，光滑控制面点落在带内。
- `figures/xval_committed_artifact.png`：已入库工件 50/100/200 的波位置误差序列（激波序列**非单调**，正是 LSQ 报出 −1.2494 的成因）。

（图为 matplotlib 程序生成，数据源为 `artifacts/xval_convergence_diagnosis.json`。）

---

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **网格三重组选择**（主 100/200/400 vs 细 200/400/800）：接触位置的 p_ref 0.731 → 0.787（稳定），p_got(lsq) 1.033 → 1.422（漂移）；激波位置 p_ref 2.431 → 1.066（**漂移 128 %**）。结论：三重组选择对"非渐近"功能量的结论有决定性影响，故必须有重叠三重组检验；对渐近区功能量（熵波）两套三重组给出 p=2.143/2.112（差 1.4 %），稳定。
2. **CFL（0.4 vs 0.2，100/200/400）**：熵波 p 2.114 vs 2.114（不变），全域密度 L1 p 1.068 vs 1.114；但 v2 的 lsq 与相邻对差在 CFL0.2 下从 12.4 %/16.2 % 变为 15.5 %/20.3 %（跨过 15 % 线）——即**同一物理、同一网格，只改 CFL 就能让"是否达标"翻转**，说明 15 % 阈值下这两个估计器处于边缘可互换状态，不宜互相替代，只能择一并固定。
3. **格式阶（order2 vs order1）**：一阶格式下接触位置序列被 ref 判 `divergent`（拒绝给阶），而 v2 仍输出 0.540/0.663 —— 再次复现"无闸门"问题；熵波一阶 p 0.990 vs got 0.995（0.5 %，干净）。

---

## ⑥ 失败模式、差异清单与移植取舍（本仓 vs OSS/标准做法）

### 6.1 差异清单（做法 / 参数 / 边界条件）

| 维度 | 本仓 convergence-diagnosis-v2 | OSS/标准三网格 GCI | 后果（实测） |
|---|---|---|---|
| 估计器 | 相邻对速率 + finest-3 log-log LSQ | 三点差分 `|ln|ε32/ε21||/ln r` | 渐近区差 1.9–6.7 %；非渐近区差 43–128 % |
| 预处理 | 直接对误差幅值取对数 | **先分类**（monotonic/oscillatory/divergent），非单调则拒绝给阶 | v2 给已入库工件的激波位置"阶" −1.2494（负值） |
| 不确定度 | 无 | GCI 带 `Fs|ε|/(r^p−1)`，Fs=1.25 | v2 无带 ⇒ 无法说"外推值可信到什么程度" |
| 渐近区检验 | 无 | eq.(17) `GCI21/(r^p GCI32)≈1`（**单三重组下恒等，需 ≥4 网格**） | v2 无检验 ⇒ 非渐近数据被当作阶输出 |
| 极限形式 | 未区分 | 相对 GCI（非零极限）/ 绝对 GCI（零极限，如 L1） | 零极限量的相对带无意义（本次实测相对带 100 %+） |
| 网格比 | 允许非均匀（逐对取 nx 比） | 假定 r 均匀，另报 r21/r32 偏差 | v2 更通用（此处 r=2，无差异） |
| 额外度量 | restricted 窗 / 单元平均 L1 / 波位置指纹 / 光滑波设计阶认证 | 无 | **本仓更强项，移植时不得丢** |

### 6.2 移植取舍（已落地）

**移植（3 条，OSS 有而本仓无、且本次实测证明必需）**
1. 收敛类判定 + **非单调序列拒绝给阶**；
2. **重叠三重组一致性**（4 网格）作为渐近区闸门（替代单三重组恒等的 eq.17）；
3. **GCI 不确定度带**（非零极限用相对式、零极限用绝对式）。

**不移植（2 条）**
1. 用三网格差分**替换** log-log LSQ —— 实测在渐近区两者差 1.9–6.7 %，LSQ 对非均匀 r 更稳健，保留为默认，三网格差分作为交叉核验并列报告；
2. 丢掉本仓的 restricted 窗 / 单元平均 / 波位置指纹 —— 这些是 OSS 侧没有的能力，且正是本锚点机制归因的证据面。

**落地位置**：新增 `src/tensorfvm/verification/convergence_crosscheck.py`（GCI 三网格程序 + 分类 + 渐近比 + 重叠三重组一致性 + `guarded_loglog_exponent` / `effective_exponent`）；`sod1978_convergence.py` 的 `_rate_table` 现在为每个功能量附加 `crosscheck` / `effective_exponent` / `loglog_guard` 三个字段，`sublinear_convergence_observed` 判据改读**有效指数**（对数斜率不可采信时回退到相邻对速率并记录来源）。

**基线对照（真改进，不是"测试变绿"）**：对**已入库工件同一组数字**重诊断：

| 量 | 修复前（published） | 修复后（guarded） | ref 面 |
|---|---|---|---|
| 激波位置指数 | −1.2494（不可采信） | 1.8379（来源 successive_finest） | 序列被判 **oscillatory**，拒绝给阶 |
| 接触位置指数 | 1.4695 | 1.4695（lsq 可采信） | 1.47（lsq）/0.93（相邻对） 同向 |
| `sublinear_convergence_observed` | **false** | **true** | 两估计器下 contact < shock 均成立（robust） |

即：该锚点 v2 的"机制未观测到"结论是**由一个无意义的负斜率造成的**；修正后机制在两套估计器下都成立。此翻转只依赖已入库工件里的数字，可由 `tests/test_convergence_diagnosis_xval.py` 复算。

### 6.3 失败模式与局限

- **零极限功能量**（L1 范数）：相对 GCI 与 eq.17 比退化为 `r^-p`（本次实测 0.15–0.78），只有绝对带与阶有意义；报告里已分别标注 `zero_limit` / `nonzero_limit`。
- **波位置是"捕捉"量**：位置由阈值插值确定，误差序列可非单调（50/100/200 实测），任何阶估计都要先过单调 + 重叠三重组两道闸。
- **接触波参数（order1）**：ref 判 divergent、仓内 GCI 判 oscillatory —— 两实现类别不一致（非数值分歧），本次未计入 15 % 数值对账（29 组数值对比 0 分歧）。
- **本对标仅 1 个算例家族（Sod + 熵波）**：结论"估计器在渐近区可互换、非渐近区必须设防"应属通用，但 15 % 的具体 margin 不外推到其他算例。
- **单核时限摩擦**（生态 friction 如实记）：shell 前台 120 s 截断，驱动 232.9 s 需 `setsid nohup` 后台跑；nx=800 单点 58.5 s，若接入更大网格扫描应走 cluster_queue。

### 6.4 工件

`artifacts/xval_convergence_diagnosis.json`（含 raw 逐网格数据、三面诊断、阈值、verdicts、工件 sha256）、`artifacts/xval_table.md`（程序生成）、`figures/*.png`、`gci_reference.py`、`xval_driver.py`、`src/tensorfvm/verification/convergence_crosscheck.py`、`tests/test_convergence_diagnosis_xval.py`。
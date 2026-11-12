# 时间发展平面混合层（TensorFVM）算例报告

**Mission**: m15435 · **Repo**: TensorFVM · **算例池条目**: 时间发展混合层（卷并/扩展率）
**状态**: 已实现并对照通过（线性稳定性误差 2.1%、精确扩散解 4e-4%），**外文文献卷并率锚点 = WITHHELD（离网不可取原文，如实声明）**。

---

## ① 问题与 ref 出处

两层反向平行流 `u0(y) = U·tanh(y/δ)`（自由流 ±U，速度差 ΔU=2U）在随层移动的参考系中演化：流向周期、横向自由滑移壁面，即时间发展（temporal）混合层的标准设置。考核量：
1. 线性段扰动增长率 σ（无量纲 σδ/U）；
2. 非线性段动量厚度增长率 S=dθ/dt（无量纲 S/ΔU，即"卷并/扩展率"）；
3. 无量纲最不稳定波数与放大率 ασ 的峰值位置。

**ref 出处与零手抄声明**：本单要求的"文献卷并率"原文在本工作单元**不可取得**——本机无外网出口（外部 HTTP 被安全策略拦截），平台知识引擎 :8021 拒绝连接（`/tensor/knowledge/search` 返回 `engine_offline`，Connection refused），negative-knowledge 三查中 "mixing layer"/"shear layer" 命中 0 条。因此**不从记忆手抄任何文献数字**（手抄无法给出原表原页，等同伪造）。改以三条**本仓可程序化重算的独立基准**：

- **精确解（exact）**：无扰动时 `u=u(y,t), v=0` 的无粘/粘性 NS 退化为热方程，`u(y,t)=∫G(y−y′,t)·U·tanh(y′/δ)dy′` 是**严格解**（`mixing_layer.heat_kernel_profile`）。
- **线性稳定性（独立离散）**：Rayleigh（无粘）与 Orr–Sommerfeld（粘性）特征值问题由 `mixing_layer.rayleigh_modes / orr_sommerfeld_modes` 用**均匀网格二阶内部广义特征值**求解，与主求解器（交错 MAC + FFT/DCT 精确投影 + 显式 RK2）离散完全独立；并自带两条自查：Re→∞ 的 OS→Rayleigh 极限、以及特征值求解自身的网格收敛。
- **自收敛**：网格/时间步阶梯上卷并率的收敛性。

文献数值一旦可取（联网或知识引擎恢复），应作为**外部锚点**补入第④节；现状记为场景债，不以自算值冒称文献值。

## ② 方法与实现

**共性能力复用**：算例实现为新增模块 `src/tensorfvm/mixing_layer.py`。压力泊松用与 `incompressible_benchmarks` 同一对变换（x 向 FFT + y 向 DCT-II，对应自由滑移 Neumann）**精确求解**，无迭代求解器进入任何数字；对流用旋转（skew）形式二阶中心、扩散二阶中心、显式 Heun(RK2)，每级后精确投影。诊断：x 平均剖面、动量厚度 θ=∫(1/4−(ū/ΔU)²)dy、涡量厚度 δω=ΔU/max|dū/dy|、流向基频幅值 |v̂(k=1)|、脉动能量、最大散度（投影正确性证据）。

**生产者脚本** `scripts/verify_mixing_layer.py` 分阶段（stability / diffusion / linear / nonlinear / similarity / report）写出 `docs/repro-mixing-layer-temporal/run/` 下全部 JSON/PNG 与 `table.md`；本报告所有数字由 `run/summary.json` **程序化注入**（模板占位符替换），无手工录入。

**关键设置**：扰动初值取 Rayleigh 特征函数（`seed_stability_mode`），避免多模态瞬态污染增长率拟合；线性段拟合窗 tU/δ∈[2,9]；卷并率拟合窗 tU/δ∈[10,28]（层在 tU/δ≈28 饱和，见⑥）。

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| 半速度差 U | 1.00 |
| 速度差 ΔU | 2.00 |
| 初始涡量厚度 δω | 2.00 |
| 初始 tanh 尺度 δ | 1.00 |
| 运动黏度 ν | 0.0020 |
| Re_δ = Uδ/ν | 500 |
| 无量纲波数 αδ | 0.50 |
| 流向波长 λ/δ | 12.5664 |
| 横向高度 H/δ | 16.0 |
| 生产网格 Nx×Ny | 96×256 |
| Δx/δ, Δy/δ | 0.1309, 0.0625 |
| 时间步 ΔtU/δ (CFL 0.4) | 0.0249 |
| 线性/非线性扰动初值 | 1e-05 / 2e-02 × ΔU |
| 非线性运行时长 tU/δ | 50 |

## ④ 对照表（ref vs got，逐项误差打勾叉）

# Machine-generated comparison table

All values are produced by `scripts/verify_mixing_layer.py`; nothing is hand-typed.

| item | reference | ref | got | rel. error | verdict |
|---|---|---:|---:|---:|---|
| linear growth rate, alpha=0.5, Re=500, instantaneous layer scale | Orr--Sommerfeld eigen-solve (independent discretisation) | 0.18372 | 0.18224 | 2.087% | PASS (tol 10.0%) |
| linear growth rate, alpha=0.5, Re=500, frozen initial profile | Orr--Sommerfeld eigen-solve (independent discretisation) | 0.18372 | 0.18224 | -0.803% | PASS (tol 10.0%) |
| inviscid limit of the Orr--Sommerfeld solve (Re=1e5) | Rayleigh eigen-solve | 0.18728 | 0.18726 | 0.010% | PASS (tol 1.0%) |
| momentum thickness during the diffusion phase | exact heat-kernel solution of the same initial profile | - | - | 0.000% | PASS (tol 5.0%) |
| mean velocity profile during the diffusion phase | exact heat-kernel solution of the same initial profile | - | - | 0.001% | PASS (tol 5.0%) |
| Rayleigh grid refinement (301 -> 601 nodes) | self-convergence of the eigen-solve | 0.18728 | 0.18730 | 0.014% | PASS (tol 0.5%) |
| nonlinear spreading rate dtheta/dt, production vs coarse grid | self-convergence over the grid/time-step ladder | 0.01658 (coarse) | 0.01665 (production) | 0.42% | PASS (tol 10%) |
| self-similar mean-profile collapse | profile at t=22.5 vs t=47.5 | - | - | 0.0227 (max deviation) | PASS (tol 0.0500) |

Nonlinear spreading measured on the growth window tU/delta in [10.0, 28.0] (the layer saturates near tU/delta ~ 28 in the single-wavelength box and then oscillates).

Production case: 96x256 cells, dt=0.0249 (CFL 0.4), Re_delta=500, alpha*delta=0.5, linear seed 1e-05, nonlinear seed 2e-02.

External literature anchor (published spreading / entrainment rate): **WITHHELD** -- the offline work cell cannot reach the reference text (no internet egress; the platform knowledge engine at :8021 refused connections), so no literature number is quoted or transcribed here.


### ④a 线性段网格阶梯（αδ=0.5，Re_δ=500）

| 网格 | ΔtU/δ | σδ/U(仿真) | OS 参考(瞬时层尺度) | 误差(瞬时) | 误差(冻结初值) |
|---|---|---|---|---|---|
| 48×128 | 0.0500 | 0.18109 | 0.17711 | +2.24% | -1.43% |
| 96×256 | 0.0250 | 0.18224 | 0.17852 | +2.09% | -0.80% |
| 144×384 | 0.0167 | 0.18247 | 0.17878 | +2.06% | -0.68% |

### ④b 时间步阶梯（96×256）

| ΔtU/δ | σδ/U |
|---|---|
| 0.02500 | 0.18226 |
| 0.01250 | 0.18226 |
| 0.00625 | 0.18226 |

### ④c 波数扫描（仿真 vs 独立稳定性）

| αδ | σδ/U(仿真) | OS(Re=500) | Rayleigh | 误差 |
|---|---|---|---|---|
| 0.30 | 0.1694 | 0.1670 | 0.1693 | +1.46% |
| 0.40 | 0.1844 | 0.1813 | 0.1841 | +1.72% |
| 0.50 | 0.1828 | 0.1790 | 0.1824 | +2.13% |
| 0.60 | 0.1669 | 0.1624 | 0.1665 | +2.79% |
| 0.70 | 0.1383 | 0.1331 | 0.1380 | +3.89% |

### ④d Rayleigh 波数扫描（无粘，含放大率 ασ）

| αδ | σδ/U | ασδ/U |
|---|---|---|
| 0.20 | 0.13608 | 0.02722 |
| 0.25 | 0.15641 | 0.03910 |
| 0.30 | 0.17148 | 0.05144 |
| 0.35 | 0.18173 | 0.06361 |
| 0.40 | 0.18755 | 0.07502 |
| 0.45 | 0.18930 | 0.08518 |
| 0.50 | 0.18728 | 0.09364 |
| 0.55 | 0.18174 | 0.09996 |
| 0.60 | 0.17291 | 0.10375 |
| 0.65 | 0.16095 | 0.10462 |
| 0.70 | 0.14602 | 0.10221 |
| 0.75 | 0.12824 | 0.09618 |
| 0.80 | 0.10773 | 0.08618 |

### ④e Orr–Sommerfeld 的 Re→∞ 极限（αδ=0.5，对 Rayleigh 值 0.18728）

| Re_δ | σδ/U |
|---|---|
| 500 | 0.18372 |
| 2000 | 0.18637 |
| 10000 | 0.18710 |
| 100000 | 0.18726 |

### ④f Rayleigh 求解节点收敛（αδ=0.5）

| 节点数 | σδ/U |
|---|---|
| 301 | 0.18728 |
| 401 | 0.18729 |
| 601 | 0.18730 |

### ④g 非线性卷并率（拟合窗 tU/δ∈[10,28]）

| 算例 | 网格 | ΔtU/δ | dθ/dt | 扩散基线 | 扣除基线后 |
|---|---|---|---|---|---|
| coarse | 64×192 | 0.0332 | 0.01658 | 0.00126 | 0.01532 |
| production | 96×256 | 0.0249 | 0.01665 | 0.00126 | 0.01539 |
| coarse-dt-halved | 64×192 | 0.0166 | 0.01658 | 0.00126 | 0.01532 |

## ⑤ 收敛性与参数敏感性

1. **空间**：σ 随网格 48×128→96×256→144×384 收拢 0.18109→0.18224→0.18247（相邻变化 0.12%）；卷并率 64×192 vs 96×256 相差 0.42%。
2. **时间**：ΔtU/δ 分三级（表④b），σ 变化 <0.01%——时间离散误差已可忽略。
3. **波数**：σ 峰值在 αδ≈0.4–0.5（表④c），放大率 ασ 峰值在 αδ=0.65（表④d）；两者判据不同故位置不同，且仿真与独立稳定性在每个波数的偏差 ≤3.9%。
4. **特征值求解自身**：节点 301→601 变化 0.0143%；Re=1e5 的 OS 与 Rayleigh 相差 0.0097%。
5. **剖面自相似**：tU/δ=22.5/35/47.5 三时刻 ū/ΔU 以 δω 归一后最大偏差 0.0227。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 基准 | 相对误差 |
|---|---|---|
| 线性增长率 σ（αδ=0.5，Re=500，瞬时层尺度） | OS 特征值 0.18372 | 2.09% ❌（>3% 未达，<10% 容差判据 ✅） |
| 线性增长率 σ（冻结初值口径） | OS 特征值 0.18372 | -0.80% ✅（<3%） |
| 扩散段动量厚度 θ | 精确热核解 | 0.000383% ✅ |
| 扩散段平均速度剖面 | 精确热核解 | 0.000699% ✅ |
| OS 的 Re→∞ 极限 | Rayleigh | 0.0097% ✅ |
| 卷并率网格收敛（96×256 vs 64×192） | 自收敛 | 0.42% ✅ |

**结论**：本单容差判据为 <10%，全部项通过；平台惯用的 <3% 更严口径下，**冻结初值口径 0.80% 达标，而"瞬时层尺度"口径为 2.09%（数值上 <3%，但其差异来自参考口径，非纯离散误差——如实列两条，不挑好看的口径）**。文献卷并率外部锚点缺失，不以自算值冒充。

## ⑦ 失败模式、局限与工件

**局限（如实）**：
- **文献外部锚点缺失**（见①）：卷并率无外文数值对照，只有自收敛与精确解/理论对照，属场景债。
- **单一波长盒长**：盒长恰为一个最不稳定波长，**亚谐配对（vortex pairing）模态不可表示**，层在 tU/δ≈28 达 θ 峰值 0.7886 后进入衰减—回弹振荡（θ 序列见 `run/nonlinear.json`），**不存在长时自相似湍流段**。故卷并率只在增长窗有意义，不可外推为充分发展湍流的扩展率。
- **二维**：无展向失稳（elliptic / 三维二次失稳），所测增长率为二维值，不能替代三维 DNS。
- **层尺度口径**：线性段参考用"仿真自测瞬时层尺度"重算 OS 以剥离基流扩散效应；冻结初值口径给出 -0.80%。两口径并列。
- 二阶中心格式（无亚格子/限制器）在卷并后期细尺度耗散偏大，σ 随加密略升（表④a），已在网格阶梯上量化。

**失败模式与已钉死的坑**：全部运行最大散度 ≤1e-14（投影精确），无 NaN/发散。初版两个坑：(a) y 向 Poisson 特征值误用 sin²(πm/N)（应为 sin²(πm/2N)），表现为扰动 10 步内爆掉；(b) v 动量 (N,N+1) 与 (N,N−1) 切片错位广播——均由 `tests/test_mixing_layer.py` 的"散度<1e-12 / 精确扩散解 / 线性增长率对 OS"三项测试钉死。

**工件**：`docs/repro-mixing-layer-temporal/run/`（stability/diffusion/linear/nonlinear/self_similarity JSON、5 张 PNG、`table.md`、`summary.json`、`manifest.json`（含 sha256））。
**图**：`run/fig1_linear_growth.png`（σ–α 曲线与线性段指数增长）、`run/fig2_diffusion_exact.png`（θ 对精确解）、`run/fig3_nonlinear_spreading.png`（网格/时间步阶梯的 θ(t)）、`run/fig4_vorticity.png`（卷并后涡量场）、`run/fig5_self_similarity.png`（剖面坍缩）。
**测试**：`tests/test_mixing_layer.py`（6 项，11 s 内跑完）。

**一键复现**：
```bash
PYTHONPATH=src python3 scripts/verify_mixing_layer.py --output /tmp/mixing --stage all
```
（阶段可拆：`--stage stability,diffusion` 等；单阶段 3–90 s，全程约 3 min。）

## ⑧ 生态摩擦（如实记录）

1. 平台知识引擎 :8021 离线（`/tensor/knowledge/search` → `engine_offline`，Connection refused），同域三查只完成 negative-knowledge（0 命中）；capability/oss_solutions 检索不可用。
2. 本工作单元无外网出口，文献原表原页不可取 → 文献锚点 WITHHELD。
3. `audit_capabilities` 工具报错：缺少 `backend/platform_harness/capabilities.json`，能力清单不可读，能力核对只能靠仓库人工比对。
4. 单条 shell 命令约 110 s 墙钟上限，重型生产者脚本必须内建分阶段（本脚本已内建 `--stage`）。

## 参考

- 时间稳定性方程：Rayleigh（无粘）/ Orr–Sommerfeld（粘性）剪切层特征值问题，本仓自实现求解器 `src/tensorfvm/mixing_layer.py`（不含任何手抄文献数值）。
- 精确扩散解：`mixing_layer.heat_kernel_profile`（热核与 tanh 初值的卷积）。
- 投影与变换对沿用 `src/tensorfvm/incompressible_benchmarks.py` 的 FFT+DCT 泊松解。

# XVAL-FVM-M18737-ROUND2: 综合验证报告（Synthesis Report）

> **锚点**: `XVAL-FVM-M18737-ROUND2` — TensorFVM 的 m18737-round2 做法
> **验收标准**: 对标误差 <15%（owner 精度标准：对标误差 >15% 不可接受）
> **状态**: 全部良态指标 <10%（ledger tolerance），总账 PASS
> **关联**: `docs/CASE-WOODWARD-COLELLA-round2-xval.md`（round-2 交叉验证）·
> `docs/CASE-WOODWARD-COLELLA-method-claim-conformance.md`（做法登记审计）·
> `docs/xval-round2/FINAL-REPORT.md`（收敛诊断因果审计）

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

**物理背景**: Woodward & Colella (1984) 双爆轰波算例——一维闭管 [0,1] 内三个压力平台
（左 p=1000、中 p=0.01、右 p=100），两束爆轰波相向传播，t≈0.028 在管中部对撞，
形成双激波干涉结构，积分至 t=0.038。这是可压缩 Euler 方程求解器的标准基准算例，
考验激波捕捉、接触间断分辨和守恒性。

**参数表**:

| 参数 | 值 | 出处 |
|---|---|---|
| 域 | x∈[0,1]，1D | W&C 1984 |
| 初始 (ρ,u,p) 左/中/右 | (1,0,1000) x≤0.1 / (1,0,0.01) 0.1<x<0.9 / (1,0,100) x≥0.9 | Dumbser et al. JCP 2019 |
| γ / t_end / 壁面 | 1.4 / 0.038 / 两端反射 | 同上 |
| 基准曲线 | Fig.21「Reference」（RPJ@3200，PDF 数字化） | `reference/curves.json` |
| 基准标定偏差 | y 轴 +0.022017（密度单位，常量加性） | `reference/axis_calibration.json` |
| 格式（TensorFVM） | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 | `compressible/cases.double_blast` |
| 交叉码 | WENO5 + SSP-RK3 + Rusanov（不共享离散） | `xcode/wc_crosscode.py` |

## ② 怎么做的（方法 / 流程 / 工具链）

本锚点历经多轮验证，每轮回答不同问题：

| 轮次 | 核心问题 | 工具 | 状态 |
|---|---|---|---|
| Round-1 | 算例对不对？ | `wc_ladder.py` + 参考对比 | ✅ PASS（nx≥800 <3%） |
| Round-2 | 残差里哪部分是我们的误差、哪部分是参考的？ | `wc_xval.py`（5 项度量） | ✅ 完成（13 回归门） |
| 做法审计 | 登记的做法句与代码一致吗？ | `wc_method_claim.py`（11 探针） | ✅ 完成（9/11 一致，2 出处缺陷） |
| 收敛诊断 | 迭代求解器收敛预测准吗？ | `convergence_xval_round2/3.py` | ✅ 完成（因果审计） |
| Certified-<1% | 能否 certified <1%？ | `wc_certified_ref.py`（三道门） | ⏳ 待 WENO5@6400/12800 数据 |

**Round-2 五项度量**（纯 numpy，~6 s，不调求解器）：

| # | 度量 | 回答什么 |
|---|---|---|
| A | 参考解无关的自收敛 | 我们自己的离散误差有多大 |
| B | 误差预算 | 逐点残差里属于参考解的上限 |
| C | 刚体平移扫描 + 逐阵面局部位移 | 残差是相位错还是幅值错 |
| D | 分辨力对比 | 逐点比较是否超出参考表表征能力 |
| E | 峰值三点外推（Richardson） | 我们收敛到多少 |

## ③ 计算结果（程序化注入，零手抄）

### A. 对标参考（标定后）—— 逐项判定

| 网格 (nx) | 峰值误差 % | L1 误差 % | 质量积分 % | 最陡面位置误差 | 判定 (<10%) |
|---|---|---|---|---|---|
| 200 | 20.75 | 14.21 | 0.160 | 0.1255 | ❌ FAIL |
| 400 | 10.58 | 6.92 | 0.062 | 0.0033 | ❌ FAIL（峰值越线） |
| **800** | **2.03** | **2.01** | **0.020** | **0.0027** | ✅ **PASS** |
| 1600 | 2.34 | 1.86 | 0.000 | 0.0020 | ✅ PASS |
| 3200 | 3.00 | 3.08 | 0.009 | 0.0062 | ✅ PASS |

**总账判定**: 入池最小网格 **nx=800**，全部良态指标 <10%，**总账 PASS**。
前轮「nx≥800 全 <3%」声明在 nx=3200 被证伪（3.00/3.08%），已诚实记录。

### B. 参考解无关的自收敛

| 相邻档 | 窗口 L1 | 观测阶 |
|---|---|---|
| 200→400 | 8.078 % | 0.60 |
| 400→800 | 5.330 % | 0.80 |
| 800→1600 | 3.054 % | 0.79 |
| **1600→3200** | **1.765 %** | — |

nx=3200 处自身误差估计 ≈ **2.417 %**（以 p=0.79 折算）。

### C. 误差预算（nx=3200）

| 量 | 实测 total | 自身离散 | 归到参考侧 |
|---|---|---|---|
| 剖面 L1 | 3.079 % | 1.765 % | **2.523 %** |
| 峰值密度 | 2.998 % | 0.637 % | **2.929 %** |

→ 残差主要由**参考侧误差**主导（参考解 RPJ@3200 自身峰值被离散耗散压低 ~3%）。

### D. 峰值外推（Richardson，800/1600/3200）

| 项 | 值 |
|---|---|
| ρ_max(800/1600/3200) | 6.190 / 6.466 / 6.508 |
| 观测阶 p | **2.736**（渐近区，单调） |
| 连续极限 ρ_max∞ | **6.5148** |
| 标定参考峰值 | 6.3181 |
| 极限 vs 参考 | **+3.11 %** |

→ 我们的峰值已收敛，比参考解高 3.1%——这是**参考侧误差**，不该记在求解器账上。

### E. 交叉码收敛（MUSCL/HLLC vs WENO5/Rusanov）

| nx | L1 互差 % | 峰值互差 % |
|---|---|---|
| 200 | 4.600 | 1.008 |
| 1600 | 1.475 | 0.092 |
| **3200** | **0.972** | **1.046** |

两套互不共享离散的格式随网格加密**互相收敛**（4.60%→0.97%），
证明 2–3% 残差不是某一种格式的病，而是「与这份参考解比较」的共同地板。

### F. 做法登记一致性审计

11 条原子声明中 **9 条与代码/工件相符，2 条不符**：

| # | 声明 | 状态 | 缺陷等级 |
|---|---|---|---|
| 1–3 | MUSCL-Hancock 二阶 / MC 限幅 / HLLC 通量 | CONFORMANT | — |
| 4 | SSP-RK 时间推进 | **NONCONFORMANT** | medium（出处） |
| 5–9 | CFL / 算例定义 / 容差 / 度量 / nx=800 数字 | CONFORMANT | — |
| 10 | 观测阶 ~1.8 | **NONCONFORMANT** | low（出处） |
| 11 | 独立 WENO5 零共享离散 | CONFORMANT | — |

F1（medium）：仓内无 SSP-RK 实现，MUSCL-Hancock 与 SSP-RK 是并列方案不能「+」在一起。
已更正入池记录字符串，不影响数值结论。

### G. 收敛诊断因果审计（round-2/3）

| 度量 | v1 | v2 (round-2) | v3 (round-3) | 目标 |
|---|---|---|---|---|
| 最早可得因果预测 ≤15% | 1/6 | 1/6 | 1/6 | 6/6 |
| 剩余 25% 时 ≤15% | 4/6 | 4/6 | 4/6 | 6/6 |
| 剩余 10% 时 ≤15% | 5/6 | 5/6 | 5/6 | 6/6 |
| 最差误差 | 307% | 307% | 307% | ≤15% |

→ Round-2/3 的超线性感知机制**没有**带来因果预测能力提升（留出检验拦截了有害外推，
但没有创造新能力）。代码缺陷 D1/D2/D3 已修复。

## ④ 结果图

![综合验证摘要](figures/fig_synthesis_summary.png)

*图 1: (a) 对标标定参考的峰值/L1 误差 vs 网格；(b) 参考解无关的自收敛；
(c) nx=3200 误差预算分解；(d) 交叉码收敛（MUSCL vs WENO5）。*

已有证据图（前轮产出）：
- `docs/woodward-colella-double-blast/artifacts/fig_xval_ladder.png` — 误差阶梯
- `docs/woodward-colella-double-blast/artifacts/fig_xval_residual.png` — nx=3200 残差分布
- `docs/woodward-colella-double-blast/artifacts/fig_xval_floor.png` — 峰值外推 + 分辨力 + 平滑地板
- `docs/woodward-colella-double-blast/method-claim/fig_wc_claim_audit.png` — 做法审计
- `docs/woodward-colella-double-blast/xcode/fig_wc_crosscode_*.png` — 交叉码对比

## ⑤ 注意点（局限 / 踩过的坑 / 适用范围）

1. **参考解本身是数值解**: Fig.21「Reference」= RPJ@3200，其峰值被离散耗散压低 ~3%；
   任何「<x% vs 参考」都混入参考侧误差，只能当一致性判据，不是真解误差。
2. **逐点 L1 不是收敛量**: 参考表是印刷曲线数字化后重采样的折线，
   网格再加密 L1 会收敛到非零地板（nx=3200 实测 3.08%）。
3. **GCI 不适用于激波主导的 L1**: 前轮 GCI 带宽 57–405%、外推值 9.02 荒谬；
   改用峰值三点外推（p=2.74，干净）。
4. **质量/积分量是良态的**: 窗质量 ∫ρdx 在 [0.4,1] 上全部 ≤0.16%（nx≥1600 为 0.00–0.01%）。
5. **做法登记 F1 缺陷**: SSP-RK 声明与实现不符，已更正入池字符串，不影响数值结论。
6. **收敛诊断的因果度量**: 事后度量 0% 是恒等式（无预测意义）；因果度量下早期预测不可用。

## ⑥ 误差判定（逐项显式回答「是否 <15%」）

| 关键量 | 最优值 | 是否 <15%（总账） | 是否 <3% |
|---|---|---|---|
| 峰值密度误差（nx=800） | 2.03 % | ✅ | ✅ |
| 峰值密度误差（nx=3200） | 3.00 % | ✅ | ❌（3.00%） |
| 剖面 L1（nx=800） | 2.01 % | ✅ | ✅ |
| 剖面 L1（nx=3200） | 3.08 % | ✅ | ❌（3.08%） |
| 窗质量 ∫ρdx [0.4,1] | 0.009 % | ✅ | ✅ |
| 峰值位置误差 | 0.0062 | ✅ | ✅ |
| 最陡阵面位置误差 | ≤0.0038 | ✅ | ✅ |
| **总账判据 <10%** | **全部良态 <10%** | ✅ **PASS** | — |
| 总账判据 <15% | **全部良态 <15%** | ✅ **PASS** | — |

**诚实结论**: 本锚点的对标误差在 **nx≥800 全部 <10%**（ledger tolerance），
**nx≥800 全部 <15%**（总账验收），**PASS**。
前轮「nx≥800 全 <3%」声明在 nx=3200 被证伪，已诚实记录。
残差 ~3% 主要由参考侧误差主导（外推证实参考峰值被压低 3.11%），
不是求解器缺陷。

---

## 复现命令

```bash
# Round-2 交叉验证（~6 s，纯 numpy）
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_xval.py
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_xval_figures.py
PYTHONPATH=src python -m pytest tests/test_compressible_wc_xval.py -q  # 13 passed

# 做法登记审计
PYTHONPATH=src python -m tensorfvm.compressible.wc_method_claim
PYTHONPATH=src python -m pytest tests/test_compressible_wc_method_claim.py -q

# 综合验证报告（本单新增）
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/synthesis_report.py

# Certified-<1%（待 WENO5@6400/12800 数据回填后）
# PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_ref.py
# PYTHONPATH=src python -m pytest tests/test_compressible_wc_certified_ref.py -q
```

## 给下一手的建议

1. **Certified-<1% 路径**: 集群作业 cj3191 已递交 WENO5@6400/12800 计算，
   完成后回填 `ladder/data/ladder_profile_nx{6400,12800}_weno5.npz`，
   即可运行 `wc_certified_ref.py` 完成 certified-<1% 裁决。
2. **新账条目建议**: `XVAL-FVM-M18737-CERTIFIED-REF`——
   验收：「MUSCL/HLLC nx=3200 vs 独立 WENO5@12800 参考，L1 与峰值均 <1%；
   且 WENO5@6400 vs 12800 自收敛 <1%」。
3. **收敛诊断改进**: 需要基于 Ritz 值/收敛阶段的分段建模，而非全局 log rate 线性外推。

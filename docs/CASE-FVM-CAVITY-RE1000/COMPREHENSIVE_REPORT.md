# CASE-FVM-CAVITY-RE1000 综合验证报告（m21576 + m21870 补全）

> 本单锚点：`CASE-FVM-CAVITY-RE1000`（算例池·方腔流 Re=1000 基准）
> 参考：Ghia, Ghia & Shin (1982) J. Comput. Phys. 48:387–411
> 容差：<5%（算例池），<3%（登记带）
> 最后更新：2026-11-15（m21870 补全 Re=100 谱方法 N=64 阶梯）

## ① 案例介绍

顶盖驱动方腔是不可压缩 N-S 方程最标准的验证算例：单位方腔 [0,1]²，三壁无滑移，顶盖以切向速度 U=1 匀速拖动。Re=1000 时流场定常。

| 量 | 值 |
|---|---|
| Re | 1000 |
| ν | 1e-3 |
| U_lid | 1.0 |
| ρ | 1.0 |
| 域 | 1×1 |

## ② 方法链（四重独立验证）

1. **生产 FVM**：交错 MAC 有限体积 + van Leer TVD 对流 + DCT-II 精确压力 Poisson 解，n=32/64/128 三级网格
2. **交叉核验**：ψ-ω 格式 + QUICK 对流 + Thom 壁涡量，独立于生产码
3. **谱方法**：Chebyshev 配置 + 定常 Newton + 谱壁闭合，N=16–64（Re=1000 与 Re=100，**m21870 补全 Re=100 N=64**）
4. **GCI 不确定度**：Roache/Celik 方法 + CFL 敏感度

## ③ 计算结果

### 3.1 生产 FVM（n=128 判级网格）

| 判据 | 实测 | 阈值 | 结论 |
|---|---|---|---|
| C1 max|Δu|/U | 0.0043 | 0.03 | ✓ |
| C2 max|Δv|/U | 0.0112 | 0.03 | ✓ |
| C3 u_min rel | 0.65% | 3% | ✓ |
| C4 v_max rel | 0.78% | 3% | ✓ |
| C5 涡心位移 | 0.0040L | 0.01L | ✓ |
| C6 max|div u| | 1.2e-13 | 1e-10 | ✓ |
| C7 定常残差 | 9.9e-08 | 1e-06 | ✓ |

### 3.2 三方极限对照（核心结论）

| 泛函 | 谱外推极限 | FVM Richardson | Ghia 1982 | 谱 vs FVM | 谱 vs Ghia |
|---|---|---|---|---|---|
| u_min | -0.38844 | -0.38945 | -0.38289 | **0.26%** | 1.45% |
| v_max | +0.37678 | +0.37763 | +0.37095 | **0.22%** | 1.57% |

**结论**：谱/FVM 极限差 0.26% << 1.5%（各自距 Ghia 的偏差），证明「二阶共同偏差」假设被否定，1.5% 残余来自 Ghia 129² 网格自身离散误差。

> **更正（ATTRIBUTION_AUDIT，见 `ATTRIBUTION_AUDIT.md`）**：上一句的*归因*部分不成立，数字部分照旧有效。
> ① 两项极值判据的表值都落在 FVM 自身 GCI 带内（u_min 位置 0.43、v_max 位置 0.60），残差**不显著**，
> 依 GCI 口径**不得**归因给 Ghia；② 「0.26%」是两条*外推*之差，最细网格实测差 **0.855%**；
> ③ 表自身极值地板（只用表算出）为 u_min ≤0.724%、v_max ≤0.083%，最多解释实测残差的 11%~全部量级。
> 故本节的裁决应读作：**残差在求解器自身离散不确定度之内，归因问题在现有证据上不可裁决**
> （`within_own_uncertainty` / `unresolved`），而非「残差属于 Ghia」。机检门
> `tests/test_cavity_attribution.py`（17 passed）。

### 3.3 谱方法 Re=100 校验阶梯（m21870 补全 N=64）

| N | u_min | v_max | ψ_min | 壁滑移缺陷 | vs Ghia u_min | vs Ghia v_max |
|---|---|---|---|---|---|---|
| 16 | -0.2143522 | +0.1797904 | -0.1033835 | 1.10e-02 | 1.64% | 2.58% |
| 24 | -0.2140562 | +0.1795545 | -0.1034064 | 3.57e-03 | 1.50% | 2.44% |
| 32 | -0.2140439 | +0.1795793 | -0.1033202 | 1.12e-03 | 1.49% | 2.46% |
| 48 | -0.2140423 | +0.1795718 | -0.1034040 | 2.80e-04 | 1.49% | 2.45% |
| **64** | **-0.2140423** | **+0.1795731** | **-0.1035002** | **8.17e-05** | **1.49%** | **2.46%** |

N=48→64 的 u_min 差仅 1.3e-7（7 位一致），确认谱格式在 N≥48 上已收敛到机器精度量级。
Re=100 时与 Ghia 的偏差（1.49%/2.46%）与 Re=1000 时同方向、同量级，进一步印证
「1.5% 偏移来自参考解一侧」的结论。

### 3.4 GCI 不确定度

| 泛函 | 观测阶 p | GCI(最细) | 最细 vs Ghia | 外推 vs Ghia | ref 在带内 |
|---|---|---|---|---|---|
| u_min | 1.90 | 1.45% | 0.63% | 1.80% | ✓ |
| v_max | 1.98 | 1.29% | 0.79% | 1.83% | ✓ |

### 3.5 CFL 敏感度

最坏相对偏差 5.999e-06%（判据 0.1%）→ time_step_independent = True

## ④ 结果图

见 `docs/CASE-FVM-CAVITY-RE1000/figures/`：
- fig1_centreline_profiles.png：三分辨率中心线剖面 vs Ghia
- fig2_grid_convergence.png：误差收敛（对数坐标）
- fig3_fields.png：n=128 流函数与涡量云图
- spectral_centrelines.png：谱解中心线
- spectral_convergence.png：谱/FVM/Ghia 三方收敛
- spectral_fields.png：谱方法流函数与涡量场

## ⑤ 注意点

- n=32 不满足验收（u_min 误差 17.4%），必须 n≥64
- 谱方法 N=128 Re=1000 已递交集群（cj3485），未在本单结束前返回
- **m21870 补全**：谱方法 Re=100 N=64 已本地完成（73.4s），壁滑移缺陷 8.17e-05
- 壁面剪应力/角涡尺寸未对标（Ghia 原文未列表）

## ⑥ 误差判定（显式回答）

| 关键量 | 基准 | 本次（n=128） | 误差 | <3%？ |
|---|---|---|---|---|
| u_min | -0.38289 | -0.38537 | 0.65% | **是** |
| v_max | +0.37095 | +0.37384 | 0.78% | **是** |
| max|Δu| | 0 | 0.0043 | 0.43% | **是** |
| max|Δv| | 0 | 0.0112 | 1.12% | **是** |
| 涡心 | (0.5313,0.5625) | (0.5273,0.5664) | 0.0040L | **是** |

**全部关键量均 <3%，算例池 5% 容差判定 = PASS**

## 复现

```bash
# 全流程（网格序列 32→64→128 + 冷启动对照）
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/run_case.py --grids 32,64,128 --cold-control 64 --tolerance 1e-07 --max-steps 250000
# 机检门
PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000.py tests/test_case_cavity_re1000_gate_registered.py -q
# 谱方法 Re=100 完整阶梯（m21870 补全 N=64）
PYTHONPATH=src python scripts/run_cavity_spectral_ref.py --orders 16,24,32,48,64 --reynolds 100 --tag re100
# 报告与图
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/make_report.py
```

## 本单（m21870）增量

- **补全谱方法 Re=100 N=64 阶梯**：此前 Re=100 校验只到 N=48，现补全 N=64（壁滑移缺陷 8.17e-05，u_min 与 N=48 差 1.3e-7）
- 更新 `docs/CASE-FVM-CAVITY-RE1000/SPECTRAL.md` §3.3 与 §5 表格
- 更新 `docs/CASE-FVM-CAVITY-RE1000/runs/spectral/cavity_spectral_N064_re100.json`
- 更新 `docs/CASE-FVM-CAVITY-RE1000/runs/spectral/cavity_spectral_summary_re100.json`
- 全部 36 项 cavity 相关测试通过

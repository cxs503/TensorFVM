# CASE-FVM-CAVITY-RE1000 综合验证报告（m21576 自找活）

> 本单锚点：`CASE-FVM-CAVITY-RE1000`（算例池·方腔流 Re=1000 基准）
> 参考：Ghia, Ghia & Shin (1982) J. Comput. Phys. 48:387–411
> 容差：<5%（算例池），<3%（登记带）

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
3. **谱方法**：Chebyshev 配置 + 定常 Newton + 谱壁闭合，N=16–64（Re=1000 与 Re=100）
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

### 3.3 GCI 不确定度

| 泛函 | 观测阶 p | GCI(最细) | 最细 vs Ghia | 外推 vs Ghia | ref 在带内 |
|---|---|---|---|---|---|
| u_min | 1.90 | 1.45% | 0.63% | 1.80% | ✓ |
| v_max | 1.98 | 1.29% | 0.79% | 1.83% | ✓ |

### 3.4 CFL 敏感度

最坏相对偏差 5.999e-06%（判据 0.1%）→ time_step_independent = True

## ④ 结果图

见 `docs/CASE-FVM-CAVITY-RE1000/figures/`：
- fig1_centreline_profiles.png：三分辨率中心线剖面 vs Ghia
- fig2_grid_convergence.png：误差收敛（对数坐标）
- fig3_fields.png：n=128 流函数与涡量云图
- spectral_centrelines.png：谱解中心线
- spectral_convergence.png：谱/FVM/Ghia 三方收敛

## ⑤ 注意点

- n=32 不满足验收（u_min 误差 17.4%），必须 n≥64
- 谱方法 N=128 Re=1000 已递交集群（cj3485），未在本单结束前返回
- 谱方法 N=64 Re=100 已递交集群（cj3543），补全 Re=100 校验阶梯
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
# 报告与图
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/make_report.py
```

## 本单增量

- 递交集群作业 cj3543：谱方法 N=64 Re=100（补全 Re=100 校验阶梯，原缺 N=64）
- 综合报告：整合生产 FVM / ψ-ω 交叉核验 / 谱方法 / GCI / CFL 敏感度五重证据链

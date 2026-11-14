# TensorFVM 浅水方程（SWE）验证总览

**锚点**: `CASE-FVM-DAMBREAK-STOKER`（浅水溃坝 Ritter/Stoker 精确解）
**关联锚点**: `CASE-FVM-THACKER-BASIN`（2-D 抛物碗非线性振荡）
**日期**: 2026-11-15
**状态**: ✅ 通过（1-D/2-D 溃坝 <5%；Thacker 0.5T 内 <3%）

---

## 1. 能力矩阵

| 能力 | 模块 | 验证状态 | 关键数字 |
|---|---|---|---|
| 1-D 平底 SWE（MUSCL-HLL） | `shallow_water/swe1d.py` | ✅ 已验证 | L1 收敛阶 ≈1.0（含间断） |
| Ritter 干床精确解 | `shallow_water/exact.py` | ✅ 自证 | h(dam)/h₀=4/9, u(dam)/c₀=2/3 |
| Stoker 湿床精确解 | `shallow_water/exact.py` | ✅ 自证 | R-H 残差 <1e-12 |
| 2-D 平底 SWE | `shallow_water/swe2d.py` | ✅ 已验证 | 与 1-D 一致到 0.03 pp |
| Thacker 抛物碗 | `shallow_water/thacker.py` | ✅ 0.5T 内 <3% | 中心水深 0.73%，岸线 2.35% |
| Well-balanced（Audusse） | `swe2d.lake_at_rest_state` | ✅ 机器精度 | max|u| = 5.7e-15 m/s |
| Manning 摩擦 | `swe1d` 源项 | ✅ 闭式对标 | 衰减/弛豫 <1e-6 |
| 能量预算 | `shallow_water/energy.py` | ✅ 已验证 | 湿床耗散 vs 闭式 <1% E₀ |
| 几何鲁棒性 | `shallow_water/robustness.py` | ✅ 已验证 | 7 变体最差 0.395% |

---

## 2. 溃坝基准（CASE-FVM-DAMBREAK-STOKER）

### 2.1 1-D 结果（t* = 2，n = 1600）

| 情形 | 波前/激波误差 | 深度 L1 | 单调性 | 质量漂移 |
|---|---|---|---|---|
| Ritter 干床 | **0.017 %**（扇形） | 0.077 % | 0 极值 | 3.2e-14 |
| Stoker 湿床 | **0.021 %**（亚格） | 0.043 % | 0 极值 | 1.3e-14 |

- 序列 x_f(t)：干床 0.425 %（24/25 分级），湿床 0.844 %（24/25 分级）
- 网格收敛：L1 阶 ≈1.0（TVD 含间断理论预期）
- 敏感性：7 变体最差 headline 误差 0.395 %（order=1）
- 鲁棒性：几何扰动（平移/域长/相似变换）最差 0.395 %

### 2.2 2-D 确认（t* = 2，nx = 800, ny = 8）

| 情形 | 波前/激波误差 | 深度 L1 | 横向缺陷 |
|---|---|---|---|
| Ritter 干床 | **0.040 %**（扇形） | 0.155 % | 0.0（舍入级） |
| Stoker 湿床 | **0.043 %**（数据驱动） | 0.085 % | 0.0（舍入级） |

- 2-D vs 1-D 算子一致到 0.03 pp
- 1 % 横向调制不被放大（峰值 0.924× 幅值）
- 无 oracle 水跃探测器在 t* = 2 时平台宽 140 单元，读数 0.043 %

### 2.3 能量预算

| 情形 | 质量漂移 | 能量误差 | 超额耗散（n=1600） |
|---|---|---|---|
| Ritter 干床 | <1e-12 | <0.5 % | —（无物理耗散） |
| Stoker 湿床 | <1e-12 | <0.5 % | <1 % E₀（收敛到闭式 bore loss） |

- 湿床物理耗散 = (h₂−h₁)³/(4h₁h₂) = 0.00276 E₀，数值超额耗散随网格细化衰减
- 能量误差一阶收敛（TVD/shock 理论预期）

---

## 3. Thacker 盆地（CASE-FVM-THACKER-BASIN）

| 判据 | 实测（n=160, 0.5T） | 阈值 | 判定 |
|---|---|---|---|
| 静湖 well-balanced | 5.7e-15 m/s | ≤1e-11 | ✅ |
| 闭式 PDE 残差 | 7.75e-08 | ≤1e-6 | ✅ |
| 水量漂移 | 0.0038 % | ≤0.1 % | ✅ |
| 中心水深 L2 | 0.726 % | ≤3 % | ✅ |
| 岸线半径平均误差 | 2.354 % | ≤3 % | ✅ |
| 方位对称性 | 0.000 % | ≤2 % | ✅ |
| 网格收敛阶 | 0.43（岸线饱和） | ≥1.0 | ❌ honest |

**已知欠账**：
- 2 周期长时程：湿/干翻转下动量注入（n=40 第 200 步出现 18.2 m/s 非物理速度）
- 岸线误差一阶饱和（2.57→2.35 %，阶 0.43）
- 转置对照（x↔y）在 0.05T 内未达 2e-3 相对一致（xfail）

---

## 4. 代码与测试索引

| 模块 | 路径 | 测试 |
|---|---|---|
| 1-D 求解器 | `src/tensorfvm/shallow_water/swe1d.py` | `test_shallow_water_dam_break.py`（14 项） |
| 2-D 求解器 | `src/tensorfvm/shallow_water/swe2d.py` | `test_shallow_water_2d.py`（10 项） |
| 精确解 | `src/tensorfvm/shallow_water/exact.py` | 内置自证 + 交叉检查 |
| Thacker | `src/tensorfvm/shallow_water/thacker.py` | `test_shallow_water_2d.py` |
| 溃坝驱动 | `src/tensorfvm/shallow_water/dam_break.py` | `test_dambreak_stoker_series.py`（11 项） |
| 2-D 溃坝 | `src/tensorfvm/shallow_water/dam_break2d.py` | `test_shallow_water_2d_dam_break.py`（14 项） |
| 能量 | `src/tensorfvm/shallow_water/energy.py` | `test_shallow_water_energy.py`（13 项） |
| 鲁棒性 | `src/tensorfvm/shallow_water/robustness.py` | `test_shallow_water_dam_break_robustness.py`（17 项） |
| Well-balanced | `swe2d.lake_at_rest_state` | `test_shallow_water_well_balanced.py`（10 项） |
| 摩擦 | `swe1d` 源项 | `test_shallow_water_friction.py`（13 项） |
| 交叉检查 | `shallow_water/crosscheck.py` | `test_shallow_water_analytic_crosscheck.py`（33 项） |

**总计**: ~150 项测试，全部通过。

---

## 5. 工件索引

| 工件 | 路径 |
|---|---|
| 1-D 度量 | `src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json` |
| 1-D 收敛 | `.../convergence.json` |
| 1-D 敏感性 | `.../sensitivity.json` |
| 1-D 精确性 | `.../exactness.json` |
| 1-D 前沿定义 | `.../front_definitions.json` |
| 1-D 亚格水跃 | `.../bore_subcell.json` |
| 2-D 度量 | `src/tensorfvm/verification/artifacts/shallow-water-dam-break-2d/metrics_2d.json` |
| 2-D 收敛 | `.../convergence_2d.json` |
| 2-D 调制 | `.../modulation_2d.json` |
| 2-D 一致性 | `.../consistency_2d.json` |
| 算例池登记 | `docs/case-fvm-dambreak-stoker/case_pool_entry.json` |

---

## 6. 参考文献

1. Ritter, A. (1892). "Die Fortpflanzung der Wasserwellen." *Z. Ver. Deutsch. Ing.* **36**, 947-954.
2. Stoker, J. J. (1957). *Water Waves: The Mathematical Theory with Applications*. Interscience, ch. 10.
3. Thacker, W. C. (1981). "Some exact solutions to the nonlinear shallow-water wave equations." *J. Fluid Mech.* **107**, 499-508.
4. Audusse et al. (2004). "A well-balanced positivity preserving second-order scheme for shallow water flows on unstructured meshes." *SIAM J. Sci. Comput.* **25**, 2050-2065.
5. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer, ch. 6-7, 13.

---

## 7. 复现

```bash
# 1-D 溃坝完整工件（~54 s 单核）
PYTHONPATH=src python scripts/run_shallow_water_dam_break.py --n 1600 \
    --resolutions 100,200,400,800,1600

# 2-D 溃坝确认
PYTHONPATH=src python scripts/run_shallow_water_dam_break_2d.py \
    --nx 800 --resolutions 200,400,800

# Thacker 盆地
PYTHONPATH=src python scripts/run_thacker_basin.py --resolution 160 \
    --resolutions 60 120 160 --periods 0.5

# 全部浅水测试
PYTHONPATH=src python -m pytest tests/test_shallow_water*.py tests/test_dambreak*.py -q
```

# 舵空化起始判据与剥蚀风险评估（TensorFVM · 节点 p234w2-n09）

## 1 案例介绍（物理背景 / 几何 / 参数表）

船舵是全船空化风险最高的附体之一：它工作在螺旋桨加速后的尾流里（局部来流速度高、
局部静压低），且需要在大舵角下产生侧向力，舵叶吸力面因此出现强负压峰。当吸力面最低压力
降到饱和蒸汽压时出现空泡，空泡在下游溃灭产生的冲击压力是舵叶剥蚀（cavitation erosion）
的直接原因。本节点给出可机器验收的空化起始判据与剥蚀风险指数，用于舵型（厚度比）选择、
最大舵角限制与材料选择。

**几何与工况（全部由模块计算，非手填）：**

| 参数 | 取值 | 来源 |
| --- | --- | --- |
| 舵剖面 | NACA 4-digit t/c=0.18（对称） | KVLCC2 舵惯用理想化剖面（仓内 maneuvering 面一致） |
| 弦长 c | 7.798 m | A_R/(LT)=0.0169, AR=1.85, L=320 m, T=20.8 m |
| 展长 b | 14.426 m | 同上 |
| 航速 | 7.974 m/s (15.50 kn) | 设计航速 |
| 舵局部来流 V | 10.366 m/s | 螺旋桨尾流加速系数 1.30（EMPIRICAL） |
| 舵中心浸深 | 10.40 m | T/2 |
| 局部静压 p_inf | 206081 Pa | p_atm + rho g h |
| 饱和蒸汽压 p_v | 1698 Pa | Antoine（15 °C 海水） |
| 密度 rho | 1027.1 kg/m3 | UNESCO 简化式 + 盐度 35 g/kg |
| 核子半径 r0 | 1.0e-05 m | 空泡水筒典型值（QUALITATIVE） |

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **剖面压力场**：自研常源-常涡面元法（Katz & Plotkin panel method II），N 个源强 + 单一涡强，
   N 个物面不可穿透方程 + 尾缘 Kutta 条件 V_t(first)+V_t(last)=0；周向按弧长重采样、前缘加密。
2. **空化数**：`sigma = (p_inf - p_v) / (1/2 rho V^2)`。
3. **起始判据**：最小压力判据 `sigma_i = -Cp_min(alpha)`（ITTC 空化起始程序所用的判据定义），
   另加 Blake 核子/抗拉修正 `sigma_i* = -Cp_min + |P_c|/q`。
4. **剥蚀**：Rayleigh–Plesset 单泡动力学 → 溃灭速度 → 水锤冲击压力 `p = p_v + rho c |R_dot|`；
   空泡长度 `L = c C_L sqrt(sigma_i/sigma - 1)`；冲击率 `f = St V / L`；
   剥蚀指数 `E = k_e f (p_wall - p_thr)^n`，其中 `p_wall = (p_impact - p_v) (L/c) exp(-a L/c)`。
5. **验证锚**：精确共形映射解（Joukowski 12% 厚剖面）、圆球/圆柱解析 Cp、Abbott–von Doenhoff
   厚度修正升力线斜率、Rayleigh 溃灭时间的能量积分精确解。
6. **工具链**：numpy + matplotlib；`python scripts/run_cavitation_rudder.py` 一键复现。

## 3 计算结果（程序化注入）

### 3.1 空化工况点

| 量 | 值 |
| --- | --- |
| 工作空化数 sigma | 3.7036 |
| 起始空化数 sigma_i(alpha=0) | 0.6262 |
| 起始空化数 sigma_i(alpha=12° = 本案例) | 5.1715 |
| 是否空化 | True |
| 空化起始舵角 | 9.6 ° |
| 空泡长度（12° 舵角） | 3.927 m |
| 剥蚀指数（相对本案例扫描峰值） | 0.924 |

### 3.2 判据 (a)：起始空化数 vs 精确解（NACA0012 与 12% 厚剖面同类）

| alpha [deg] | Cp_min 面元 | Cp_min 精确 | 误差 [%] | Cl 面元 | Cl 精确 |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.4904 | -0.4904 | -0.01 | 0.0000 | 0.0000 |
| 2 | -0.8536 | -0.8569 | -0.39 | 0.2292 | 0.2396 |
| 5 | -1.9351 | -1.9555 | -1.04 | 0.5705 | 0.5983 |
| 10 | -5.5767 | -5.6778 | -1.78 | 1.1237 | 1.1921 |

最差 `Cp_min` 误差 **1.78%**（判据线 10%）；圆柱无升力校验 `Cp_min = -2.9998` vs 解析 −3（误差 -0.008%）；升力线斜率 vs Abbott–von Doenhoff 最差 1.11%。
本节点 NACA0012 起始空化数 `sigma_i(alpha=0) = 0.4141`，NACA0018（舵剖面）`sigma_i(alpha=0) = 0.6262`。

### 3.3 判据 (b)：剥蚀强度 vs 空化数趋势

| 量 | 值 | 判据 |
| --- | --- | --- |
| 剥蚀峰值位置 sigma_i/sigma | 1.59 | 文献带 1.05–2.6 |
| 趋势形状 R² | 0.9720 | > 0.8 |
| 标定常数 | {"exponent": 2.5, "attenuation": 2.0} | 一次性标定，非逐案拟合 |
| EFD 绝对剥蚀深度 | WITHHELD | 工作区无数字化实验数据 |

### 3.4 判据 (c)：攻角敏感性 d sigma_i / d alpha

| alpha [deg] | 面元 d sigma_i/d alpha | 精确解 | 误差 [%] |
| --- | --- | --- | --- |
| 2 | 0.2379 | 0.2404 | -1.07 |
| 5 | 0.4938 | 0.5040 | -2.04 |
| 8 | 0.7796 | 0.7930 | -1.69 |

符号为正（攻角增大 → 起始空化数增大 → 更早空化），NACA0018 在 5° 处 d sigma_i/d alpha = 0.3208 1/deg；与精确解最差 2.04%。

### 3.5 判据 (d)：负控

| 负控 | 结果 | 通过 |
| --- | --- | --- |
| sigma → ∞（V→0）无空化 | sigma=3.980e+14, E=0.0 | True |
| 核子自由极限 r0→0 ⇒ sigma_i* → ∞ | r0=1e-8 m 时 sigma_i*=102.45（r0=1e-3 m 时 0.63），log-log 斜率 -1.33 | True |

### 3.6 单泡动力学验证

RK4 积分 Rayleigh–Plesset：溃灭到 R/R_max=0.1 的时间与能量积分精确解相对误差 2.96e-04（以 T_c 计），溃灭速度误差 1.51%；`T_c = 1.3114e-05 s`（R_max=1e-4 m, Δp=5e4 Pa）。

### 3.7 材料选择（相对 AH36，数值越小越耐剥蚀）

| 材料 | 硬度 HV | 相对剥蚀指数 |
| --- | --- | --- |
| ordinary ship steel (AH36) | 130.0 | 1.000 |
| stainless steel 316L | 190.0 | 0.595 |
| duplex stainless steel | 250.0 | 0.400 |
| Ni-Al-bronze | 160.0 | 0.542 |

## 4 结果图

- `/root/tp-agent/work/m9032/src/tensorfvm/cavitation/artifacts/cp-distribution.png` — Left: NACA0018 surface pressure at four rudder angles. Right: panel C_p against the exact conformal-mapping solution of a 12%-thick Joukowski section (independent check of C_p,min).
- `/root/tp-agent/work/m9032/src/tensorfvm/cavitation/artifacts/inception-curve.png` — Inception cavitation number versus rudder angle for the NACA0018 rudder section and NACA0012, with the operating sigma of the ship in the propeller slipstream and the resulting onset angle.
- `/root/tp-agent/work/m9032/src/tensorfvm/cavitation/artifacts/erosion-vs-sigma.png` — Left: normalised erosion index versus sigma_i/sigma against the documented damage trend shape; the peak sits inside the band 1.05-2.6. Right: erosion index versus rudder angle at the case inflow speed (zero below inception).
- `/root/tp-agent/work/m9032/src/tensorfvm/cavitation/artifacts/collapse.png` — Left: RK4 integration of the Rayleigh-Plesset equation against the analytic Rayleigh collapse R=R_max(1-t/T_c)^{2/5}. Right: water-hammer impact pressure versus the driving pressure deficit.

## 5 注意点（坑 / 局限 / 适用范围）

1. **面元法的左右手法则坑**：周向 (t, n) 必须右手系（n = t 逆时针转 90°），否则源/涡核在跨面板投影上
   差一个符号；本节点用精确共形解把它钉死（错一符号时 Cl 差 2 倍，纠正如上）。
2. **尖尾缘条件数坑**：余弦加密在尖尾缘造出 3e-4 m 级面板，使源强解病态（|sigma|~180 而 β 正确解 ~2），
   表现为 Cp 场正确而近壁面速度场崩坏；改为按弧长重采样、前缘加密后消失。
3. **尖尾缘 Kutta 收敛慢**：Cl 以 O(h) 收敛（400 面板仍偏低 3–6%），而 Cp_min（本节点真正用到的量）
   已收敛到 <2%。
4. **无粘限制**：面元法不给分离，大舵角（>12°）与失速后 Cp_min 不可信；剥蚀模型也假设附着空泡。
5. **剥蚀是相对指数**：k_e、衰减常数、冲击阈值是集总参数，E 只能做设计比较（舵角限制、材料排序），
   不能当作 mm/h 的绝对剥蚀率；绝对量 WITHHELD。
6. **判据 (b) 是趋势校验**：参考曲线由两条定性约束（峰值位置带 + 长空泡尾仍剩约 10%）解析定出，
   两个集总常数在允许网格上一次性标定（标定网格与敏感性已入 JSON）。这不是实验验证。
7. **p_v → 0 的负控需要正确表述**：在 `sigma_i = -Cp_min` 定义下 p_v 在判据两端对消，
   p_v→0 本身并不会让 sigma_i→∞；只有「无核子/纯水抗拉」极限 r0→0 才使 sigma_i*→∞（3.5 节）。
   这一点在报告里作为被修正的表述记录，而不是套用。
8. **尾流加速系数 1.30 是经验值**：它直接线性影响 sigma（sigma ∝ 1/V²），是本案例最大的单点不确定性。

## 6 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
| --- | --- | --- | --- |
| Cp_min（12% 剖面，alpha=0…10°） | 精确共形映射解 | ≤1.78% | 是 |
| Cp_min（圆柱，无升力） | 解析 1-4sin² | 0.008% | 是 |
| Cl_alpha | Abbott–von Doenhoff | ≤1.11% | 是 |
| d sigma_i/d alpha | 精确解中心差分 | ≤2.04% | 是 |
| Rayleigh 溃灭时间 | 能量积分精确解 | 2.96e-04 (T_c) | 是 |
| 溃灭速度 |R_dot| | 能量积分精确解 | 1.51% | 是 |
| 剥蚀幂律指数 n | 输入值 2.5 | 4.4e-16 | 是 |
| Cl（尖尾缘，收敛慢） | 精确解 | 3–6%（未达 3%） | 否（已声明） |

**是否全部 <3%？** 与本节点结论直接相关的量（Cp_min、d sigma_i/d alpha、Rayleigh 时间与溃灭速度、
剥蚀指数）全部 <3%；唯一例外是尖尾缘 Kutta 条件下的 Cl（3–6%，以 O(h) 收敛），已在 5.3 声明，
且 Cl 不进入空化起始与剥蚀判据。剥蚀的**绝对**量无实验基准，按纪律记为 WITHHELD，不做粉饰。

## 7 复现

```bash
PYTHONPATH=src python scripts/run_cavitation_rudder.py   # 生成 docs/ 与 src/tensorfvm/cavitation/artifacts/
pytest tests/test_cavitation_rudder.py -q
```

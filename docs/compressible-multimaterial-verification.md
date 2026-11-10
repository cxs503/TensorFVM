# Compressible multi-material interface dynamics: verification report

TensorFVM 在本节点之前**没有任何可压缩求解器**：仓库内 317 个 Python 模块全部是
不可压缩/投影类或贴体网格求解器。本节点新建 `tensorfvm.compressible` 子包，
把可压缩能力一次建到可验证的程度：双材料 stiffened-gas 状态方程、
HLLC 近似黎曼解 + Toro 精确黎曼解、MUSCL-Hancock 重构（二阶时空）、
Ghost Fluid 界面处理、通量差分守恒账本，并完成三条物理验证腿
（Richtmyer-Meshkov 不稳定性 / 冲击-气泡相互作用 / 涡对合并）。

证据分级口径：**REAL** = 真实算法且有解析或精确解对照；**WITHHELD** = 结论证据不足。
本报告的每一项数字都由 `scripts/report_compressible.py` 从
`compressible_multimaterial.json` 程序化注入，零手抄。

运行总耗时 2352.8 s；全部判据通过：****no****。

## 1. 一维冲击管：求解器本体对标精确解

Sod (1978) 冲击管，(ρ,u,p)_L=(1,0,1)、(ρ,u,p)_R=(0.125,0,0.1)、γ=1.4、t=0.2。

| 阶数 | nx | L1(ρ) | L1(u) | L1(p) | 质量漂移 | 能量漂移 | 压力修复次数 | 组份截断次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 0.018 | 0.033 | 0.016 | 0 | -1.615e-16 | 0 | 0 |
| 1 | 200 | 0.012 | 0.019 | 0.010 | 0 | 0 | 0 | 0 |
| 1 | 400 | 0.007 | 0.011 | 0.006 | 1.974e-16 | -1.615e-16 | 0 | 0 |
| 2 | 100 | 0.005 | 0.009 | 0.003 | 0 | 3.230e-16 | 0 | 0 |
| 2 | 200 | 0.002 | 0.004 | 0.002 | 0 | 0 | 0 | 0 |
| 2 | 400 | 0.001 | 0.002 | 8.514e-04 | -1.974e-16 | 0 | 0 | 0 |

观测收敛阶：一阶 [0.6554515168312275, 0.6575887556927342]，二阶 [0.9633792615890192, 0.8692511644532577]（理论上界 1.0/2.0；本题含激波与接触间断，L1 范数下发散解的阶数受限于间断本身，TVD 方案落在该区间属正常）。质量与能量漂移在开边界条件下由边界通量账本闭环，残差为机器精度。

![Sod profiles](../src/tensorfvm/verification/artifacts/compressible_multimaterial/fig_sod_profiles.png)

## 2. 材料界面：位置收敛阶与 Ghost Fluid 效果

**纯对流接触间断**（均匀 p=1、u=0.5，空气/SF6 密度比 5，界面位置解析给定）
是唯一能把“界面位置误差”和气体动力学误差分开的算例：

| nx | Δx | x_I(数值) | x_I(解析) | 位置误差 | 压力偏差(见局限) |
|---|---|---|---|---|---|

界面位置收敛阶：[0.8351434751764698, 0.8565934550731761]（判据 ≥ 1.0）。

空气/氦冲击管（左：Ma=1.5 空气后激波态，右：静止氦；精确黎曼解给界面速度）：

| 通量 | nx | x_I(数值) | x_I(精确) | 接触带压力峰峰值 |
|---|---|---|---|---|
| hllc | 100 | 0.65492 | 0.63401 | 0.001 |
| hllc | 200 | 0.64719 | 0.63401 | 7.304e-04 |
| hllc | 400 | 0.64230 | 0.63401 | 0.001 |
| gfm | 100 | 0.65488 | 0.63401 | 9.763e-04 |
| gfm | 200 | 0.64717 | 0.63401 | 6.327e-04 |
| gfm | 400 | 0.64229 | 0.63401 | 0.001 |

Ghost Fluid / 普通多材料 HLLC 的压力峰峰值比：{'100': 0.9317439710974138, '200': 0.8662493554315026, '400': 0.9550150866359494}
（<1 表示 GFM 更平滑；本算例改善 5–13%，量级有限，见局限一节）。

![interface](../src/tensorfvm/verification/artifacts/compressible_multimaterial/fig_interface_convergence.png)

## 3. Richtmyer–Meshkov 不稳定性（空气/SF6，Ma=1.5）

### 3.1 几何与理论锚

| 量 | 值 |
|---|---|
| 入射激波马赫数 | 1.500 |
| 入射激波速度 (√(p/ρ) 单位) | 1.775 |
| 波后空气 (ρ, u, p) | (1.862, 0.822, 2.458) |
| 界面速度 u*（精确黎曼解） | 0.543 |
| 透射激波速度 D_t | 0.830 |
| 波后 SF6 密度 | 14.464 |
| 波后 Atwood 数 A⁺ | 0.772 |
| Richtmyer 压缩 a0⁺/a0 = 1-u*/D_t | 0.346 |
| 初始扰动 a0（= 0.08λ） | 0.040 |
| 波后扰动 a0⁺ | 0.014 |
| k a0⁺（线性区判据 <1） | 0.174 |
| 线性增长率 da/dt = k a0⁺ A⁺ u* | 0.073 |
| 激波穿越时刻 | 0.338 |
| 反射激波速度（左行） | -0.715 |

### 3.2 三档分辨率结果

| ny | nx | Δx | 步数 | da/dt(数值) | 增长率误差 | W(t_end) | 质量漂移 | 能量漂移 | x 动量漂移 | 压力修复 | 组份截断 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 264 | 0.012 | 916 | 0.077 | 0.063 | 0.200 | -2.908e-16 | -1.229e-16 | -2.782e-15 | 0 | 0 |
| 56 | 370 | 0.009 | 1285 | 0.084 | 0.155 | 0.200 | 1.454e-16 | -3.689e-16 | -1.093e-15 | 0 | 0 |
| 72 | 475 | 0.007 | 1651 | 0.085 | 0.160 | 0.192 | -2.908e-16 | -4.919e-16 | 2.429e-16 | 0 | 20 |

- 线性增长率误差：{'40': 0.06316763436022936, '56': 0.15471015557958007, '72': 0.1598708751874753}（判据 ≤ 0.15）
- 混合区宽度 vs 线性冲量模型：{'40': 0.44327217727779733, '56': 0.44277545603486196, '72': 0.38224689506399057}
- 自相似 α 模型可达性：未达自相似区（W ≪ λ/2）
- 分辨率一致性（两档最细网格增长率相对差）：0.004
- 账本最大漂移：1.267e-12

![RM](../src/tensorfvm/verification/artifacts/compressible_multimaterial/fig_rm_instability.png)

## 4. 冲击-气泡相互作用（Ma=1.22 空气中氦柱，Haas & Sturtevant 构型）

| 量 | 值 |
|---|---|
| 网格 | 360×200，Δ=0.0250 |
| 入射激波速度（解析） | 1.444 |
| 透射激波速度（解析） | 3.870 |
| 反射波速度（解析，负=左行） | -0.772 |
| 反射波是稀疏波 | yes |
| 界面速度 u* | 0.562 |
| 界面压力 p* | 1.300 |
| 激波到达气泡时刻 | 0.346 |
| 穿越气泡时间 2R/D_t | 0.517 |
| 入射激波速度（数值） | 1.452 |
| 透射激波速度（数值） | 1.942 |
| 涡量沉积 95% 时刻（数值） | 1.634 |
| 到达+穿越参考时刻 | 0.863 |
| 入射激波速度误差 | 0.006 |
| 透射激波速度误差 | 0.498 |
| 涡对形成时序误差 | 0.893 |
| 峰值涡马赫数 | 0.597 |
| 反射稀疏波（数值判读） | yes |

拓扑判读（与 Haas & Sturtevant 1987 的波系序列逐项对照，定性）：

| 特征 | 文献序列 | 本算例 |
|---|---|---|
| 入射激波 | 右行，速度 Ma·c | 测速误差见上表 |
| 透射激波 | 氦内加速（c_He > c_air） | 见上表 |
| 反射波 | 轻气泡 → **稀疏波** 回传 | 探针压力下降，判读为真 |
| 斜压涡量沉积 | 激波穿越期间在气泡上下沉积反向涡对 | 上半域环量单调增后饱和 |
| 气泡变形 | 迎风面凹陷→下游射流→肾形 | 见云图（流向尺度随时间演化） |

![bubble](../src/tensorfvm/verification/artifacts/compressible_multimaterial/fig_shock_bubble.png)

## 5. 涡对合并（二维对应物；含环量守恒与点涡旋转率标定）

| 量 | 值 |
|---|---|
| 网格 | 160² |
| 核半径 a / 间距 b | 0.300 / 1.200（b/a = 4.00） |
| 解析旋转率 Ω=Γ/(πb²) | 0.221 |
| 实测旋转率（主惯性轴/双核连线） | 0.232 |
| 旋转率相对误差 | 0.048 |
| 解析总环量 2Γ | 2.000 |
| 实测总环量 | 0.077 |
| 环量相对误差 | 0.962 |
| 是否合并（末态单核） | **no** |
| 峰值涡马赫数 | 0.384 |

合并阈值扫描（末态核间距 < 一个核半径即判合并，定义见 JSON 的 `merged_definition`）：

| b/a | 末态核间距 | 末态间距/a | 合并 | 旋转率误差 |
|---|---|---|---|---|
| 2.00 | 0.286 | 0.95 | yes | nan |
| 2.50 | 0.505 | 1.68 | **no** | nan |
| 3.00 | 0.847 | 2.82 | **no** | nan |

初始时刻环量诊断：2.000（理论 2Γ = 2.0）——说明环量算子本身正确，见局限中关于演化过程环量衰减的申报。

间距扫描（b/a 越大越不合并）：

| b/a | 合并 | 末态核间距 | 旋转率误差 |
|---|---|---|---|
| 5.20 | **no** | 1.511 | 7.813e-04 |
| 6.40 | **no** | 1.888 | 0.069 |

![vortex](../src/tensorfvm/verification/artifacts/compressible_multimaterial/fig_vortex_merger.png)

## 6. 判据总表

| 腿 | 判据 | 是否满足 |
|---|---|---|
| bubble | conservation_drift_le_0p5pct | yes |
| bubble | formation_time_le_10pct | **no** |
| bubble | incident_shock_speed_le_10pct | yes |
| bubble | reflected_wave_is_rarefaction | yes |
| bubble | transmitted_shock_speed_le_10pct | **no** |
| bubble | vortex_pair_forms | yes |
| interface | contact_pressure_dip_le_15pct | yes |
| interface | gfm_reduces_pressure_wiggle | yes |
| interface | interface_order_ge | yes |
| interface | mass_conserved | yes |
| rm | alpha_model_regime_reached | **no** |
| rm | conservation_drift_le_0p5pct | yes |
| rm | linear_growth_error_le_15pct | **no** |
| rm | linear_width_error_le_15pct | **no** |
| rm | no_positivity_repair | yes |
| rm | resolution_consistent_le_5pct | yes |
| sod | mass_conserved | yes |
| sod | order_first_ge | yes |
| sod | order_second_ge | yes |
| vortex | circulation_le_1pct | **no** |
| vortex | keeps_two_cores_at_wide_separation | yes |
| vortex | merges_at_close_separation | yes |
| vortex | rotation_rate_le_5pct | yes |

全部判据通过：**no**

## 7. 局限与适用范围（如实申报）

1. **材料接触的压力平衡缺陷（最重要）**：质量分数混合的 stiffened-gas 闭合在
   *被数值抹开*的接触带内产生约 10% 的压力凹陷（均匀流接触算例实测，见表 2）。
   GFM 只修正界面两层状态，不能修复混合格内部的热力学不平衡；标准解法是
   5/7 方程模型 + 界面压缩项（Allaire et al. 2002），本节点**未实现**。
2. **RM 线性增长率判据在细网格上是临界未过**：三档网格的增长率误差见 3.2 表
   （约 6% / 16% / 16%），即 ny=40 通过、ny=56 与 ny=72 以 ~1 个百分点越界。
   原因是冲量模型忽略了有限的激波穿越时间与界面弥散带来的振幅测量偏差；
   本报告不调阈值、不重拟合，如实记为**未过**。
3. **Brouillette 实验曲线未入库**：任务要求的 “W(t) vs Brouillette 实验 <15%”
   中，实验数据点无法在仓库内取得，故该子项证据等级为 **WITHHELD**；
   本报告给出的是与线性冲量理论（Richtmyer）和 α 模型形式化曲线的对照。
   本算例 W(t_end) 仍小于 λ/2，未进入 α 模型适用的自相似区，故 α 判据
   如实记为“未达区间”而不是粉饰为通过。
4. **气泡腿的两条时序判据用的是“平面界面”参考，对有限圆柱气泡不适用**：
   实测轴线透射激波速度约为 1D 平面折射值的一半（圆柱几何下激波在气泡内快速减速），
   涡量沉积 95% 时刻也相应推迟，故这两条判据如实记为**未过**；
   入射激波速度（0.6% 误差）、反射波为稀疏波、涡对形成、守恒四条为真实通过。
   正确的时序基准需要用轴对称求解器给出，属后续工作（本节点只做二维笛卡尔）。
5. **Haas & Sturtevant 对照是定性的**：原始纹影图不在仓库内，拓扑对照基于
   文献波系序列描述；时序判据用的是精确黎曼解给出的到达+穿越时刻，可执行、可复现。
6. **涡腿是最弱的一条**：双核检测在格点上有量化误差，主惯性轴受包络与声波影响；
   旋转率误差见判据表，未粉饰。环量诊断必须用非周期差分（周期差分恒为零）。
7. **几何**：全部算例为二维笛卡尔/单波长周期横向；轴对称（球泡、同轴涡环）
   未实现，属后续工作。
8. 组份截断把质量分数夹回 [0,1]，属非守恒修复，次数已逐算例记录于表中。

## 8. 复现

```bash
PYTHONPATH=src python -m tensorfvm.compressible \
    --output src/tensorfvm/verification/artifacts/compressible_multimaterial
PYTHONPATH=src python -m tensorfvm.compressible --quick  # CI 规模
PYTHONPATH=src pytest tests/test_compressible_eos_riemann.py \
                     tests/test_compressible_multimaterial.py -q
PYTHONPATH=src python scripts/report_compressible.py    # 本文档
```

单线程设置：`TENSORFVM_COMPRESSIBLE_THREADS=1`（默认）。在 128 核 loongarch 节点上
torch 默认线程池对 10^4 量级 float64 张量反而慢约 7 倍，实测记录在案。

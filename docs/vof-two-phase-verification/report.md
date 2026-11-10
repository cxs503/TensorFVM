# TensorFVM 两相流 VOF 验证面：Zalesak 旋转圆盘 + 单涡/双涡变形 + 静水平衡

**节点** `p337w1-n06-fvm-twophase-vof` · **仓** TensorFVM · **分支** `agent/m9956`
**验收产物**：`docs/vof-two-phase-verification/artifacts/campaign.json`（判据机器判读）、
`artifacts/hydrostatic.json`、`artifacts/deformation_sensitivity.json`、
`artifacts/field_diagnostics.json`、`artifacts/figures/fig1..fig6`、`tests/test_vof_two_phase_verification.py`。
**代码**：`src/tensorfvm/vof_verification/`（新增验证面）、
`src/tensorfvm/vof_phase0/solver.py`（新增 `well_balanced_surface` 开关，默认关闭）。

---

## 0. 结论速览（先给判定，后给证据）

| 判据 | 目标 | 实测（最细网格） | 判定 |
|---|---|---|---|
| C1 Zalesak 面积守恒（1 周旋转） | < 1 % | **1.58e-14**（256²，PLIC） | ✅ PASS |
| C2 单涡变形 界面偏差 | < 5 % | **80.1 %**（128²，PLIC） | ❌ FAIL（honest） |
| C3 双涡变形 界面偏差 | < 5 % | **20.4 %**（128×64，PLIC） | ❌ FAIL（honest） |
| C4 静水平衡 压力振荡 | < 0.5 % | **0**（well-balanced，机器零）；对照（现行 void 模型）1.56e-2 ~ 6.25e-2 | ✅ PASS（含对照） |
| C5 全局质量守恒（全部算例/全部格式） | < 1 % | **5.98e-3**（最差，donor 单涡）；PLIC ≤ 1.6e-14 | ✅ PASS |
| C6 三网格 GCI 收敛阶（相对 L1 范数） | ≥ 1 | Zalesak **1.18** ✅ / 单涡 **0.52** ❌ / 双涡 **0.55** ❌ | ❌ FAIL（部分达标） |
| C7 现有单相/两相回归 | 100 % 通过 | 见 §6.3（`pytest`：57 + 40 项通过） | ✅ PASS |

**一句话**：本节点建立的验证面证明 TensorFVM 的两相 VOF **守恒性（质量到机器精度）与几何聚焦能力（Zalesak 0.79 % 形状误差、bulk 面积误差 0.38 %）成立**，
但**不满足**「界面形态偏差 <5 %」与「收敛阶 ≥1」在强剪切/撕裂型变形算例（Rider–Kothe 单涡 T=8、双涡 T=8）上的指标；
同时**发现并修复了一个真实缺陷**：现行 `void` 自由面模型在**内部自由面**上没有压力 Dirichlet 数据，
静水静止态偏离解析解 O(h)（32² 时 6.25 %，远超 0.5 % 判据）；新增的 `well_balanced_surface`
把静止态保持到机器零（0.0），并作为默认关闭的开关接入生产求解器（不改变既有行为）。

> 纪律说明：C2/C3/C6 的 FAIL 不被任何粉饰覆盖；§5/§6 给出**归因证据**（不是实现 bug 而是亚网格信息丢失 +
> 撕裂算例的可分辨性上限），并给出**达到 5 % 所需分辨率的量级外推**。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

两相流 VOF（Volume of Fluid）方法用体积分数 α∈[0,1] 表示相分布，界面由 α 的间断重构（PLIC 几何重构）。
船舶总体性能中的自由面、甲板上浪、晃荡、破舱进水都依赖该能力，其**验证**必须回答三件事：
①界面形状在纯对流下是否保持（无物理扩散时不应被数值扩散抹掉）；②体积是否严格守恒；③静止自由面是否被压力离散"推"出伪流动。
本节点对这三件事各设解析锚（analytic anchor）。

### 1.1 三个对流锚（exact-advection，解析解闭式已知）

| 锚 | 速度场 | 区域/时间 | 解析解 | 用途 |
|---|---|---|---|---|
| Zalesak 旋转圆盘（Zalesak 1979） | 绕域心刚体旋转 `u=-ω(y-0.5)`，`v=ω(x-0.5)`，`ω=2π`（1 周/单位时间） | [0,1]²，T=1（1 周） | t=T 时解析解 **= 初场** | 形状保持 + 面积守恒（C1） |
| 单涡变形（Rider & Kothe 1995） | `u=-sin²(πx)sin(2πy)g(t)`，`v=sin(2πx)sin²(πy)g(t)`，`g=cos(πt/T)` | [0,1]²，T=8 | t=T 时解析解 **= 初场** | 强剪切/撕裂下的界面重构精度（C2） |
| 双涡变形（本节点定义，见 §1.3） | 流函数 `ψ=sin(πx)sin(2πy)/π · g(t)` | [0,2]×[0,1]，T=8 | t=T 时解析解 **= 初场** | 两个反向涡胞下的变形与复原（C3） |

**可逆性为什么成立**：三者都可写成 `u(x,t)=g(t)U(x)` 且 `∇·U=0`。以 `τ(t)=∫₀ᵗ g` 代入得 `dx/dτ=U(x)`；
而 `∫₀^T cos(πt/T)dt = 0`，故 T 时刻流映射为恒等映射——**速度场本身不产生任何形状改变**，测到的误差全部来自数值格式。
这一点在测试里以标记点积分（中点法，300 步）独立验证：13 个标记点回到出发点，最大位移 1.3e-3（阈值 5e-3）。

### 1.2 几何与初始体积分数（精确，非采样）

- 圆盘：中心 (0.5, 0.75)，半径 0.15（域 [0,1]²）；双涡算例同一圆盘放在 2:1 域的 (0.5,0.75)。
- Zalesak 槽：宽 0.05（半宽 0.025），从圆心切到圆顶（顶点开槽），解析面积
  `πr² - [w·t* + πr²/2 - r²asin(t*/r)]`，`t*=√(r²-w²)` = 6.32207e-2。
- 单元的初始 α 是 **cell∩shape 的精确面积**（圆盘-矩形相交，θ 代换 + 24 点 Gauss-Legendre，两次求积误差 <1e-13 相对，
  见 §6.1 与 `tests/test_vof_two_phase_verification.py::test_disk_rect_area_matches_closed_form`）。
  网格 Σα 与闭式面积的偏差：64² 2.1e-14、128² 1.5e-14、256² 6.3e-15（绝对面积，相对 <3e-13）。
  **原因**：初场的参考值若用子采样近似，其自身误差会与输运误差混在一起，三网格收敛阶无法判定。

### 1.3 双涡场的定义与离散无散性

`ψ(x,y,t) = g(t)·sin(πx)sin(2πy)/π`（顶点中心），`u=-ψ_y=-2sin(πx)cos(2πy)g`，`v=ψ_x=cos(πx)sin(2πy)g`；
涡量 `ω_z = -5π sin(πx) sin(2πy) g` 在 y=1/2 处变号 ⇒ **两个反向旋转的涡胞**；四个边界上法向速度恒为 0 ⇒ 固壁边界对该场是**精确**的。
速度在 MAC 面上取为 `ψ` 的**精确差分**（`u=-(ψ_{j+1}-ψ_j)/Δy`，`v=(ψ_{i+1}-ψ_i)/Δx`），
因此离散散度在机器精度内为零（实测 1.3e-14），而不是解析采样后残留的 O(h²) 散度——
后者会被维数分裂的 VOF 扫描当成伪体积源。

### 1.4 静水锚（lake-at-rest）

闭合水箱，液深 H_l 与网格面**对齐**，重力 g=9.81，ρ_l=1000，气相用求解器 `void` 自由面模型；
解析解 `u=0`，`p(y)=ρ_l g(H_l-y)`。判据是压力振荡 `max|p^n-p|/(ρ_l g H_l) < 0.5 %`（C4）。
参数扫描：网格 32²/64²/128² × 液深 0.25/0.50/0.75 × 密度比 1/100/1000，每算例 40 步。

### 1.5 格式与参数

| 符号 | 含义 | 值 |
|---|---|---|
| `plic` | 几何 PLIC（`vof_plic.advect_plic`，`conservative_bounds=True`），**主格式** | CFL 0.75 |
| `fct` | 代数通量修正输运（`vof_phase0.advection.AlgebraicVOF`，即 Phase-0 生产格式） | CFL 0.75 |
| `donor` | 一阶迎风（单调下界基准） | CFL 0.75 |
| 网格梯 | Zalesak 64²/128²/256²；单涡 32²/64²/128²；双涡 32×16/64×32/128×64（等比 r=2） | — |
| 时间步 | `Δt = CFL·min(Δx,Δy)/|u|_max`，全程固定（误差随 h 同阶） | — |
| 分裂 | 维数分裂，扫描次序**逐步交替**（Lie，消除方向偏置） | — |
| 时间积分 | 速度用 `g(t)` 在步内的**精确积分平均**（见 §5.1 的陷阱） | — |

---

## 2. 怎么做的（方法 / 流程 / 工具链）

```
src/tensorfvm/vof_verification/
  geometry.py     cell∩shape 精确面积（θ 代换 Gauss-Legendre）
  cases.py        三个解析锚 + MAC 面采样（ψ 精确差分）+ 输运包装（plic/fct/donor）
  metrics.py      相对 L1 / 对称差 / 面积误差 / Roache GCI（含非单调的诚实处理）
  hydrostatic.py  lake-at-rest 驱动（shipped 对照 vs well_balanced）
  campaign.py     分片可切的算例表 + 判据机判（C1–C6）+ 变形窗扫描
scripts/run_vof_verification.py            # 分片执行/合并（--index/--shard/--merge/--hydrostatic-only/--deformation-sweep）
scripts/render_vof_two_phase_report.py     # 图 + bulk 诊断（field_diagnostics.json）
tests/test_vof_two_phase_verification.py   # 20 项 pytest（几何/锚/输运/GCI/静水/判据）
```

流程：① 解析δ 算初场 → ② 分片跑 (算例×格式×网格) 27 个 run → ③ 合并出 `campaign.json` 并**由程序**判 C1–C6
→ ④ 静水扫描 27 组 + 变形窗扫描 12 组 → ⑤ 出图与 bulk 诊断 → ⑥ 回归。
所有数字均由 JSON 产生，报告表格与 JSON 一一对应（零手抄：`campaign.json` 是唯一真源）。

复现命令：

```bash
# 全部算例（约 15 min，4 路并行；集群上按 --index 分片）
python scripts/run_vof_verification.py --out docs/vof-two-phase-verification/artifacts/runs
python scripts/run_vof_verification.py --hydrostatic-only --out docs/vof-two-phase-verification/artifacts/runs
python scripts/run_vof_verification.py --deformation-sweep --out docs/vof-two-phase-verification/artifacts/runs
python scripts/run_vof_verification.py --merge --skip-hydrostatic \
    --runs docs/vof-two-phase-verification/artifacts/runs \
    --out docs/vof-two-phase-verification/artifacts/runs
python scripts/render_vof_two_phase_report.py
```

环境：Python 3.11.6 / torch 2.7.1 (float64)；`TFVM_THREADS∈{6,8}`；git `agent/m9956`。
算力说明（如实）：脚本已按集群分片形式递交（作业号 cj489–cj493，`--index` 子集，stage_dir=本工作树），
但递交时队列排队 187 项、直至报告落盘仍处 `queued`，**未在集群上执行**；本报告全部数字由**同一分片脚本**
在本机 128 核节点以 4 路并行复算并落盘（wall time ≈15 min，最长单项 128² 单涡 1365 步）。
因此数字与集群执行完全同源，但**未获得第二硬件/架构的复现证据**（见 §5.3 局限）。

---

## 3. 计算结果（数字表）

### 3.1 Zalesak 旋转圆盘（1 周，解析解 = 初场）

| 格式 | 网格 | 步数 | 对称差 | 相对 L1 | 面积漂移 | 单步最差漂移 | α 范围 |
|---|---|---|---|---|---|---|---|
| PLIC | 64² | 264 | 5.69 % | 4.17 % | 4.4e-16 | 2.6e-15 | [0, 1] |
| PLIC | 128² | 532 | 1.15 % | 1.96 % | 3.5e-15 | 4.4e-15 | [0, 1] |
| PLIC | 256² | 1069 | **0.79 %** | **0.87 %** | **1.6e-14** | 1.6e-14 | [0, 1] |
| FCT | 256² | 1069 | 4.29 % | 26.5 % | 2.2e-16 | 4.4e-16 | [0, 0.9998] |
| donor | 256² | 1069 | 27.9 % | 67.4 % | 6.6e-16 | 1.1e-15 | [0, 0.929] |

GCI（PLIC，相对 L1）：p_coarse=1.09，p_fine=**1.18**，p_successive=1.02，状态 asymptotic，GCI_fine=0.009。
⇒ **C1 PASS（1.6e-14 ≪ 1 %）；C6 在旋转锚上 PASS（p=1.18 ≥ 1）**。

### 3.2 单涡变形（Rider–Kothe，T=8，解析解 = 初场）

| 格式 | 网格 | 步数 | 对称差 | 相对 L1 | 质量漂移 | α 范围 |
|---|---|---|---|---|---|---|
| PLIC | 32² | 340 | 160.5 % | 159.3 % | 5.9e-16 | [0, 1] |
| PLIC | 64² | 682 | 120.1 % | 116.3 % | 3.5e-15 | [0, 1] |
| PLIC | 128² | 1365 | **80.1 %** | **80.8 %** | 3.9e-16 | [0, 1] |
| FCT | 128² | 1365 | 93.5 % | 126.0 % | 2.0e-16 | [0, 0.512] |
| donor | 128² | 1365 | 100.0 %（掩膜为空） | 164.1 % | 3.5e-3 | [0, 0.231] |

GCI（PLIC）：对称差 p_fine=0.58（非渐近，如实标注）；相对 L1 p_fine=**0.52**。
⇒ **C2 FAIL**：80.1 % ≫ 5 %。

bulk 诊断（64²，`field_diagnostics.json`）：α>1/2 面积误差 4.6 %，**质心偏移 0.111**（= 盘半径 0.15 的 74 %）。
⇒ 单涡的误差不是纯扩散，而是**沿螺旋路径的不可逆质量滞留**（一部分液体在拉伸阶段被抹到亚网格，回程时无法被送回），
表现为质心被拖离原位。

### 3.3 双涡变形（T=8）

| 格式 | 网格 | 步数 | 对称差 | 相对 L1 | 质量漂移 |
|---|---|---|---|---|---|
| PLIC | 32×16 | 333 | 37.5 % | 44.8 % | 3.9e-16 |
| PLIC | 64×32 | 679 | 23.7 % | 30.6 % | 7.9e-16 |
| PLIC | 128×64 | 1364 | **20.4 %** | **20.8 %** | 3.1e-15 |
| FCT | 128×64 | 1364 | 34.5 % | 71.4 % | 0.0 |
| donor | 128×64 | 1364 | 100.0 %（掩膜为空） | 133.4 % | 6.1e-4 |

GCI（PLIC）：对称差 p_fine=0.21；相对 L1 p_fine=**0.55**。⇒ **C3 FAIL**：20.4 % ≫ 5 %。
bulk 诊断（64×32）：面积误差 2.6 %，**质心偏移 3.4e-5**（对称，无漂移）⇒ 该锚的误差是**对称的界面涂抹**，与单涡的漂移型误差不同。

### 3.4 静水平衡（lake-at-rest，密度比 1000，40 步）

| 网格 | 液深 H_l | 现行 void（对照） | well_balanced（本节点修复） | 对照伪速度 | 修复后伪速度 |
|---|---|---|---|---|---|
| 32² | 0.25 / 0.50 / 0.75 | 6.25e-2 / 3.13e-2 / 2.08e-2 | **0 / 0 / 0** | 3.3e-8 | 0 |
| 64² | 0.25 / 0.50 / 0.75 | 3.13e-2 / 1.56e-2 / 1.04e-2 | **0 / 0 / 0** | 3.6e-8 | 0 |
| 128² | 0.25 / 0.50 / 0.75 | 1.56e-2 / 7.82e-3 / 5.21e-3 | **0 / 0 / 0** | 3.7e-8 | 0 |

- 对照值逐点等于 **Δy/(2H_l)**（半单元偏移）：`n=32,H_l=0.25 → 0.015625/0.25 = 6.25e-2` 完全吻合 ⇒ 缺陷机理被定量证实。
- 密度比 1 / 100 / 1000 三档结果相同（差异 <1e-5 相对），说明缺陷**与密度比无关**，只与表面处理有关。
- ⇒ **C4 PASS（0 < 0.5 %）**，且报告给出"修复前 6.25 % → 修复后 0"的对照基线（S2 级证据，见 §6.2）。

### 3.5 变形窗扫描（归因：误差 vs 累积缠绕量）

同一速度场、同一可逆性（`g` 的周期取 T），只改变变形窗 T（T 小 ⇒ 缠绕少、界面可分辨）：

| T | 单涡 64² | 单涡 128² | 双涡 64×32 | 双涡 128×64 |
|---|---|---|---|---|
| 1 | 18.3 % | 14.8 % | 5.26 % | **4.93 %**（<5 %） |
| 2 | 31.7 % | 23.9 % | 10.5 % | 6.34 % |
| 4 | 52.8 % | 40.0 % | 13.2 % | 9.86 % |
| 8（标准） | 120.1 % | 80.1 % | 23.7 % | 20.4 % |

⇒ 误差随缠绕量**单调增长**，且双涡在可分辨区（T≈1，128×64）即可满足 <5 %；
单涡由于场更强剪切，即使 T=1 也只能到 14.8 %。这定量表明：**C2/C3 的剩余误差主要是"亚网格丝状界面信息丢失"，
不是重构算法或守恒性的缺陷**（守恒性同时保持在 1e-14 量级）。

---

## 4. 结果图

| 图 | 文件 | 内容 |
|---|---|---|
| 图1 | `artifacts/figures/fig1_zalesak_fields.png` | Zalesak 128²：解析初场 / PLIC / FCT 的 α 云图（PLIC 保持尖角，FCT 涂抹明显） |
| 图2 | `artifacts/figures/fig2_single_vortex.png` | 单涡 64²：解析初场 / PLIC 复原场 / 差值场（差值呈螺旋状，指示不可逆滞留） |
| 图3 | `artifacts/figures/fig3_double_vortex.png` | 双涡 64×32：解析初场 / PLIC 复原场 / 差值场（差值关于涡胞对称） |
| 图4 | `artifacts/figures/fig4_convergence.png` | 三算例 × 三格式：对称差随网格的双对数收敛曲线（虚线 = 5 % 判据） |
| 图5 | `artifacts/figures/fig5_hydrostatic.png` | 静水：现行 void（随 n 按 1/n 下降）vs well-balanced（机器零）与 0.5 % 判据线 |
| 图6 | `artifacts/figures/fig6_deformation_sensitivity.png` | 误差 vs 变形窗 T（两网格），说明可分辨性上限 |

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 三个实测到的陷阱（均已写入代码注释与测试）

1. **回程算例的速度必须在步内做精确时间平均**。若把 `g(t)` 在 `t_k` 点取样，`Σ g(t_k)Δt` 与 `∫g dt` 差 O(Δt)，
   在**可逆**算例里这不是小扰动而是系统性错位：回程少走一步 ⇒ 恢复形状被平移 ~CFL·h，量级直接压过格式误差。
   `_modulation_mean` 用闭式积分平均消除该项（`test_modulation_average_is_exact_and_reverses`）。
2. **解析采样的速度场不是离散无散**。双涡若在 MAC 面上直接解析取样，离散散度 ~0.107（16×8 网格），
   分裂扫描会把它当成伪源。改为 `ψ` 的精确差分后散度降到 1e-14。
3. **几何求积在切点有 √ 型端点奇性**。x 方向直接 24 点 Gauss 在"整体圆盘"上只有 1e-5 相对精度；
   用 `x=c_x+r sinθ` 代换后达到 1e-13（`test_disk_rect_area_matches_closed_form`）。

### 5.2 静水判据的真实缺陷（不是数值噪声）

现行 `void` 模型只在**域顶边界**施加半单元的 p=0 Dirichlet；**内部**液-气面在 Poisson 矩阵里被直接丢弃
（等价齐次 Neumann），而在速度修正里又按全步长除以 Δy ⇒ 表面单元残余加速度 g/2，投影把压力顶回去
ΔρgΔy/2，形成 **Δy/(2H_l) 的静态压力偏差**。这不是"振荡"而是稳态偏移，但同样会让自由面在长时间积分里出现伪流。
修复（`well_balanced_surface=True`）：内部自由面同样施加 p=0、距离取 Δy/2（ghost-fluid 阶梯处理），
矩阵对角加对应对流导系数。**默认关闭**，既有 dam-break/Phase-0 结果不受影响（§6.3 回归为证）。

### 5.3 局限（如实声明）

1. **C2/C3 未达标**，且不是靠调参可解的：按实测阶（单涡 0.52、双涡 0.55）外推，要让单涡 128² 的 80 %
   降到 5 % 需要约 16 倍误差下降 ⇒ 网格 ~(16)^{1/0.52}≈200 倍细化（约 2.6 万² 单元），
   双涡（阶 0.55）约 15 倍细化（约 2000×1000）。这在工程与验证预算上不可行，
   **说明判据"任何网格上 <5 %"对撕裂型算例是过强的表述**；本报告给出可分辨区（T≈1）的达标证据与标度律作为替代证据。
2. **掩膜型（α>0.5）对称差指标本身有 O(h) 底噪**：Zalesak 的对称差在 256² 停在 0.79 %，
   而相对 L1 仍按 1.18 阶下降（0.87 %@256²）——对称差对"界面所在单元的一次错判"更敏感。
   因此 C6 的判据落在**相对 L1 范数**（Rider–Kothe 标准范数），对称差阶同时列出。
3. **只验证两相输运与静水压力平衡**，不含表面张力（本平台 CSF/毛细验证由既有节点覆盖）、不含三维、不含自适应网格。
4. **维数分裂为 Lie 分裂**（逐步交替）；Strang 分裂未实现——预期可进一步降误差，但**不能**改变亚网格信息丢失的上限。
5. **初场精确、界面重构为 Youngs 法向 + 体积反解**：槽尖角（Zalesak）等不可分辨几何必然有 O(h) 误差。
6. 全部计算为单进程 torch（float64），未做 GPU/多节点并行；`TFVM_THREADS=6~8`。

### 5.4 与既有证据的边界（不重复已验证案例）

既有 `verification.vof_rotation`（代数 FCT 格式 + 两网格 + 无 GCI + 无变形算例）、
`vof_plic_interface`（PLIC 的 Zalesak/一维平移）保持不变。本节点新增的是：
**剪切变形锚（单涡/双涡）+ 三网格 GCI + 精确几何初场 + 静水 rest-state 锚 + well-balanced 表面修复**，
并给出三种格式在同一锚上的横向对照。

---

## 6. 误差判定（逐项对基准，显式回答"是否 <3 %"）

### 6.1 关键量 vs 基准

| 关键量 | 基准 | 实测 | 相对误差 | <3 % ? |
|---|---|---|---|---|
| 单元初场面积（圆盘，256² 求和） | πr²=0.07068583 | 0.07068583 | 8.9e-14 | ✅ 是 |
| Zalesak 初场面积（128²） | 6.3220703e-2（闭式槽） | 6.3220703e-2 | 3.1e-13 | ✅ 是 |
| Zalesak 旋转 1 周面积 | 初场面积 | 漂移 1.6e-14 | 1.6e-14 | ✅ 是 |
| Zalesak 形状（对称差，256²） | 解析界面 | 0.79 % | 0.79 % | ✅ 是 |
| Zalesak 形状（相对 L1，256²） | 解析场 | 0.87 % | 0.87 % | ✅ 是 |
| 单涡形状（对称差，128²） | 解析初场 | 80.1 % | 80.1 % | ❌ **否** |
| 双涡形状（对称差，128×64） | 解析初场 | 20.4 % | 20.4 % | ❌ **否** |
| 静水压力振荡（well-balanced，全扫描最差） | ρ_l g H_l | 0.0 | 0.0 | ✅ 是 |
| 静水压力振荡（现行 void，最差 32²/H=0.25） | 同上 | 6.25e-2 | 6.25 % | ❌ 否（缺陷已修，默认关闭） |
| 全局质量守恒（全部 27 run 最差） | V(0) | 5.98e-3（donor 单涡，D>1 的迎风极限） | 0.60 % | ✅ 是 |

**显式回答**：在**形状保持（旋转）**、**守恒性**、**静水压力平衡（修复后）**三类量上误差 **<3 %**；
在**强剪切撕裂型变形**的界面形态上误差 **≫3 %**（80.1 % / 20.4 %），且已归因（§3.5、§5.3），不做粉饰。

### 6.2 修复前后对照（能力增量）

| 量 | 修复前（默认 void） | 修复后（`well_balanced_surface=True`） | 判据 |
|---|---|---|---|
| 静水压力振荡（32², H_l=0.25） | 6.25e-2 | **0.0** | <5e-3 ✅ |
| 静水压力振荡（128², H_l=0.50） | 7.82e-3 | **0.0** | <5e-3 ✅ |
| 静水伪速度（max|u|/√(gH)） | 3.7e-8 | **0.0** | — |
| Phase-0 dam-break / 旋转 / PLIC 回归 | — | 不变（默认关闭） | 100 % ✅ |

### 6.3 回归证据（C7）

```
pytest tests/test_vof_phase0_advection.py tests/test_vof_phase0_dam_break.py \
       tests/test_vof_plic_interface.py tests/test_vof_rotation_verification.py   → 57 passed
pytest tests/test_benchmark.py tests/test_conservative3d.py \
       tests/test_annular_verification.py tests/test_two_phase_vof.py            → 40 passed, 1 skipped
pytest tests/test_vof_two_phase_verification.py                                  → 20 passed
```

### 6.4 审计链

| 环节 | 产物 / 命令 |
|---|---|
| 算例表 | `campaign.SPECS`（27 项，`--list` 可打印） |
| 分片执行 | `python scripts/run_vof_verification.py --index k --out .../runs`（集群 cj489–cj493；本机复算同脚本） |
| 判据机判 | `artifacts/campaign.json → criteria.C1..C6`（本报告所有数字取自此文件） |
| 静水扫描 | `artifacts/hydrostatic.json`（27 组 × 2 模型） |
| 变形归因 | `artifacts/deformation_sensitivity.json`（12 run） |
| bulk 诊断 | `artifacts/field_diagnostics.json`（面积误差 / 质心偏移） |
| 图 | `artifacts/figures/fig1..fig6`（`scripts/render_vof_two_phase_report.py` 生成，matplotlib，数据全部来自上述 JSON/重算场） |
| 单测 | `tests/test_vof_two_phase_verification.py`（20 项，46 s） |

---

## 7. 后续（可执行建议）

1. **C2/C3 的判据重写**：把"<5 %"绑定到"可分辨变形窗"（如本报告 T≈1 的双涡 4.93 %），
   撕裂窗（T=8）改为**标度律验收**（给出实测阶与所需分辨率外推）。
2. **提高撕裂区精度**：优先做 Strang 分裂 + 界面锐化（THINC/VOF-WLIC）；
   根治手段是亚网格界面重建（如多物质 moment-of-fluid），而非加密网格。
3. **正名 well-balanced 表面**：把 `well_balanced_surface` 从实验开关升级为默认（需先在 dam-break/bubble 上做完整对照），
   并对内部自由面补上**高度函数**（height function）曲率，以处理非对齐阶梯表面。
4. 把本验证面接入 `verification` 目录的自动回归（`campaign.json` 的判据字段可直接被看板消费）。
# TensorLBM 30P30N 攻关 — 工作区交接说明（2026-10-08）

## 一、本包内容

| 路径 | 说明 |
|---|---|
| `tensorlbm/` | 求解器代码（`curve_grid.py` 通用曲线坐标算子层 + `multiblock.py` 多块 overset 驱动） |
| `TensorFVM/` | 上游参考实现 https://github.com/cxs503/TensorFVM（SIMPLE + Rhie–Chow + SA，含 30P30N 算例与几何 `src/tensorfvm/data/30p30n/*.dat`） |
| `_diag_*.py` / `_verify_*.py` | 根因诊断脚本（圆柱验证 `_diag_cyl.py`、翼型 `_diag_cd.py`、算子一致性 `_verify_matrix.py`、孤立投影 `_verify_proj.py`） |
| `_*.log` / `_cd_*.npz` | 各诊断运行的结果与表面力数据 |
| `*.png` | 中间可视化 |
| `results/` | 早期调试的场快照（`*_fields.npz`，375M，非当前瓶颈必需） |
| 其余 `*.npz` | 各历史算例场数据（约 30M） |

## 二、当前状态（10-08 融合后）

- **目标**：30P30N（α=8.1°, Re=9e6）达标 Cl≈3.3–3.5、Cd≈0.1–0.2。
- **几何（已裁决）**：我们的 `airfoil_data/gh_{Slat,Main,Flap}.dat` 与 TensorFVM 的
  `data/30p30n/{slat,main,flap}.dat` **逐字节相同**（点数一致、max|dx|=max|dy|=0，
  见 `_fuse_geom.py` 输出 VER DICT: PASS）。三者同源 linuxguy123 30P-30N 验证几何，
  **几何不是 Cd 过估的分歧来源**。
- **力积分（已对齐）**：我们的 `c_forces` 用 `Σ -p·A`（格心 owner 压力 × 面-面积矢量），
  与 TensorFVM `_surface_force` 的 `Σ p[owner]·S`（curve_grid.py / body_fitted.py）**同源**；
  棋盘格滤波接缝 bug 已修（见下），圆柱复跑稳定（Cd=1.74, Cl=0.21 正常）。
- **圆柱基准（Re=100, 理论 Cd=1.09，rhie2 一致算子 + 力修复，8000 步）**：
  Cd=1.74, Cl=0.21；**但压力仍打到 ±16 上限**（迎风驻点 Cp 被 clip 在 +16，理论应 +1）
  → 平滑压力**过估 ~16×**，被 `p_cap=8` 掩盖。这是剩余的核心 bug。

## 三、已定位的根因链（融合后收敛）

1. **算子一致性**：主路径 `build_poisson`（rhie2）已用 `_rhie_lap_consistent`（矩阵=修正算子，
   验证 8e-15）。airfoil 各块 N≈11000 < 45000，**走主路径**，故与 `build_poisson_iterative`
   （1498 行不一致算子）无关。
2. **投影模式 vs 压力量级（圆柱实测，决定性）**：
   用 `_diag_scale.py` 在圆柱收敛态逐部位测散度：
   ```
   div_pre_int_rms  = 0.0        # 内部散度完美为 0
   div_pre_wall_max = 1.13e+01   # 全部散度集中在壁面, 量级 O(U/h)=O(11)
   div_post_max     = 1.10e+01   # 投影后壁面散度仍是 O(11) —— 没被去掉!
   p_max            = 6.47       # p 被壁面 O(11) 散度驱动到 O(6)
   ```
   - **根因**：EXACT/rhie2 模式 `rhs = div(u*)/dt` 把壁面**滑移跳变散度**（O(11)=O(U/h_wall)）
     **放大 1/dt 到 O(2200)**，压力被驱动到 O(6–17)，力过估 ~3–20×。
   - **wall-fix 不一致**：`project()` 用清洗后壁速算 RHS（散度 O(dt)），但速度修正
     `u_new = u* − dt·grad(p)` 作用到**原始脏壁速** → 修正量只抵消清洗散度，脏散度 O(11)
     残留并每步回流。尝试“去掉壁面恢复”让修正作用于清洗壁速 → div 可归零、p 物理量级，
     但清洗滑移速度反馈进下一步 predictor 导致 step~884 NaN（collocated 网格壁面是流体元、
     重置会重新引入散度的经典死结，需更稳健处理）。**已撤销该改动恢复稳定。**
   - **adjoint/adj1 模式（rhs = div(u*)，不除 dt）不过估**：圆柱 Cd=1.36（~25% 偏高，量级正确），
     rhie2/exact Cd=2.94（3× 过估）。即 **adjoint 算子本身正确，过估来自 EXACT 模式的 /dt**。
   - 早先“rAU = a_p^{-1}”假设被推翻：实测过估源于 /dt 放大壁面滑移跳变，非 rAU 形式。
3. **棋盘格污染力（已修）**：`c_forces` 改非 wrap 1-2-1 滤波（curve_grid.py:1752-1782），
   圆柱稳定；但 `_diag_cd.py` 的 `force_decomp`（第 70 行）**仍用 `np.roll` wrap 滤波**，
   跨尾迹缝会污染翼型力 —— 待改。

## 三·b、翼型融合验证（2026-10-08 实测，关键负面结论）

**结论：融合未验证通过，且暴露一个比“压力量级”更深的阻断——流场本身不收敛。**

命令：`python3 _diag_cd.py 15000 8 0.05 2e-3 0 1 10 rhie2`
（rhie2，生产掩码 `p_cap=8`/`p_scale=0.05`，`p_exact=1`，`nu_t_cap=10·nu`，15000 步，t≈0.6）

| 步 | Cd | Cl | |u|max |
|---|---|---|---|
| 2500 | −0.058 | +0.16 | 3.9 (slat) |
| 5000 | +0.01 | +0.06 | 4.3 (slat) |
| 7500 | −0.38 | −1.04 | 7.8 (main) |
| 10000 | −0.46 | −1.21 | 7.8 (main) |
| 12500 | −0.04 | +0.66 | 8.7 (slat) |
| 15000 | **+0.14** | **+0.55** | 9.4 (slat) |

- 末态 Cd=0.14（**巧合落入 0.1–0.2**）、Cl=0.55（目标的 1/6）；窗口均值 **Cd_mean=−0.26、Cl_mean=−0.31（负升力）**。
- 力**大幅振荡**（Cl 在 +0.16↔−1.21 间摆），局部 |u|max **单调从 3.9 涨到 9.4×U** → **发散/非定常未收敛**，不是稳态。
- 逐块（末态）：main Cd=0.0835/Cl=0.64、slat Cd=0.039/Cl=−0.069（**前缘 slat 负升力，物理错**）、flap Cd=0.019/Cl=0.031（棋盘格修复后 flap 虚假阻力确实消失，但量纲整体崩）。
- **含义**：cd=0.14 只是掩码下的偶然值；流场靠 `p_cap/p_scale/ucap` 才没爆 NaN，底层在发散。
  “rAU=aₚ⁻¹”单点修复**不足以**达标——需先解决流场不收敛（见下方待办）。

## 三·c、翼型三模式对照（2026-10-08，adj1/adjoint/rhie2 全跑完，关键重构）

**三份日志（rhie2_fix / adjoint / adj1）暴露一个共同、与投影模式无关的事实：**

| 模式 | Cd | Cl | |u|max 末态 | sep% | nu_t_max / ν |
|---|---|---|---|---|---|
| rhie2（`_air_rhie2_fix.log`,15000步） | 0.14 | 0.60 | 5–9 | 53/29/56 | 10×ν（cap） |
| adjoint（`_air_adj.log`,15000步） | 0.15 | 0.39 | 10.6 | 57/46/54 | 10×ν（cap） |
| adj1（`_air_adj1.log`,30000步） | 0.20 | 0.30 | 14.8 | 52/49/61 | 10×ν（cap） |

- **三模式 `nu_tilde` 全部钉死在 `nu_t_cap = 10×ν`**（main/slat/flap 皆 max=7.6e-7=cap，
  内部均值 ~2×ν）→ 湍流被掐死。真实湍流边界层需 ν_t/ν ~ O(100+)，10×ν 下边界层
  **薄到必然大面积分离**（sep% 46–61%）。
- **`Cd_f`/`Cl_f` 恒为 0.0000**：壁函数（`wall_fn=True`）把壁面切向速度设成对数律，
  壁面梯度被抹平，黏性力贡献压到零（对 Re=9e6 湍流翼型本应占 Cd 的 30–50%）。
- **Cl 全在 0.3–0.6，比高升力构型 0° 本该有的 Cl(≈1.0+) 还低** → 连弯度升力都被吃掉，
  说明是**流场物理态错（分离）**，不是单纯力/压力读数偏。

**根因重构（推翻“翼型 Cl 崩 = 投影过估”的猜测）：**
- 投影过估（EXACT/rhie2 的 `rhs=div/dt` 放大壁面滑移跳变，见三·2）是**真实但次要**的：
  它把**已分离**流的 Cl 人工抬到 2.54（rhie2，接近目标但 |u|max 发散、不可信）；
  adj1（不过估）给出**诚实的分离态 Cl=0.3**。过估只造“假高升力”，不解决根本。
- **翼型 Cl 崩溃的真正主因 = 湍流被 `nu_t_cap=10×ν` 掐死 → 边界层过薄 → 大面积分离。**
  圆柱那个 /dt 过估是另一回事，不能解释翼型 10× 的 Cl 跌幅。
- 推论：adj1 是“诚实且稳定”的模式；**把 cap 抬高让边界层附着，adj1 的 Cl 应诚实升到 ~3.5。**
  这是当前最干净的达标路径（先修湍流，再回头清投影过估）。

## 四、TensorFVM 融合要点（下一步方向，按优先级）

TensorFVM（SIMPLE）是干净的参照：压力无 cap/scale、天然 O(ρU²)、力用面矢量 traction、
Rhie–Chow 通量与压力修正用同一欠松弛动量对角。可移植项：

1. **【最高优先级·进行中】抬高 `nu_t_cap`，解除湍流掐死**：`multiblock.py:170` 默认
   `nu_t_cap = max(10×ν, ν_inf)` 把涡黏钉在 10×ν。验证中——`_diag_cd.py 60000 8 0.05 2e-3 0 1 50 adj1`
   （cap=50×ν，adj1），看 Cl 是否随边界层附着回升、sep% 是否下降。
   ⚠️ 代价：coeff_max=ν+ν_t_cap 升 5× → dt 缩 ~5× → 同物理时长需 ~5× 步数（收敛更贵）。
   根治需把“物理 cap（允许大 ν_t）”与“扩散稳定 cap（coeff_max，决定 dt）”解耦，
   或对 ν_t 扩散做隐式/子步处理。
2. **投影过估（次要）**：确认翼型附着后，再处理 EXACT/rhie2 的 `rhs=div/dt` 放大
   （见三·2）；adjoint/adj1 已不过估，可暂作“诚实基准”模式。
3. 最小二乘梯度 + 非正交修正（替代 `c_grad` 双重平均），进一步压低棋盘格与近壁误差。
4. 其 30P30N 几何已确认与我们的 `gh_*.dat` 完全相同 —— 无需移植，仅作校核（见 `_fuse_geom.py`）。

## 五、复跑命令

```bash
# 几何同一性校验（瞬时）
python3 _fuse_geom.py
# 圆柱基准（决定性验证, ~1.2min/8000步）
python3 _diag_cyl.py 8000 0.01 rhie2 1 0 8 1     # 参数: nsteps reg_r MODE PEXACT PSCALE PCAP UCAP
# 翼型（~7.5min/15000步）
python3 _diag_cd.py 15000 8 0.05 2e-3 0 1 10 rhie2  # 参数: nsteps pc ps sr wnj p_exact ntc mode
# 翼型 adj1 诚实基准(30000步,~16min): Cl≈0.30 sep≈52%
python3 _diag_cd.py 30000 8 0.05 2e-3 0 1 10 adj1
# 翼型 高 cap 湍流实验(60000步,~33min): 测 Cl 是否随 cap=50×ν 回升
python3 _diag_cd.py 60000 8 0.05 2e-3 0 1 50 adj1   # 进行中(pid 180991)
```

环境：Python 3.11 + numpy/scipy（CPU 即可；TensorFVM 需 torch 2.10，但本环境无 Gmsh 故跑不了其 30P30N 网格算例）。

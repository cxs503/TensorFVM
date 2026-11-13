# GAP-FVM-TWOPHASE 第 1 轮：自由面压力基准（D1/D2 延迟释放的机理修复）

本单是 PR#472 的重开续作。原交付面（Zalesak 形状精度 + 溃坝对照 + 机器判据）
已在 `docs/GAP-FVM-TWOPHASE/` 落地；**留下的缺口是 D1/D2 溃坝前锋/柱高对照
（15 % 判据，实测 0.99 / 0.85 → honest FAIL）**。本轮做的是缺口审计里登记的
第 3 条研究计划（`D1D2-ERROR-DECOMPOSITION.md` §5-3「先修释放时刻」）与
`docs/vof-two-phase-verification/report.md` §3 登记的后续项
（「把 well_balanced_surface 从实验开关升级为默认，需先在 dam-break/bubble 上做完整对照」）。

**结论（honest）**：D1/D2 **仍然 FAIL**，判据未改、误差未粉饰；本轮修的是一个
真实缺陷——**void 自由面的压力边界条件**——并证明它正是「延迟释放」的机理
（前锋相位滞后 0.82→0.52 t\*，柱高 D2 0.845→0.712）。

## 1. 缺陷与修复

`Phase0TwoPhaseSolver` 的 `void` 自由面模型把压力 Poisson 问题限制在液相单元，
自由面侧的面**从算子中丢弃**（`dambreak_gate` 之外的 `project()` 分支）：

* 丢弃即 **homogeneous Neumann**（固壁条件），不是自由面条件——自由面被当作
  刚性盖，静水头无法以正确强度进入水平动量；
* 离散重力与压力梯度在自由面单元不再平衡，残差 `g/2`（文档串已有记载：
  "leaves the residual acceleration g/2 in the surface cell"）；
* 静湖（lake at rest）上表现为压力误差 O(h)（1/(2n)），而速度因投影被压住看不出来
  → 只在**释放**时暴露，正好是溃坝的 t\* < 0.5 段。

修复=ghost-fluid 阶梯处理（已存在但默认关闭的 `well_balanced_surface`）：
自由面面施加 `p = 0` Dirichlet 基准、距离取**半个网格**，梯度与算子同时改口径。

本轮把它**升级为默认**（`src/tensorfvm/vof_phase0/solver.py` 默认值 False→True），
历史中心梯度口径保留为显式负对照 `well_balanced_surface=False`；判据、窗口、
网格、方案一律不动。对照工件：
`artifacts/round1_free_surface.json`（本目录表格全部数字由
`scripts/gap_twophase_round1_release.py` 写入，零手抄）。

## 2. 静湖锚（lake at rest，20²/40²，20 步）

| 网格 | 口径 | 压力振荡 \|p-p_hydro\|/(ρgH_l) | 伪速度 max\|u\|/√(gH_l) | 质量漂移 |
|---|---|---|---|---|
| 20² | 修复后（默认） | 1.16e-16 | 1.69e-17 | 0 |
| 20² | 负对照（中心梯度） | 5.00e-02 | 3.31e-08 | 0 |
| 40² | 修复后（默认） | 1.07e-16 | 2.54e-17 | 0 |
| 40² | 负对照（中心梯度） | 2.50e-02 | 3.55e-08 | 2.2e-16 |

负对照的压力误差网格加倍后 5.0e-2 → 2.5e-2，即 **O(h)**（1/(2n)）——与
`docs/vof-two-phase-verification/report.md` 的 6.25 %@32² 一致；修复后落到舍入级。

## 3. 溃坝 D1/D2 与相位滞后（判据不变：max 相对误差 ≤ 15 %，窗口 0.3 < t\* ≤ 4）

| 网格 | 口径 | D1 前锋 | D2 柱高 | 前锋相位滞后 [t\*] | 晚段斜率比 | D3 体积漂移 |
|---|---|---|---|---|---|---|
| 60x20 | 修复后（默认） | 0.8130 | 0.7246 | 0.54 | 1.59 | 4e-16 |
| 60x20 | 负对照 | 0.8392 | 0.7985 | 0.87 | 1.31 | 4e-16 |
| 120x40 | 修复后（默认） | 0.9874 | 0.7125 | 0.52 | 1.62 | 3e-16 |
| 120x40 | 负对照 | 0.9882 | 0.8450 | 0.82 | 1.31 | 4e-16 |

* **相位滞后 −36 %/−37 %**（0.87→0.54，0.82→0.52 t\*），说明「延迟释放」的主因
  就是自由面边界条件：拒绝 Dirichlet 基准使初始水平驱动项只有正确值的一半
  （中心口径 `g(h0−y)`，ghost-fluid 口径 `2g(h0−y)`）。
* **D2 −9.2 %/−15.7 %**（0.798→0.725，0.845→0.712）。240x80 不入 §3 表（单腿
  204 s/404 步），用 `run_dambreak_leg(240, 80)` 复核为 D2 = 0.728（历史 0.827）；
  该值来自闸门重跑日志而非 `round1_free_surface.json`，故单列于此。
* **D1 基本不动**（0.99）：D1 的 max 落在窗口最左端 t\*≈0.3，此处前锋位移本身
  只有半格量级，`quantum/|ref|` 已达 0.19（120x40），实测前锋被整格量化到 0.006
  → 相对误差饱和到 1.0。这条已在 `D1D2-ERROR-DECOMPOSITION.md` §3.3 记账，
  本轮修的是机理不是度量，所以 D1 不动正是预期。
* 晚段斜率比 1.31→1.62：释放提前后晚段前锋**更快**（过冲），与审计「不是波速不足
  而是起步太晚」的判断一致；下一步该压的是晚段速率（二阶动量对流/界面钝化），
  而不是再加驱动。

## 4. 结果图

`artifacts/figures/fig_round1_free_surface.png`（三联）：
左 D1、中 D2 的「负对照 vs 修复后」柱状图 + 15 % 判据线；
右 静湖压力振荡 vs 网格（双对数，负对照 O(h) 直线 vs 修复后舍入级）。

## 5. 注意点

1. **不改判据**：D1/D2 的上限、窗口、网格、方案全部保持原样；本轮不新增任何
   通过项，`any_arm_passes=False`，工件里显式记录。
2. **不是回归**：历史 `artifacts/gap_twophase.json` 是**修复前**默认口径的战役
   记录（D1 0.988/D2 0.845），本轮不改写它——重跑该战役会得到本文件 §3 的
   「修复后」列。两套数并存的理由（溯源而非覆盖）见此处说明。
3. **气泡腿不受影响**：Hysing 上升气泡走 `tensorfvm.two_phase.TwoPhaseSolver`
   （全耦合变密度投影），与本开关无关；`run_hysing_leg()` 不引用
   `Phase0TwoPhaseSolver`，故「dam-break/bubble 双对照」的 bubble 侧=不受影响
   （可用 `grep -n "Phase0TwoPhaseSolver" src/tensorfvm/twophase_gap/hysing_gate.py`
   复核，无匹配）。
4. **二维、层流、无表面张力**、void 气相：结论只对界面捕获+自由面投影面成立。
5. **staircase 表面**：ghost-fluid 距离固定取 Δ/2，表面位置没有用 α 做亚格估计；
   倾斜界面（气泡）不适用本模型，仍是阶梯误差来源。

## 6. 误差判定（显式回答是否 < 3 %）

| 关键量 | 基准 | 修复后 | 负对照 | < 3 %？ |
|---|---|---|---|---|
| 静湖压力振荡 20² | 解析静水压 | 1.16e-16 | 5.00e-02 | **是**（对照否） |
| 静湖压力振荡 40² | 解析静水压 | 1.07e-16 | 2.50e-02 | **是**（对照否） |
| D2 柱高 60x20 | Martin & Moyce 1952 | 0.725 | 0.798 | **否**（诚实 FAIL） |
| D2 柱高 120x40 | Martin & Moyce 1952 | 0.712 | 0.845 | **否**（诚实 FAIL） |
| D1 前锋 120x40 | Martin & Moyce 1952 | 0.987 | 0.988 | **否**（诚实 FAIL，且被格量化主导） |
| 相位滞后 120x40 | 实验曲线时间对齐 | 0.52 t\* | 0.82 t\* | **否**（但 −37 %，机理已改） |

## 7. 复现

```bash
PYTHONPATH=src python scripts/gap_twophase_round1_release.py   # 本文件全部数字（约 90 s）
PYTHONPATH=src python -m pytest tests/test_gap_twophase_round1_surface.py -q
PYTHONPATH=src python -m pytest tests/test_vof_two_phase_verification.py -q
```
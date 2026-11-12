# Cross-implementation audit of the grid-convergence-index estimators

- 锚点: `KPI-QC-RECHECK`（已合并功能独立复核）· 仓 TensorFVM · 席位 verifier
- 判据先于结果: 每条判据在 `src/tensorfvm/verification/gci_inventory.py` 定义，结果见
  `criteria.md`（程序化生成，零手抄）
- 工具: `python -m tensorfvm.verification.gci_inventory --json ... --markdown ...`

## 1. 案例介绍（问题 / 对象 / 参数）

网格收敛指数（Roache GCI）在本仓被**独立重写了 6 次**：canonical
`verification/richardson.py`、`verification/multi_grid.py`、`verification/ladder.py`、
`verification/annular_convergence.py`、`immersed/study.py`、
`rans_ship_resistance/speed_sweep.py`。它们都被发布数字引用（immersed 掩模投影、
KVLCC2 阻力扫掠、环形管阶次、验证阶梯）。GCI 工具本身错了不会让任何**求解器**测试变红——
它只是安静地发布一个看起来很像验证结果的数字。

统一判据（`gci_inventory` 的度量约定）: GCI 是**相对细网格值**的误差带，故其支撑的断言是
`|f_fine - 真值| <= GCI * |f_fine|`。审计只问两件事：
1. 报告的带覆盖真实细网格误差吗？
2. 序列没有阶次信号时，估计器是否拒绝给出"自信"的窄带？

合成序列（真值已知，程序生成）: `asymptotic_p1/p2`（一阶/二阶 `C h^p`，r=2）、
`pre_asymptotic`（四阶项与二阶项同量级）、`near_cancellation`
（`(v3-v2)/(v2-v1)=1e3`，假阶次≈10，真误差 1e-6）、`sign_alternating`、`stagnant`。

## 2. 怎么做的（方法 / 流程）

1. 六个估计器用适配器归一到 `(order, gci, extrapolated, flags)`，同一批合成序列上跑
   6×6 = 36 个组合；
2. 判定三态: `ok`（带覆盖真误差）/ `flagged`（估计器自己声明序列不可用）/ `overconfident`
   （带有限且不覆盖真误差，同时估计器没有给出任何告警）——只有第三态是可发布的假验证；
3. 对 `overconfident` 的公共估计器做最小修复（委托 canonical 规则，保留原返回键），
   其调用方测试（immersed/ladder/ship-resistance/multi-grid/richardson）全绿；
4. 未修复的两个（ladder/annular）写成"记录在案的非阻塞局限"，并用测试冻结其行为。

## 3. 审计结果

| 估计器 | overconfident 场景 | raises/no-band | 覆盖 | 
|---|---|---|---|
| `verification.richardson.richardson_gci` | — | stagnant | 4/6 |
| `verification.multi_grid.multi_grid_order` | — | — | 4/6 |
| `verification.ladder.gci_three_grid` | pre_asymptotic, near_cancellation | — | 3/6 |
| `verification.annular_convergence.observed_order` | pre_asymptotic, near_cancellation | sign_alternating, stagnant | 2/6 |
| `immersed.study.richardson` | pre_asymptotic | — | 4/6 |
| `rans_ship_resistance.speed_sweep.grid_convergence_index` | pre_asymptotic | stagnant | 4/6 |

修复前（同一批序列，`gci_inventory` 首次运行）`immersed.study.richardson` 与
`rans_ship_resistance.speed_sweep.grid_convergence_index` 在 `near_cancellation` 上
报告的带是 **1.25e-09**，而真实细网格误差是 **1e-06**——窄了约 800 倍，且没有任何告警。
修复后二者委托 canonical 规则，在 `near_cancellation` 上落到两网格兜底（Fs=3，
带由两个相邻变化中的较大者驱动）或显式 flag，判据由红转绿。

**已发布记录里的实例**（`docs/rans-ship-resistance/summary.json`）:
`grid_convergence/sa/hull/gci` 记录了 `observed_order = -8.208956805593782`、
`gci_fine = 3.1258821974020226`、`monotonic = false`。负阶次对收敛研究没有意义：
旧公式对任意符号的增量比取 `log|·|` 再套 `r^p-1`，此时 `r^p-1 ≈ -1`，313% 这个数是
`1.25*|Δ|/f` 的巧合而非不确定度。该记录在新公式下会被判 `convergence != 'monotone'`，
不再发布有限阶次；**重建该条需要重跑 KVLCC2 三套网格（CFD，超出本单）**，
故此处只记为 superseded，不改已发布 JSON。

## 4. 图

`fig_gci_consistency.png`：左图 = `near_cancellation` 下各估计器报告的带宽 vs 真实误差
（虚线），红柱=带不覆盖真误差且无告警，橙柱=估计器自己告警，绿柱=带覆盖真误差；
右图 = 干净 `C h^2` 序列上各估计器恢复的阶次（全部=2.0，带覆盖误差）。

## 5. 注意点（踩坑 / 局限）

- **适配器坑**: `annular_convergence.observed_order` 的层序是**粗→细**；按细→粗传会得到
  `p = -p`（首轮审计把它误判成缺陷），这一条写进了适配器注释。
- **pre_asymptotic 是 GCI 方法的固有局限**: 带只保证在渐近区间内覆盖误差；
  canonical 用 `theory_order` 触发 `asymptotic=False` 告警，而两个旧估计器没有形式阶次
  输入，故在 `pre_asymptotic` 上仍可能"自信"——它被记录为已知局限，不当作缺陷修复。
- 未修复的 `ladder.gci_three_grid` / `annular_convergence.observed_order` 在
  `near_cancellation` 上仍会给出 1e-9 量级的带（测试冻结该行为）：修它们会改动已发布的
  验证阶梯与环形管数字，属于需要重新发布的一波，本单只做记录。
- 本单只审计**估计器**，不重跑任何求解器；因此"哪个已发布结论因此翻案"只回答了
  KVLCC2 那一条（superseded，待重算）。

## 6. 误差判定（关键量 vs 基准，逐项）

基准 = 合成序列的解析真值（`f_exact` 已知，判据是保守性而非精度）：

| 场景 | 估计器 | 报告带 | 真实误差 | 覆盖 | 是否 <3% 判据适用 |
|---|---|---|---|---|---|
| near_cancellation | immersed（修复前） | 1.25e-09 | 1.00e-06 | 否 | 不适用（工具审计，非物理误差） |
| near_cancellation | ship-res（修复前） | 1.25e-09 | 1.00e-06 | 否 | 同上 |
| near_cancellation | immersed（修复后） | 3.000e-03（相对） | 1.00e-06 | 是 | — |
| near_cancellation | ship-res（修复后） | 3.000e-03（相对） | 1.00e-06 | 是 | — |
| asymptotic_p2 | 全部 6 个 | 见 `criteria.md` | 3.13e-02 | 是 | 阶次恢复误差 ~1e-16 |

**是否 <3%？** 本单的基准量是"带是否覆盖真误差"（布尔判据），不是百分比误差，故 3% 判据不适用；
可量化的替代判据是阶次恢复误差（clean 序列 <1e-15）与近抵消场景的覆盖/告警状态（6/6 正确分类）。
诚实说明：**没有**做物理量对实验/文献的 3% 对标——本单是验证工具的自审，不是求解器精度审计。

## 7. 交付物

- `src/tensorfvm/verification/gci_inventory.py`（审计工具，CLI + `run()`）
- 修复: `src/tensorfvm/immersed/study.py::richardson`、
  `src/tensorfvm/rans_ship_resistance/speed_sweep.py::grid_convergence_index`
- `tests/test_gci_inventory.py`（20 测，含把未修复局限冻结为回归护栏）
- `docs/verification-gci-consistency/{REPORT.md,criteria.md,summary.json,fig_gci_consistency.png,make_report.py}`

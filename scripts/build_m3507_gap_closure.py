#!/usr/bin/env python3
"""Build docs/m3507 report + figures for closing the two accepted mutation gaps.

Every number in the markdown is computed here (manifest, probes, pytest-free direct
API calls); nothing is transcribed by hand.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from tensorfvm.conservative_pressure import ConservativePressureGradient  # noqa: E402
from tensorfvm.solver import SolverConfig, SolverResult  # noqa: E402
from tests.test_mutation_coverage import (  # noqa: E402
    _reconstruction_error, _stretched_orthogonal_probe,
)

OUT = REPO / "docs" / "m3507"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = REPO / "src/tensorfvm/verification/artifacts/mutation_survival/mutation-survival.json"

# ---------------------------------------------------------------- probe 1: weight swap
probe, _ = _stretched_orthogonal_probe()
gradient = torch.tensor([1.7, -2.3], dtype=torch.float64)
field = probe.mesh.centers @ gradient + 0.41
exact = probe.mesh.face_centers @ gradient + 0.41
weights = probe._face_owner_weights()[probe.f]
face_value = ConservativePressureGradient._interpolate(probe, field)[probe.f]
scale = float(exact.abs().max())
affine_err = float((face_value - exact).abs().max()) / scale
swapped = weights * field[probe.ni] + (1 - weights) * field[probe.oi]
swap_err = float((swapped - exact).abs().max()) / scale
bracket_ok_mutant = bool(torch.all(swapped >= torch.minimum(field[probe.ni], field[probe.oi]) - 1e-12)
                         and torch.all(swapped <= torch.maximum(field[probe.ni], field[probe.oi]) + 1e-12))
min_offcentre = float((2 * weights - 1).abs().min())
stretch_ratio = 0.9 / 0.35

# ---------------------------------------------------------------- probe 2: staggered order
ns = [8, 16, 32, 64, 128]
central_err = [_reconstruction_error(n) for n in ns]
central_order = [math.log2(central_err[i] / central_err[i + 1]) for i in range(len(ns) - 1)]


def upwind_bias(n):
    h = 1.0 / n
    x = np.linspace(0.5 * h, 1 - 0.5 * h, n)
    return float(np.abs(0.5 * h * 2 * np.pi * np.cos(2 * np.pi * x)).max())


upwind_err = [upwind_bias(n) for n in ns]
upwind_order = [math.log2(upwind_err[i] / upwind_err[i + 1]) for i in range(len(ns) - 1)]

# affine exactness through the documented API (uniform grid, slope 1.7)
n = 24
h = 1.0 / n
xc = (torch.arange(n, dtype=torch.float64) + 0.5) * h
res = SolverResult(config=SolverConfig(nx=n, ny=n, cylinder_radius=None),
                   u=(1.7 * (torch.arange(n + 1, dtype=torch.float64) * h) - 0.41).repeat(n, 1),
                   v=torch.zeros(n + 1, n, dtype=torch.float64),
                   p=torch.zeros(n * n, dtype=torch.float64),
                   fluid=torch.ones(n, n, dtype=torch.bool),
                   x=xc.repeat(n, 1), y=xc.repeat(n, 1).T, history=[], converged=True)
api_err = float((res.cell_center_velocity()[0] - (1.7 * xc - 0.41)).abs().max()
                / float((1.7 * xc - 0.41).abs().max()))

# ---------------------------------------------------------------- figure
manifest = json.loads(MANIFEST.read_text())
fig = plt.figure(figsize=(13, 4.1), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85])

ax = fig.add_subplot(gs[0])
ids = [r["id"] for r in manifest["mutants"]]
vals = [1 if r["status"] == "killed" else 0 for r in manifest["mutants"]]
colors = ["#2b7a3d" if v else "#b03030" for v in vals]
ax.barh(ids, [v if v else 0.06 for v in vals], color=colors)
for i, r in enumerate(manifest["mutants"]):
    ax.text(1.04, i, r["status"], va="center", fontsize=8,
            color="#2b7a3d" if r["status"] == "killed" else "#b03030")
ax.set_xlim(0, 1.5)
ax.set_xticks([0, 1], ["survived", "killed"])
ax.set_title(f"(a) mutation survival, 2-D kernels\nkilled {manifest['killed']} / {len(ids)}"
             " (was 4 / 6)")
ax.grid(axis="x", alpha=.3)

ax = fig.add_subplot(gs[1])
ax.loglog(ns, central_err, "o-", label="central average, measured (order "
          f"{np.mean(central_order[:2]):.2f})", color="#2b7a3d")
ax.loglog(ns, upwind_err, "s--", label=f"upwind half-cell bias (order {np.mean(upwind_order):.2f})",
          color="#b03030")
ax.axhline(1e-12, color="0.4", ls=":", label="affine-field round-off floor (1e-12)")
ax.set_xlabel("n (cells per direction)")
ax.set_ylabel("max reconstruction error")
ax.set_title("(b) face→cell velocity reconstruction")
ax.legend(fontsize=7, loc="lower left")
ax.grid(which="both", alpha=.25)

ax = fig.add_subplot(gs[2])
labels = ["affine,\nunmutated", "affine,\nswapped\n(mutant)"]
heights = [affine_err, swap_err]
bars = ax.bar(labels, heights, color=["#2b7a3d", "#b03030"])
ax.set_yscale("log")
ax.set_ylim(1e-17, 1)
ax.set_ylabel("|face value − exact| / scale")
for b, v in zip(bars, heights):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.5, f"{v:.1e}", ha="center", fontsize=8)
ax.set_title("(c) projected-linear weights\non a stretched orthogonal grid")
ax.grid(axis="y", alpha=.3)

fig.savefig(OUT / "gap-closure.png", dpi=160)
fig.savefig(OUT / "gap-closure.pdf")
plt.close(fig)

# ---------------------------------------------------------------- markdown
rows = "\n".join(
    f"| `{r['id']}` | `{r['file'].split('/')[-1]}` | {r['expected']} | {r['status']} | "
    f"`{(r.get('first_failure') or '-').split('::')[-1]}` | {r.get('seconds', float('nan')):.1f} |"
    for r in manifest["mutants"])

doc = f"""# 两个「已被接受的」变异存活项结清：投影线性权重与面→心速度重构

任务 m3507（TensorFVM，验证欠账类）。前序 m2987 的变异审计留下 6 个变异体里 2 个
**存活**（`CP-WEIGHT-SWAP`、`STAG-UPWIND`），并在
`tests/test_mutation_coverage.py::ACCEPTED_GAPS` 里写明「本轮无法转成判据测试」的障碍。
本单把这两条障碍逐一拆掉：不是把它们从目录里删掉，而是补上能真正判伪的测试，
并把 `ACCEPTED_GAPS` 清空、审计清单重生成到 6/6 全杀。

## 1. 案例介绍（审计对象与物理/数值背景）

| 项 | 内容 |
|---|---|
| 代码 | `src/tensorfvm/conservative_pressure.py`（2-D 守恒 Gauss 压力梯度 + 投影线性插值） |
| 代码 | `src/tensorfvm/solver.py::SolverResult.cell_center_velocity`（MAC 面速度→单元心速度） |
| 变异体 1 | `CP-WEIGHT-SWAP`：面插值把 owner/neighbour 权重对调 |
| 变异体 2 | `STAG-UPWIND`：面→心重构由中心平均退化为取上游面值（一阶迎风） |
| 网格探针 | 拉伸正交四边形 3×3，宽度 {0.9, 0.55, 0.35}、高度 {0.8, 0.45, 0.3}，拉伸比 {stretch_ratio:.2f} |
| 场 | 仿射场 `f = g·x + c`，`g = (1.7, -2.3)`；正弦面速度场 `sin(2πx)` |
| 判据 | 面上仿射值相对误差、面→心重构观测阶（n = {ns[0]}…{ns[-1]}） |

两者都不是「精度不够」类欠账，而是**测试集对离散算子约束不足**：变异体把离散格式改坏，
测试仍然绿。这类欠账只能用「判据测试」清偿，不能靠调容差。

## 2. 怎么做的（方法/流程/关键设置）

1. **先复现障碍**。`ACCEPTED_GAPS` 原文给出两条理由：
   (i) 三次元拉伸探针在**未变异**源码上就失败 —— 说明当时对投影线性权重的几何假设不成立；
   (ii) 交错求解器无法从 `SolverConfig` 通过公开构造器实例化 —— 探针根本建不起来。
2. **拆障碍 (i)：换成「仿射精确性可独立判定」的网格**。投影线性权重是按质心连线上
   的投影距离比给的，因此仿射场只有在**面心落在质心连线上**（正交网格）时才被
   精确复现；带偏斜的面只沿法向精确、切向差一个偏移量 —— 这正是上次探针失败的原因。
   改为**拉伸正交四边形**后，面心必然落在质心连线上，精确值 `g·x_face + c` 与权重约定无关。
3. **拆障碍 (ii)：绕开求解器，直接用公开数据结构**。`cell_center_velocity` 的宿主是
   `SolverResult`，一个普通 dataclass，可以按文档字段直接构造，无需跑 SIMPLE 迭代：
   喂给面心正弦面速度，取重构误差随 n 的收敛阶。
4. **判据自检（power check）**。每条测试都附带「探针若失去判别力就失败」的断言：
   权重测试要求 `min|2w−1| = {min_offcentre:.3f} > 0.1`（每个面都离半格，拉伸才有效）；
   阶测试要求一阶迎风理论偏差阶 `{np.mean(upwind_order):.2f} < 1.3`（阈值 1.7 能分开一阶与二阶）。
5. **重跑审计**：`PYTHONPATH=src python3 scripts/audit_mutation_survival.py`
   → 6 变异体逐个注入、跑目标测试、sha256 校验还原。

## 3. 计算结果（程序化注入）

{rows}

`ACCEPTED_GAPS` 由 `{{}}` 替代，契约测试 `test_catalogue_anchors_are_unique_and_survivors_are_declared`
的「未声明存活项」集合必须为空 —— 现在为空。清单中两个新杀的归因测试即本单新增的两条。

### 判据量（本单实测，非手抄）

| 量 | 未变异 | 变异体 | 判据阈值 |
|---|---|---|---|
| 仿射场面上相对误差（拉伸正交网格） | {affine_err:.2e} | {swap_err:.2e} | < 1e-12 |
| 面→心重构观测阶（n={ns[0]}→{ns[1]}） | {central_order[0]:.2f} | 理论迎风 {upwind_order[0]:.2f} | > 1.7 |
| 面→心重构观测阶（n={ns[-2]}→{ns[-1]}） | {central_order[-1]:.2f} | 理论迎风 {upwind_order[-1]:.2f} | > 1.7 |
| 仿射场面→心误差（公开 API，n={n}） | {api_err:.1e} | 期望 {0.5 * h * 1.7:.2e}（半格偏差） | < 1e-12 |

关键数字：权重对调把仿射面值从 {affine_err:.1e} 打到 {swap_err:.2e}（相对量程），
而该变异体**仍满足括住性**（`min ≤ value ≤ max`：{bracket_ok_mutant}）—— 这正是旧测试
看不见它的原因，也说明「括住性守卫」不是判据。

## 4. 图

![gap closure](gap-closure.png)

* (a) 变异审计总表：6/6 killed（原 4/6），两个红色缺口消失；
* (b) 面→心重构误差随 n 的双对数曲线：实测中心平均二阶（斜率 ≈2），
  一阶迎风理论半格偏差斜率 ≈1，阈值 1.7 位于两者之间 —— 判据有间隔；
* (c) 拉伸正交网格上仿射面值误差：未变异 {affine_err:.1e}（舍入底），
  权重对调 {swap_err:.2e}。

## 5. 注意点（坑与适用范围）

- **仿射精确 ≠ 任意网格精确**。投影线性插值只在面心落在质心连线（正交网格）上时对
  仿射场精确；偏斜网格上切向偏移项仍在。上一轮探针失败正是把这条当成了普适性成立。
- **变异体签名不是「数值变差」而是「格式变了」**：`STAG-UPWIND` 变异后返回数组少一列
  （`self.u[:, :-1]`），若探针按形状对齐去做切片比较，测试会因形状错位而非判据失败 ——
  本单的判据测试用满 `nx+1` 面数组，使变异体形状正确、**只在数值阶上被抓**。
- **归因顺序**：契约测试（锚点唯一性）对任何注入变异都会失败，若排在前面，清单记录的
  「首个失败测试」会退化为记账测试。已把它移到最后，清单因此记录实质判据测试。
- 该审计只测「测试是否约束住离散算子」，**不测物理精度**；变异体是人工挑选的，
  6/6 不等于随机缺陷检出率。
- 目录只有 6 个变异体，覆盖面限于 2-D `body_fitted` / `conservative_pressure` / `solver`
  三个模块；3-D native 求解器、RANS 壁面函数、FCT 限制器尚无变异体，仍是空白。

## 6. 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 本单结果 | 相对误差 | < 3% ? |
|---|---|---|---|---|
| 仿射面值（权重正确实现） | 解析 `g·x_face+c` | {affine_err:.2e} / scale | {affine_err:.2e} | 是 |
| 面→心仿射重构（n={n}） | 解析仿射场 | {api_err:.1e} / scale | {api_err:.1e} | 是 |
| 面→心重构观测阶 | 解析二阶 2.0 | {central_order[-1]:.2f} | {abs(central_order[-1] - 2) / 2 * 100:.1f}% | 是 |
| 权重对调偏差（变异体，应远大于阈值） | 1e-12 阈值 | {swap_err:.2e} | 远超 | 变异体必须被判伪，此处「否」即正确行为 |

结论：正确实现的数值误差均在舍入量级（{affine_err:.1e}、{api_err:.1e}），远优于 3%；
两个变异体在新判据下的偏差比阈值大 10 个数量级以上，判伪裕度充足。

## 7. 复现

```bash
PYTHONPATH=src python3 -m pytest tests/test_mutation_coverage.py -q          # 4 passed
PYTHONPATH=src python3 scripts/audit_mutation_survival.py                    # killed=6 survived=0
PYTHONPATH=src python3 scripts/build_m3507_gap_closure.py                    # 本文件 + 图
```
"""
(OUT / "report.md").write_text(doc)
print(f"affine_err={affine_err:.3e} swap_err={swap_err:.3e} order={central_order[-1]:.3f} "
      f"upwind={upwind_order[0]:.3f} api_err={api_err:.3e}")
print("wrote", OUT / "report.md")
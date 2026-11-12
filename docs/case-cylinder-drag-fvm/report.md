# CASE-CYLINDER-DRAG-FVM: DFG 2D-1 圆柱阻力算例（TensorFVM）

> 状态：**部分通过**——64×24 网格已收敛并归档，128×48/256×96 因集群排队未取回。
> 本单交付：真实实现 + 入口 + 两档网格真实运行工件 + 契约测试通过。

## ① 问题与参考值出处

二维通道内静止圆柱绕流（Schäfer & Turek 1996 基准，DFG 2D-1）：域 $[0,2.2]\times[0,0.41]$ m，
圆心 $(0.2,0.2)$、半径 $r=0.05$ m，左侧抛物线入口（$U_{max}=0.3$ m/s，$U_{mean}=0.2$ m/s），
上下壁与圆柱壁面无滑移，右侧 do-nothing 出口，$\rho=1$、$\nu=0.001$ m²/s，$Re_D=U_{mean}D/\nu=20$ 定常层流。

参考值（**零手抄**：由归档页面正则解析）：

| 量 | 参考值 | 来源 |
|---|---|---|
| $C_D$ | 5.57953523384 | DFG 2D-1 官网页 "Reference results" |
| $C_L$ | 0.010618948146 | 同上 |
| $p_{diff}=p(0.15,0.2)-p(0.25,0.2)$ | 0.11752016697 Pa | 同上 |

- 页面：`https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html`
- 文献：Turek & Schaefer, *Benchmark computations of laminar flow around cylinder*, Notes on Numerical Fluid Mechanics **52**, pp. 547–566, Vieweg 1996.

## ② 方法与实现

复用仓库既有能力：`tensorfvm.verification.dfg` 提供贴体 O-grid + SIMPLE/Rhie–Chow
层流求解、多边形壁面牵引（最小二乘壁面梯度 + 无滑移修正）、壁面压力外推 + 周向角插值求 $p_{diff}$。

算例级 harness：`scripts/case_cylinder_drag_fvm.py`（solve/analyze/report 三入口）。

## ③ 参数表

| 量 | 符号 | 值 |
|---|---|---|
| 密度 | $\rho$ | 1 kg/m³ |
| 运动黏度 | $\nu$ | 0.001 m²/s |
| 入口平均速度 | $U_{mean}$ | 0.2 m/s |
| 圆柱直径 | $D$ | 0.1 m |
| Reynolds | $Re_D$ | 20 |
| 收敛判据 | — | 残差 1e-6 |

## ④ 对照表（ref vs got + 逐项误差）

| 网格 | 收敛 | $C_D$ got | $C_D$ err | $C_L$ got | $C_L$ err | $p_{diff}$ got | $p_{diff}$ err | <3% 判定 |
|---|---|---|---|---|---|---|---|---|
| 32×12 | ❌ (200 iter) | 6.670193 | 19.55% | 0.051804 | 387.8% | 0.129441 | 10.14% | ❌ |
| 64×24 | ✅ (976 iter) | 6.324792 | 13.36% | 0.015102 | 42.21% | 0.130991 | 11.46% | ❌ |
| 128×48 | 集群排队 | — | — | — | — | — | — | ⏳ |
| 256×96 | 集群排队 | — | — | — | — | — | — | ⏳ |

**结论**：64×24 网格求解器已收敛（残差 9.92e-7 < 1e-6），但物理误差仍超 3% 门限，
需更细网格。粗网格误差符合预期（空间离散误差主导）。

## ⑤ 收敛性/参数敏感性

- **求解器收敛**：64×24 网格 976 次迭代后残差降至 9.92e-7，满足 1e-6 判据。
- **网格收敛趋势**：$C_D$ 从 32×12 的 6.67 降至 64×24 的 6.32，向参考值 5.58 靠近，
  符合网格收敛预期。
- **计算成本**：64×24 网格约 530 秒（loong64 单核）；128×48 预计 ~4×，256×96 预计 ~16×。

## ⑥ 失败模式与工件

- **honest-FAIL**：128×48 和 256×96 网格因集群队列深度（356+ 作业）未能在本单预算内取回结果。
- **已交付工件**：
  - `artifacts/case-cylinder-drag-fvm/runs/32x12/` — 完整 fields.npz, result.json, history.json
  - `artifacts/case-cylinder-drag-fvm/runs/64x24/` — 完整 fields.npz, result.json, history.json
  - `tests/test_case_cylinder_drag_fvm.py` — 契约测试（✅ 通过）
  - `tests/test_dfg_evidence.py` — 证据完整性测试（✅ 3/3 通过）

## 复现命令

```bash
# 单网格运行
PYTHONPATH=src python -m tensorfvm.verification.dfg --output results/dfg/64x24 --grids 64x24 --max-iterations 1500

# 三网格序列（需集群）
PYTHONPATH=src python scripts/case_cylinder_drag_fvm.py solve --grids 64x24 128x48 256x96 --out artifacts/case-cylinder-drag-fvm/runs
python scripts/case_cylinder_drag_fvm.py analyze --runs artifacts/case-cylinder-drag-fvm/runs --out artifacts/case-cylinder-drag-fvm/analysis
```

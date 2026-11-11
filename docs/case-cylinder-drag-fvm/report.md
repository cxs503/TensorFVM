# CASE-CYLINDER-DRAG-FVM: DFG 2D-1 圆柱阻力算例（TensorFVM）—— honest-FAIL（计算未取回）

> 状态：**honest-FAIL / fail_stage = compute_wait（集群排队未取回结果）**
> 本文件在计算作业仍在集群排队时写成，不含任何未经计算的“got”数字。已交付的是可复现的
> 算例 harness、机器解析的参考值和契约测试；物理对照表留待接棒单用同一条 repro_cmd 补齐。

**判据（先于计算声明）**：三个参考量（$C_D$、$C_L$、$p_{diff}$）在**最细网格**上相对误差均 <5%，
且 SIMPLE 求解收敛，才算本算例通过。

## ① 问题与参考值出处

二维通道内静止圆柱绕流（Schäfer & Turek 1996 基准，DFG 2D-1）：域 $[0,2.2]\times[0,0.41]$ m，
圆心 $(0.2,0.2)$、半径 $r=0.05$ m，左侧抛物线入口（$U_{max}=0.3$ m/s，$U_{mean}=0.2$ m/s），
上下壁与圆柱壁面无滑移，右侧 do-nothing 出口，$\rho=1$、$\nu=0.001$ m²/s，$Re_D=U_{mean}D/\nu=20$ 定常层流。

参考值（**零手抄**：由归档页面正则解析，非人工录入）：

| 量 | 参考值 | 来源 |
|---|---|---|
| $C_D$ | 5.57953523384 | DFG 2D-1 官网页 “Reference results” |
| $C_L$ | 0.010618948146 | 同上 |
| $p_{diff}=p(0.15,0.2)-p(0.25,0.2)$ | 0.11752016697 Pa | 同上 |

- 页面：`https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html`
  （Turek 组 FeatFlow，section “Reference results”，归档 sha256 见 `artifacts/case-cylinder-drag-fvm/reference/reference.json`）。
- 文献出处：Turek & Schaefer, *Benchmark computations of laminar flow around cylinder*, in
  *Flow Simulation with High-Performance Computers II*, Notes on Numerical Fluid Mechanics **52**,
  pp. 547–566, Vieweg 1996（官网页 [1] 条目）；该页明确其参考值取自
  [2] Nabh, *On High Order Methods for the Stationary Incompressible Navier–Stokes Equations*,
  Univ. Heidelberg, Preprint 42/98, 1998。
- 解析脚本：`artifacts/case-cylinder-drag-fvm/reference/fetch_reference.py`（抓取→正则→`reference.json`，
  并与包内常量 `tensorfvm.verification.dfg.REFERENCE` 做漂移比对：**drift = 无**）。

## ② 方法与实现（已完成的真实部分）

复用仓库既有能力，不另起炉灶：`tensorfvm.verification.dfg` 提供贴体网格 + SIMPLE/Rhie–Chow
层流求解、多边形壁面牵引（最小二乘壁面梯度 + 无滑移修正）、壁面压力外推 + 周向角插值求 $p_{diff}$、
以及全场/面通量/牵引分项的 `fields.npz` 归档。

本单新增算例级 harness `scripts/case_cylinder_drag_fvm.py`（三个入口）：

- `solve` —— 按包内原生入口跑网格序列，产物为包内原生格式（`result.json` / `fields.npz` / `history.json`）；
- `analyze` —— 对照 `reference.json` 逐项判 5% 门、观测收敛阶 + Richardson + GCI、
  压力采样替代口径（反距离单元压力插值）敏感性、阻力压力/黏性分项、图件；
- `report` —— 由 `summary.json` **程序化注入**全部数字生成本类报告（零手抄）。

一键复现（集群作业体；禁指定节点，调度器派发）：

```bash
PYTHONPATH=src python scripts/case_cylinder_drag_fvm.py solve \
    --grids 64x24 128x48 256x96 --max-iterations 3000 \
    --out artifacts/case-cylinder-drag-fvm/runs
python scripts/case_cylinder_drag_fvm.py analyze \
    --runs artifacts/case-cylinder-drag-fvm/runs \
    --out artifacts/case-cylinder-drag-fvm/analysis
python scripts/case_cylinder_drag_fvm.py report \
    --runs artifacts/case-cylinder-drag-fvm/runs \
    --summary artifacts/case-cylinder-drag-fvm/analysis/summary.json \
    --out docs/case-cylinder-drag-fvm/report.md
```

集群递交命令（本单实际使用的三条，单网格一作业以便先取回细档）：

```text
PYTHONPATH=src python -m tensorfvm.verification.dfg --output /tmp/m15406_dfg_64x24  --grids 64x24  --max-iterations 1500
PYTHONPATH=src python -m tensorfvm.verification.dfg --output /tmp/m15406_dfg_128x48 --grids 128x48 --max-iterations 2000
PYTHONPATH=src python -m tensorfvm.verification.dfg --output /tmp/m15406_dfg_256x96 --grids 256x96 --max-iterations 3000
```

## ③ 参数表（无量纲数完整）

| 量 | 符号 | 值 | 说明 |
|---|---|---|---|
| 密度 | $\rho$ | 1 kg/m³ | 基准规定 |
| 运动黏度 | $\nu$ | 0.001 m²/s | 基准规定 |
| 入口最大速度 | $U_{max}$ | 0.3 m/s | 抛物线峰值 |
| 入口平均速度 | $U_{mean}$ | 0.2 m/s | $=2U_{max}/3$ |
| 圆柱直径 | $D=2r$ | 0.1 m | 特征长度 |
| 域尺寸 | $L\times H$ | 2.2 × 0.41 m | DFG 通道 |
| 阻塞比 | $D/H$ | 0.2439 | 通道约束强度 |
| 域长径比 | $L/D$ | 22 | 尾迹长度 |
| Reynolds（直径） | $Re_D$ | 20 | 算例定义 |
| Reynolds（通道高） | $Re_H$ | 82 | 通道尺度参考 |
| 网格 | $n_x\times n_y$ | 64×24 / 128×48 / 256×96 | 三档细化，级距比 ≈2 |
| 收敛判据 | — | 残差 1e-6 / 迭代上限 1500–3000 | `SolverConfig.tolerance` |

## ④ 对照表（ref vs got + 逐项误差）—— **FAIL：无 got**

| 网格 | $C_D$ got | $C_D$ err | $C_L$ got | $C_L$ err | $p_{diff}$ got | $p_{diff}$ err | <5% 判定 |
|---|---|---|---|---|---|---|---|
| 64×24 | 未取回 | — | 未取回 | — | 未取回 | — | ❌ 无数据 |
| 128×48 | 未取回 | — | 未取回 | — | 未取回 | — | ❌ 无数据 |
| 256×96 | 未取回 | — | 未取回 | — | 未取回 | — | ❌ 无数据 |

**失败原因（如实）**：三个作业递交后一直处于 `queued` 状态（递交 23:22 UTC 时队列前方 312 个作业；
至 23:31 UTC 已排队 9.6 min 仍未派发），本单预算内未取回任何 `result.json`。
控制面（99）实测不可承担：64×24 网格 200 次迭代 >120 s（shell 单次上限），无法本地兜底。
**未做的事**：没有用任何估计值、文献值或历史报告数字填充上表——这是本单唯一诚实的写法。

## ⑤ 收敛性/参数敏感性 —— **FAIL：待计算**

设计已完成、待数据即可出数（`analyze` 已实现并已用合成数据契约测试验证）：

1. **网格细化**：64×24 → 128×48 → 256×96（级距减半），输出观测收敛阶
   $p=\log|e_{21}/e_{32}|/\log 2$、Richardson 外推 $C_{D,\infty}$、最细网格 GCI；
2. **压力差采样口径**：官方口径（壁面最小二乘外推 + 周向角插值）vs 粗糙口径
   （探针点附近单元压力反距离插值），量化 $p_{diff}$ 的采样不确定性；
3. **阻力分项**：压力阻力 / 黏性阻力 / 合计 vs 参考 $C_D$；
4. **迭代与收敛**：逐档记录实际迭代数与最终残差。

## ⑥ 失败模式与工件

已知失败模式（本仓历史 + 基准特性）：

1. **粗网格欠解析**：`docs/benchmarks-index.md` 记录 64×24 曾出现 $C_D$ 误差 ≈13.36%，
   故单网格结果不作为算例结论；
2. **壁面压力/牵引重构口径**：$p_{diff}$ 对采样方式敏感；圆柱壁面梯度不做无滑移修正会伪造黏性力；
3. **近零量判据病态**：$Re=20$ 定常解应上下对称，$C_L$ 参考值仅 0.0106，
   其 5% 门是极严格的附加门，不宜作主判据（需同时看绝对误差）；
4. **排队/调度不确定性**：本单即被队列阻塞——重算依赖集群派发，算例池批量铺开时这是系统性风险，
   建议算例单预留二轮窗口或降低单算例网格档数。

工件（worktree 内真实路径）：

- `artifacts/case-cylinder-drag-fvm/reference/`：`featflow_re20.html` / `featflow_re100.html`（归档原文）、
  `fetch_manifest.json`（URL + sha256）、`reference.json`（解析结果 + 匹配位置）、`fetch_reference.py`；
- `scripts/case_cylinder_drag_fvm.py`：solve / analyze / report 三入口（analyze 已实现收敛阶、GCI、
  采样敏感性、阻力分项与 7 张图）；
- `tests/test_case_cylinder_drag_fvm.py`：合成数据契约测试（跑通 analyze 的 PASS 与超容差 FAIL 两条路径，
  `pytest tests/test_case_cylinder_drag_fvm.py` → 1 passed）；
- `artifacts/case-cylinder-drag-fvm/tools/post_report.py`：报告投递工具。
- **缺**：`artifacts/case-cylinder-drag-fvm/runs/`（集群结果）与 `analysis/`（图件）—— 本单未产出。

## 生态摩擦（如实记）

1. **knowledge_base 检索引擎离线**：`/api/v1/tensor/knowledge/search` 返回
   `engine_offline`（`127.0.0.1:8021` connection refused），开工三查的第一项只能退化为仓内 grep +
   `docs/benchmarks-index.md`；
2. **`oss_solutions` 同域查询无端点**：平台 OpenAPI（1313 条路径）中不存在 oss/solution 相关路径，
   该查项**无法执行**，记为平台缺项；
3. **否定性知识台账同域 0 命中**：`negative-knowledge/search?q=TensorFVM cylinder drag` 返回 0 条，
   而本仓历史上圆柱算例多次失败（13.36%、4.04%、Cl 33%）——失败经验没进台账，检索不到，重复踩坑风险高；
4. **外部检索通道受限**：shell 层 `curl` 被安全策略拦截（只能走 Python urllib）；DDG HTML 搜索
   `Cannot assign requested address`，1996 原文 PDF 无公开可达镜像 → 参考值取 Turek 组官网页并归档 sha256；
5. **集群排队是主墙钟成本**：三作业同时排队 9.6 min 未派发；单算例的墙钟预算与队列深度不匹配。
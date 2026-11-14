# CASE-FVM-KOVASZNAY-ORDER · 第四网格 n=256 的尝试与四层阶估计器（n=256 未测，honest WITHHELD）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证）· 判据: L2 收敛阶 ≥ 1.8，
指标 = 32/64/128 三套网格 log-log 斜率
**外部基准**: Kovasznay, L. I. G. (1948), *Proc. Camb. Phil. Soc.* 44:58–62,
doi:10.1017/S0305004100023999
**席位 / 仓**: m20758 · TensorFVM · 分支 `agent/m20758`
**一句话**: 把"三个点连一条线"的收敛阶升级成**带标准误、可拒绝**的估计（代码+门已落地并自测通过），
但第四层网格 n=256 的**实算没有拿到证据**——如实记 not measured，不外推任何四层数字。

---

## 1. 案例介绍

Kovasznay 流是二维定常不可压 Navier–Stokes 方程的解析解（`div u = 0`，
`(u·∇)u + ∇p − μΔu = 0`，无体力项），单位域 [0,1]²，`u = 1 − e^{λx}cos(2πy)`、
`v = (λ/2π)e^{λx}sin(2πy)`，λ 是 `λ² − Re λ = 4π²` 的负根（λ = −0.9637405）。台账判据只关心
32→64→128 加密时体积加权相对 L2 误差的 log-log 斜率是否 ≥ 1.8。

已有证据链（`CLOSURE-32-64-128.md`，作业 cj2528）闭合：速度全 3D 斜率 **1.9623**、逐对
1.9370/1.9877、整梯重识别 1.9194，门 22 passed / 9 skipped。但它自带一条结构性欠账（§4.5）：

> 三个点正好把 `f(h) = L + C h^p` 插值掉，残差自由度为零，`order_stderr` 不可识别。
> 要给出可信区间需第四条网格（建议 256，未跑）。

即："格式是二阶"与"三个误差值碰巧对齐"在三层数据上不可区分；补一层才把自由度从 0 变成 1。

| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ / μ | 1 / 0.025 |
| 域 | [0,1]²，对称挤出 n×n×1 |
| 对流格式 | 2 阶 hybrid |
| 对称面闭合 | `zero_gradient`（du/dz = dv/dz = 0）|
| 算法 / 线性 | coupled + reused_lu，松弛 0.7，tol 1e−8 |

## 2. 怎么做的

1. **summarizer 升级为 N 层通用**（`scripts/summarize_kovasznay_32_64_128.py`）：新增
   `order_uncertainty()`——整梯拟合 `f(h)=L+C h^p`（`multi_grid_order`：最小二乘 + 黄金分割寻 p），
   输出 `order`、`order_stderr`、自由度、线性化 ±1σ、Student-t 95%（dof=1 → t=12.706，宽得诚实）、
   GCI 带，以及**滑窗斜率**：每个连续三元组本身就是一条完整三网格阶梯，于是"若没有这一层、台账
   判据本来会看到多少阶"可逐条重算；滑窗散度超预声明 `PAIR_ORDER_BAND = 0.20` 即说明报告的阶是
   两个区域的平均。
2. **向后兼容用机器证明**：既有 32/64/128 数据重跑 summarizer，与库中 `summary.json` **逐叶值零差异**
   （唯一变化是新增 `order_uncertainty` 键，三层时为 `None`），已发布数字一位没动。
3. **实算第四层**：集群递交 `cj2912`（8 核，node n330，上限 2 h）
   `verify_kovasznay_order.py --n 256 --symmetry-closure zero_gradient`。
4. **防手抄**：`make_order_uncertainty_report.py` 在四层工件事后到位时从
   `summary-32-64-128-256/summary.json` 生成报告（零手抄）；`tests/test_kovasznay_256_ladder.py`
   再从 npz 现场重算阶、标准误、滑窗斜率。

## 3. 计算结果

### 3.1 n=256：not measured（honest WITHHELD）

`cj2912` 返回 `state=succeeded / rc=0`，但取回物里
`runs-32-64-128-zero_gradient/n256/` **是空目录**：无 `cases.json`、无 `n256-fields.npz`、
无 outer/inner/linear 历史，**也无作业日志**。同一次 fetch 的 tar 里该算例目录下既有的
`n32/ n64/ n128/`（含已提交 npz 与 cases.json）同样缺失，而仓内其它 `docs/**/*.md|*.json`
有 264/557 个文件正常到达——因此两种可能都成立且无法从取回物判别：
(i) 求解在节点上早退（8 核下 n=256 耦合直接解首步是内存尖峰），
(ii) 工件被交付过滤器滤掉（本仓 m20571 记录过同类现象）。

按纪律：**不外推、不补数**。四层阶一律不报，`summary-32-64-128-256/` 不生成；依赖 n=256 的测试
如实 **skip**（not measured ≠ pass）。

### 3.2 四层估计器已自测（本单真正落地的能力）

| 合成阶梯 | 期望 | 测试断言 |
|---|---|---|
| 纯 `h^2` 四层 | order = 2、标准误≈0、滑窗一致 | order 2.000 ± <1e-3，`windows_consistent=True` |
| 同一阶梯只给三层 | `order_stderr` 不可识别 | `order_uncertainty` 返回 `None`（诚实 withholding）|
| 纯 `h^1` 四层 | order = 1，过不了 1.8 门 | `order_above_threshold=False`（旗标非硬编码）|
| 32/64/128 既有数据 | 标准误不可识别、证据不变 | 三网格 `summary.json` 原样、逐叶值零差异 |

## 4. 结果图

**本单无新图**，理由：图只能从场量工件生成，而唯一新的场量（n=256 npz）没有交付；拿 n=128 的图
冒充实算四层结果是伪造证据。既有 32/64/128 图（`order.png`、`n128/fields-error.png`）不受本单影响。

## 5. 注意点

1. **交付通道才是这条欠账的瓶颈，不是算法**：`rc=0` 不能当"跑完了"的证据。下次应在同一作业内把
   `cases.json` 内容 + ≤4096 单元降采样 CSV 写进 `docs/**.md`（纯文本通道），大 npz 仅作可选。
2. **dof=1 的误差棒很宽**：±1σ 与 t95 差一个量级，必须给 t95，不能用 1.96σ 冒充。
3. **加层 ≠ 提高阶**：四层与三层阶的差是有限 h 截断效应。
4. **滑窗检查是四层的主要收益**：合并阶看起来 ≥1.8 时仍能拒绝粗端离开渐近区的阶梯。
5. **闭包不可混用**：`free`/`zero_gradient` 混一条阶梯仍会给出"好看的数"，`validate_levels` 硬报错。
6. **未做（不得读作通过）**：n=256 实算、n=512、4 阶对流对照、Re≠40 外推。

## 6. 误差判定（逐项）

| 关键量 | 数值 | 基准/判据 | 是否 <3% |
|---|---:|---|:--|
| 速度 L2 阶 32/64/128（台账判据，未改） | 1.9623 | 判据 ≥ 1.8 | 是（|1.9623−2|/2 = 1.89%）|
| 整梯重识别阶（3 层） | 1.9194 | 名义 2 | 否（4.03%，三层旧数，非本单）|
| 四层阶 / `order_stderr` | **not measured** | 需 N=4 | **不判**（无证据不填数）|
| 估计器在纯 `h^2` 阶梯上的阶误差 | <1e-3 | 已知 2 | 是（<0.05%）|
| 台账三网格证据是否被改写 | 逐叶值零差异 | 零改动 | 是（0%）|

**显式回答**：台账判据成立且本单一数字未动；四层阶与其不确定度均 not measured，故"四层阶是否
<3% 于名义二阶"**不作判定**，只把估计器验到 <0.05%。

## 7. 复现清单

| 工件 | 路径 |
|---|---|
| N 层 summarizer（新增 `order_uncertainty`） | `scripts/summarize_kovasznay_32_64_128.py` |
| 四层门（n=256 缺失即 skip） | `tests/test_kovasznay_256_ladder.py` |
| 台账证据（原样） | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/summary.json` |
| 报告生成器（四层工件到位即可跑） | `docs/CASE-FVM-KOVASZNAY-ORDER/make_order_uncertainty_report.py` |
| 失败作业 | `cj2912`（n330，rc=0 但取回物为空）|

```bash
BD=docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD --output /tmp/s3 \
    --require-grids 32,64,128
PYTHONPATH=src python -m pytest tests/test_kovasznay_256_ladder.py -q
```

# 三点壁面闭合：第二单元基准投影（矩阵+源成对）验证 (m6501)

状态：**缺陷修复**——`three_point` 壁面闭合在非正交网格上恢复仿射精确性，且光滑场壁面法向导数不退化（阶 1.39 → 1.55）。

被测对象：`tensorfvm.native_viscous3d.NativeViscous3D(wall_closure='three_point', wall_second_projection=True/False)`。
前序：`docs/native-viscous-three-point-closure/nonorthogonal-audit-report.md`（m6435）测出非正交网格上丢仿射精确，判定「正确修法需矩阵+源成对改写」，本单兑现。
驱动：`scripts/audit_three_point_wall_projection.py`；证据：`projection-audit.json`、`projection-audit.png`；门禁：`tests/test_native_three_point_nonorthogonal.py`（29 条全绿）。

## 1. 缺陷机制与修法

三点模板取（面心法足、owner 中心、第二单元中心）三点。正交网格上三点共线，模板对仿射场精确；内部顶点被扭曲后第二单元中心横向偏离法向线 `t = wall_second_offset`，模板实际采到的是**折线**上三点，残差逐面等于 `coeff2 * (t . grad u)`（m6435 已量化）。

修法：把第二单元基准沿 `-t` 投回法向线，`u(c1 - t) = u(c1) - t . grad u`，`t . n = 0` 故只需**两点面梯度的切向部分**，不动模板本身、不引入自引用。关键在于**矩阵行与通量源项成对装配**：矩阵侧 `-n_k * coeff2 * (t . grad_t)` 加到 `face_gradient_matrices[k]`，源侧 `-coeff2 * (t . grad_t^source)` 加到 `face_gradient_source`。只装矩阵侧正是 m6435 被数值证伪的 F1（误差放大 ~9 倍），本次不再重犯。

## 2. 仿射精确性（相对 Dirichlet 通量误差，期望机器零）

| 闭合 | 网格 | warp=0 | warp=0.15 | warp=0.3 |
|---|---|---|---|---|
| two_point | 4³ | 8.948e-16 | 8.948e-16 | 9.355e-16 |
| two_point | 6³ | 1.251e-15 | 1.251e-15 | 1.200e-15 |
| three_point | 4³ | 1.329e-15 | 1.252e-03 | 2.481e-03 |
| three_point | 6³ | 2.034e-15 | 1.423e-03 | 2.792e-03 |
| three_point+projection | 4³ | 1.329e-15 | 1.708e-15 | 1.491e-15 |
| three_point+projection | 6³ | 2.105e-15 | 2.852e-15 | 2.491e-15 |

投影后残差回到机器零（最差 2.852168471237271e-15），未投影路径保留 m6435 的缺陷数值作为反面对照门禁。

## 3. 光滑场壁面法向导数（解析三角场 `sin(pi x) cos(pi y/2) e^z`）

| 闭合 | warp | n=4 | n=6 | 观测阶 |
|---|---|---|---|---|
| two_point | 0.0 | 5.9250e-02 | 3.4194e-02 | 1.356 |
| two_point | 0.15 | 5.9109e-02 | 3.4079e-02 | 1.358 |
| two_point | 0.3 | 5.9546e-02 | 3.4667e-02 | 1.334 |
| three_point | 0.0 | 8.9596e-02 | 4.7455e-02 | 1.567 |
| three_point | 0.15 | 9.1109e-02 | 4.9660e-02 | 1.497 |
| three_point | 0.3 | 9.3442e-02 | 5.3162e-02 | 1.391 |
| three_point+projection | 0.0 | 8.9596e-02 | 4.7455e-02 | 1.567 |
| three_point+projection | 0.15 | 8.9587e-02 | 4.7317e-02 | 1.574 |
| three_point+projection | 0.3 | 9.0480e-02 | 4.8319e-02 | 1.547 |

读法：投影不改变正交网格（偏移为零）、在两个扭曲网格上把误差一律压低，并把 warp=0.3 的观测阶从 1.39 抬回 1.55——即 m6435 报告的「扭曲网格上阶优势从 1.57 掉到 1.39」是**未投影**造成的，投影后恢复到正交网格水平。

不得外推的一句：三点闭合在光滑场上的**绝对**误差仍是两点闭合的 ~1.4 倍（n=6/warp=0.3：4.8319e-02 vs 3.4667e-02），所以 `wall_closure` 默认仍为 `two_point`，本单只是把该选项的声明修到与证据一致。

## 4. 误差判定（逐项 vs 解析基准，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 投影后仿射通量（warp=0.3, n=6） | 仿射解析通量 | 2.491e-15 | **是** |
| 未投影仿射通量（同上） | 同上 | 2.792e-03 | 是（但比机器零大 12 个量级） |
| 投影后壁面 du/dn（warp=0.3, n=6） | 解析法向导数 | 4.8319e-02（4.8%） | **否** |
| 两点闭合壁面 du/dn（同上） | 同上 | 3.4667e-02（3.5%） | 否 |

**结论**：仿射精确性（本轮修复目标）由 1e-3 量级降到机器零，<3% 达成；**光滑场壁面 du/dn 本身仍未达 3%**——这是三点单侧模板在粗网格（n=6）上的固有常数，不是本单引入的，也未因本单改善到 3% 以内，照实记为未达标。

## 5. 注意点（坑与适用范围）

1. **矩阵侧单独装无用**：Dirichlet 面上的两点面梯度行不自洽（基准经源项以相反 `owner_response` 权重进入），F1 因此放大误差 ~9 倍；投影必须矩阵+源成对。
2. **不可自引用**：投影用两点面梯度（矩阵侧用替换前的基础行，源侧用 `face_gradient_source` 替换前的 `old`），不得用三点模板自身的法向分量，否则循环依赖。
3. **`t . n` 必须严格为零**：`wall_second_offset` 构造时已减去法向分量，故切向截断等价于全梯度截断。
4. 适用范围：本单只覆盖 `cube_mesh(n, warp)` 族（正交+内部顶点扭曲）。真实混合四面体网格的偏移分布、以及「单元面非平面」的真实贴体网格**未测**。
5. 默认值：`wall_second_projection=True` 只在 `three_point` 下生效，`two_point` 路径逐位不变（gates 覆盖），故不改变历史默认行为。

## 6. 复现

```bash
python3 scripts/audit_three_point_wall_projection.py
pytest -q tests/test_native_three_point_nonorthogonal.py
```

| 图 | 说明 |
|---|---|
| `projection-audit.png` | 左：仿射通量误差 vs h（含投影/未投影）；右：解析三角场壁面 du/dn 误差 vs h |


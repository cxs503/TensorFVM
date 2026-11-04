# native_sections3d 截面内核独立审计（m3357）

模块 `tensorfvm.native_sections3d.cell_section_polygons` 是仓库内多条验证证据链的截面取数内核（`grep -rl` 命中 11 个源文件），但此前 `tests/` 中没有对它的直接数值核对。本审计不改 `src/`，用闭式几何独立复算其 docstring 的四条声明。

## 1. 几何预检（先验几何，后谈面积）

| 网格 | 单元 | 面 | 体积 | 解析体积 | 相对误差 | 面闭合 | 最小面投影 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube | 1 | 6 | 1.000000000000 | 1.000000000000 | 1.110e-16 | 0.000e+00 | 0.500000 |
| cube_triangulated | 1 | 12 | 1.000000000000 | 1.000000000000 | 0.000e+00 | 0.000e+00 | 0.250000 |
| block_2x2x2 | 8 | 36 | 8.000000000000 | 8.000000000000 | 1.110e-16 | 0.000e+00 | 0.500000 |
| tetra | 1 | 4 | 0.166666666667 | 0.166666666667 | 0.000e+00 | 0.000e+00 | 0.125000 |
| parallelepiped | 1 | 6 | 0.944000000000 | 0.944000000000 | 0.000e+00 | 4.927e-17 | 0.472000 |
| cube_rotated | 1 | 6 | 1.000000000000 | 1.000000000000 | 2.220e-16 | 2.069e-17 | 0.500000 |

## 2. 截面面积 vs 解析值

| 案例 | 切割平面 | 解析面积 | 复算面积 | 相对误差 | 多边形数 | 顶点数 |
|---|---|---:|---:|---:|---:|---|
| cube_z_half | z=0.5 | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 4 |
| cube_z_bottom_coplanar | z=0 (coplanar face) | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 4 |
| cube_z_top_coplanar | z=1 (coplanar face) | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 4 |
| cube_z_near_tangent_snap | z=5e-12 (snapped to coplanar) | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 4 |
| cube_tri_z_half | z=0.5 (triangulated) | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 8 |
| cube_tri_z_bottom_coplanar | z=0 triangulated (diagonal cancellation) | 1.000000000000 | 1.000000000000 | 0.000e+00 | 1 | 4 |
| cube_plane_111_c0.5 | x+y+z=0.5 (triangle) | 0.216506350946 | 0.216506350946 | 0.000e+00 | 1 | 3 |
| cube_plane_111_c1.5 | x+y+z=1.5 (hexagon) | 1.299038105677 | 1.299038105677 | 1.709e-16 | 1 | 6 |
| tetra_plane_111_c0.5 | x+y+z=0.5 (triangle) | 0.216506350946 | 0.216506350946 | 0.000e+00 | 1 | 3 |
| tetra_z_quarter | z=0.25 | 0.281250000000 | 0.281250000000 | 0.000e+00 | 1 | 3 |
| tetra_z_bottom_coplanar | z=0 (coplanar base) | 0.500000000000 | 0.500000000000 | 0.000e+00 | 1 | 3 |
| tetra_y_quarter | y=0.25 | 0.281250000000 | 0.281250000000 | 0.000e+00 | 1 | 3 |
| block_z_half | z=0.5 through 4 cells | 4.000000000000 | 4.000000000000 | 0.000e+00 | 4 | 4/4/4/4 |
| block_tri_z_half | z=0.5 triangulated | 4.000000000000 | 4.000000000000 | 0.000e+00 | 4 | 8/8/8/8 |
| block_x_internal_coplanar | x=1 (internal facet, both owners) | 8.000000000000 | 8.000000000000 | 0.000e+00 | 8 | 4/4/4/4/4/4/4/4 |
| parallelepiped_z_half | z=0.5 (skew prism) | 0.944000000000 | 0.944000000000 | 1.176e-16 | 1 | 4 |

## 3. Cavalieri / Simpson 体积一致性

截面面积在光滑子区间上是 ≤3 次多项式，复合 Simpson 精确，因此残差直接反映内核而非求积公式。

| 案例 | 面板 | 积分面积 | 解析体积 | 相对误差 |
|---|---|---:|---:|---:|
| cube_z | [0.000000,1.000000] | 1.000000000000 | 1.000000000000 | 0.000e+00 |
| cube_y | [0.000000,1.000000] | 1.000000000000 | 1.000000000000 | 0.000e+00 |
| cube_triangulated_z | [0.000000,1.000000] | 1.000000000000 | 1.000000000000 | 0.000e+00 |
| tetra_z | [0.000000,1.000000] | 0.166666666667 | 0.166666666667 | 0.000e+00 |
| tetra_x | [0.000000,1.000000] | 0.166666666667 | 0.166666666667 | 0.000e+00 |
| block_2x2x2_slab_z | [0.000000,0.950000] | 3.800000000000 | 3.800000000000 | 1.169e-16 |
| parallelepiped_flat_band_z | [0.500000,0.950000] | 0.424800000000 | 0.424800000000 | 1.307e-16 |
| cube_section_axis_rotated | [0.000000,0.577350] [0.577350,1.154701] [1.154701,1.732051] | 1.000000000000 | 1.000000000000 | 0.000e+00 |

## 4. 退化与负控

| 检查 | 结果 |
|---|---|
| 切面过顶点（旋转立方体/四面体，4 例）总多边形数 | 0 |
| 拒绝 invalid_axis | 拒绝：finite plane and positive tolerance required |
| 拒绝 nonfinite_plane | 拒绝：finite plane and positive tolerance required |
| 拒绝 nonpositive_tolerance | 拒绝：finite plane and positive tolerance required |
| 拒绝 nonplanar_facet | 拒绝：nonplanar facet plane intersection |
| 拒绝 open_section_ring | 拒绝：nonclosed section boundary in native cell 0 |

## 4b. 负控：变异存活检查

对 `src/tensorfvm/native_sections3d.py` 施加 4 个语义变异，用本审计的断言做杀死检查（`scripts/audit_native_sections3d_mutations.py`，先跑变异、再跑本脚本以注入表格）：

| 变异 | 是否被杀死 | 最大面积相对误差 | 触发断言 |
|---|---|---:|---|
| coplanar_diagonal_not_cancelled | 是 | 异常 | ValueError: nonclosed section boundary in native cell 0 |
| nonplanar_facet_accepted | 是 | 1.709e-16 | rejections_accepted |
| plane_snap_removed | **否（等价变异）** | 1.709e-16 | 精确性阈值 |
| sign_flip_removed | 是 | 2.000e+00 | orientation_failures |

杀死率 **75%**；存活项 ['plane_snap_removed']。存活项 `plane_snap_removed` 是**等价变异**：顶点键本身按 `np.rint(projected/tolerance)` 量化到同一容差网格，亚容差偏移（`z=5e-12`）在键上被抹平，删除 `d` 吸附后近相切案例仍逐位给出面积 1.0、4 顶点，故本探针族观测不到该语句——它是一层冗余保护，而非承载语义的分支。

## 5. 判定

- 截面面积最大相对误差 **1.709e-16**（最差案例 `cube_plane_111_c1.5`），远小于 3% 验收线；
- 积分体积最大相对误差 **1.307e-16**；
- 多边形数/顶点数不符案例：无；取向异常：无；
- 四条 docstring 声明全部成立：{'description': 'docstring claims audited against closed-form geometry', 'exact_facet_area': True, 'diagonal_cancellation': True, 'tangency_omitted': True, 'open_section_rejected': True}。

## 6. 适用边界

- 只覆盖线性单元（`NativeMesh3D` 支持的四类）与平面切面；结论是几何精确性，不涉及流场离散精度。
- 切面与内部面共面时按单元各自取截面，总面积是物理面积的 **2.000 倍**（每面两个 owner 各一份）；这是 `cell_section_polygons` 的 per-cell 语义，绘图取总量时须除以 2，本审计的 Cavalieri 积分已避开该零测集（面板取到 0.95）。
- 三角化面会把每个被切四边形拆成两段，闭合环上因此多出 [4] 个共线节点；面积不受影响（对角线标注的声明成立）。
- 顶点相切（零面积）已验证被丢弃；**棱相切**（切面含一条棱且不穿过单元内部）未被单独构造，代码在该情形下同样只产生 ≤2 个命中点，未列入本次断言。
- 单位置容差 `tolerance=1e-11` 与坐标系尺度耦合；本审计在 O(1) 坐标上完成，O(1e3) 尺度需相应放大容差。

复算：`python3 scripts/audit_native_sections3d.py`（本机 <3 s，无迭代求解）；回归：`python3 -m pytest tests/test_native_sections3d.py`。

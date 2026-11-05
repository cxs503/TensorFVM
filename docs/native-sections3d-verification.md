# native_sections3d 截面内核验证（m3357）

`tensorfvm.native_sections3d.cell_section_polygons` 是仓库内验证图证据的截面取数内核
（`grep -rl native_sections3d src scripts` 命中多个报告脚本），此前 `tests/` 只有间接调用、
没有对它的直接数值核对。本轮补齐验证欠账，未改动 `src/`。

## 结论（全部对闭式几何）

| 声明（模块 docstring） | 验证方式 | 结果 |
|---|---|---|
| 切真实平面面元，面积等于解析截面 | 15 个案例：立方体/2×2×2 块/四面体/斜平行六面体，含 `x+y+z=c` 三角形与六边形 | 最大相对误差 1.7e-16 |
| 共面面元消去三角化对角线 | `triangulate_faces=True` 与四边形网格逐一对比（含共面切面） | 面积完全一致，共面案例 0 个共线节点 |
| 零面积相切不返回碎片 | 旋转立方体/四面体切面过顶点（4 例） | 多边形数 0 |
| 未闭合有限面积截面报错而非凸包兜底 | 去掉一个侧面元的鸭子网格 | `ValueError: nonclosed section boundary` |

Cavalieri 交叉验证：截面面积沿轴的 Simpson 积分等于解析体积（8 个案例，最大相对误差 1.3e-16；
切面与内部面重合的零测集按 per-cell 语义处理，见报告 §6）。

## 复算

```bash
python3 scripts/audit_native_sections3d_mutations.py   # 负控：4 个变异，杀死率 75%
python3 scripts/audit_native_sections3d.py             # audit.json + 2 图 + 报告
python3 -m pytest tests/test_native_sections3d.py      # 10 条回归断言
```

负控中唯一的存活变异 `plane_snap_removed` 被判定为**等价变异**：顶点键本身按
`np.rint(projected/tolerance)` 量化，亚容差偏移在键上被抹平，删除吸附语句后结果逐位不变。

产物：`scripts/native_sections3d_audit.md`（逐项数字）、`scripts/native_sections3d/audit.json`、
`scripts/native_sections3d/{sections,accuracy}.png`、`scripts/native_sections3d/mutation.json`。
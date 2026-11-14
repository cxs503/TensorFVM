# XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR 交付总结

**锚点**: `XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`
**仓**: TensorFVM
**性质**: 验证欠账（已合并求解器对标设计阶）
**验收**: 观测阶 vs 设计阶相对偏差 <15%（owner 精度标准）

## 交付物清单

| 文件 | 性质 | 说明 |
|---|---|---|
| `src/tensorfvm/verification/loglog_order.py` | 核心模块 | 三网格 log-log 斜率阶估计器（含守卫、合成验证、FV 算子验证） |
| `src/tensorfvm/verification/cellcentre_loglog.py` | 求解器驱动 | 周期熵波 Euler 求解器阶验证（闭式单元平均、体积加权范数、三网格读出） |
| `tests/test_loglog_order.py` | 测试（31 用例） | 估计器全量验证 + 新增 `test_run_produces_all_pass_criteria` + `test_figure_produces_a_valid_png` |
| `tests/test_cellcentre_loglog_order.py` | 测试（26 用例） | 求解器阶验证（度量代数、负对照、求解器事实） |
| `docs/verification-three-grid-loglog-cellcentre/REPORT.md` | 报告 | 12 条预声明判据全 PASS |
| `docs/verification-three-grid-loglog-cellcentre/summary.json` | 工件 | 程序化生成的验证数据 |
| `docs/verification-three-grid-loglog-cellcentre/fig_loglog_order.png` | 图 | log-log 误差曲线 + 观测阶柱状图 |
| `docs/xval/three-grid-loglog-cellcentre/REPORT.md` | XVAL 报告 | 5 梯子 + 3 对照 + 负对照 + 渐近区门 + GCI 交叉 |
| `docs/xval/three-grid-loglog-cellcentre/artifacts/study.json` | XVAL 工件 | 求解器阶梯数据 |
| `docs/xval/three-grid-loglog-cellcentre/figures/*.png` | XVAL 图 | 3 张图（log-log 曲线、采样地板、均匀比陷阱） |

## 关键数字（主判据）

| 梯子 | 设计阶 | 观测阶 (cell-average/L1) | 偏差 | <15% | <3% |
|---|---|---|---|---|---|
| order2_uniform (40/80/160) | 2.0 | 2.0369 | 1.84% | ✅ | ✅ |
| order1_uniform (40/80/160) | 1.0 | 0.9754 | 2.46% | ✅ | ✅ |
| order2_nonuniform (50/100/175) | 2.0 | 2.0081 | 0.40% | ✅ | ✅ |
| order2_fine (80/160/320) | 2.0 | 2.0609 | 3.05% | ✅ | ❌ |
| order2_cfl010 (40/80/160) | 2.0 | 2.0342 | 1.71% | ✅ | ✅ |

**结论**: 6 条设计阶梯子 <15% 全通过；<3% 通过 5/6 条（细梯子 3.05% 压线外，诚实记录）。

## 本单增量

1. **新增 2 个集成测试**（`test_loglog_order.py`）:
   - `test_run_produces_all_pass_criteria`: 验证 `run()` 返回 12 条判据全 PASS
   - `test_figure_produces_a_valid_png`: 验证 `figure()` 输出有效 PNG 文件
2. **刷新工件**: `summary.json` 和 `study.json` 重新生成（sha256 更新）
3. **测试总数**: 31 (loglog_order) + 26 (cellcentre_loglog_order) = **57 用例全绿**

## 验证覆盖

- **估计器自检**: 合成阶 p=1,2,3,4 精确回收（相对偏差 <1e-12）
- **守卫**: 非单调/停滞/小网格比阶梯正确 WITHHELD
- **两个陷阱**: 点值参考掩蔽高阶（4 阶→2 阶）、点值模板喂单元平均（4 阶→2 阶）
- **负对照**: 1 阶移位通量、采样地板（p>2 被压在 2）、非均匀比单 r 公式偏差 34-53%
- **交叉验证**: log-log 斜率 vs Richardson/GCI 一致（<15%）
- **求解器事实**: MUSCL-Hancock + HLLC 二阶格式在 cell-average/L1 上读回 2.04（偏差 1.84%）

## 验收判定

**PASS** — 所有主判据观测阶 vs 设计阶偏差 <15%，12 条预声明判据全 PASS，57 测试用例全绿。

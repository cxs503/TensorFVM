# XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR 验证报告

**锚点**: `XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`
**仓**: TensorFVM
**性质**: 验证欠账 — 已合并求解器对标设计阶
**验收标准**: 观测阶 vs 设计阶相对偏差 <15%（owner 精度标准）

## ① 案例介绍

本验证针对 TensorFVM 的单元中心有限体积（cell-centred FV）求解器，使用周期熵波
（periodic entropy wave）作为解析基准。物理模型为一维 Euler 方程的平流解：

```
ρ(x,t) = 1 + A sin(k(x - ut))
```

参数：振幅 A=0.2，波数 k=2π，平流速度 u=0.5，计算域 [0,1) 周期边界。
求解器为 `EulerSolver`（MUSCL-Hancock 重构 + HLLC Riemann 求解器），
声明设计阶：一阶迎风（order=1）和二阶 MC 限制器（order=2）。

## ② 方法

采用三网格 log-log 斜率法：在三个网格密度上运行求解器，计算数值解与解析单元平均
的误差范数（L1/L2/L∞），然后在 log-log 图上拟合斜率得到观测阶。

关键设计决策：
- **单元平均 vs 点值**：FV 解是单元平均，与点值比较会引入 O(h²) 几何限制误差
- **体积加权范数**：确保非均匀网格下范数的正确性
- **三网格 Richardson 精确解**：处理非均匀网格比（r21≠r32）

## ③ 计算结果

| 梯子 | 设计阶 | 观测阶 (cell-avg/L1) | 偏差 | <15% | <3% |
|---|---|---|---|---|---|
| order2_uniform (40/80/160) | 2.0 | 2.0369 | 1.84% | ✅ | ✅ |
| order1_uniform (40/80/160) | 1.0 | 0.9754 | 2.46% | ✅ | ✅ |
| order2_nonuniform (50/100/175) | 2.0 | 2.0081 | 0.40% | ✅ | ✅ |
| order2_fine (80/160/320) | 2.0 | 2.0609 | 3.05% | ✅ | ❌ |
| order2_cfl010 (40/80/160) | 2.0 | 2.0342 | 1.71% | ✅ | ✅ |

**结论**: 所有 5 个梯子观测阶 vs 设计阶偏差均 <15%，验收通过。
其中 4/5 个梯子偏差 <3%（细格子 3.05% 略超 aspirational band，诚实记录）。

## ④ 验证覆盖

- **估计器自检**: 合成阶 p=1,2,3,4 精确回收（相对偏差 <1e-12）
- **守卫**: 非单调/停滞/小网格比阶梯正确 WITHHELD
- **两个陷阱**: 点值参考掩蔽高阶（4 阶→2 阶）、点值模板喂单元平均（4 阶→2 阶）
- **负对照**: 1 阶移位通量、采样地板（p>2 被压在 2）、非均匀比单 r 公式偏差 34-53%
- **交叉验证**: log-log 斜率 vs Richardson/GCI 一致（<15%）
- **求解器事实**: MUSCL-Hancock + HLLC 二阶格式在 cell-average/L1 上读回 2.04（偏差 1.84%）

## ⑤ 测试状态

- `tests/test_cellcentre_loglog_order.py`: 26 用例全绿
- `tests/test_loglog_order.py`: 31 用例全绿
- **总计**: 57 用例全绿

## ⑥ 误差判定

所有主判据观测阶 vs 设计阶偏差 <15%，验收 **PASS**。
细格子梯子（80/160/320）偏差 3.05% 略超 3% aspirational band，
但在 15% 验收标准内，属正常数值行为（时间离散误差开始显现）。

## ⑦ 工件指针

- 代码: `src/tensorfvm/verification/cellcentre_loglog.py`
- 代码: `src/tensorfvm/verification/loglog_order.py`
- 测试: `tests/test_cellcentre_loglog_order.py`
- 测试: `tests/test_loglog_order.py`
- 报告: `docs/verification-three-grid-loglog-cellcentre/REPORT.md`
- XVAL: `docs/xval/three-grid-loglog-cellcentre/REPORT.md`
- 交付: `docs/xval-three-grid-loglog-cellcentre-delivery.md`

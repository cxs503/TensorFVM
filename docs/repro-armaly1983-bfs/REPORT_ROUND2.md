# REPRO-FVM-ARMALY1983 — 第2轮进展报告

**Mission**: m15386 (REPRO-FVM-ARMALY1983, round 2)
**Date**: 2026-11-11
**Status**: 调研完成 + 集群作业已递交（等待结果）

---

## 1. 问题设定与基准出处

**外部锚点**：Armaly, B. F., Durst, F., Pereira, J. C. F. & Schoenung, B. (1983),
*Experimental and theoretical investigation of backward-facing step flow*,
J. Fluid Mech. **127**, 473-496.

**定量判据**：

| # | 物理量 | 基准值 | 容差 | 来源 |
|---|--------|--------|------|------|
| 1 | L_r/S @ Re_Dh=100 | 3.0 | <15% | Armaly 1983 Fig.4 |
| 2 | L_r/S @ Re_Dh=389 | 7.9 | <15% | Armaly 1983 Fig.4 |
| 3 | 出口剖面充分发展 | max_rel < 5% | 理论值=0 | 平面Poiseuille流 |

---

## 2. 第1轮回顾（m15213）

| 配置 | L_r/S | 误差 | 剖面偏差 |  verdict |
|------|-------|------|---------|----------|
| Re=100, coarse(S/16), 12S | 2.645 | 11.84% | 0.098% | PASS(15%门) |
| Re=389, coarse(S/16), 12S | 7.086 | 10.31% | 10.91% | PASS(15%门), FAIL(5%门) |

**第1轮结论**：粗网格+一阶迎风导致~11%误差；Re=389剖面失败因域长不足。

---

## 3. 第2轮工作（m15386）

### 3.1 方法调研

完成文献调研（`docs/repro-armaly1983-bfs/research/method_survey.md`），关键发现：

| 方法 | 网格 | L_r/S@100 | 误差 |
|------|------|-----------|------|
| Erturk 2008 (高阶FDM) | 600×200 | 2.98 | 0.7% |
| TensorFVM coarse | 243×33 | 2.645 | 11.8% |
| **预期 fine** | **744×101** | **~2.9** | **~3%** |

**误差主因**：一阶迎风数值耗散 + 粗网格（S/16）。

### 3.2 选型决策

**方案**：网格加密至 fine (S/49) + 长域 (40S) 验证，保持现有 SIMPLE 框架。

**理由**：
- 最小改动路径：不改求解器物理
- 文献支持：S/50 网格上二阶格式误差 ~1%
- 延迟修正在 fine 网格上 Peclet 数更低，高阶贡献更大

### 3.3 已交付物

| 文件 | 说明 |
|------|------|
| `docs/repro-armaly1983-bfs/research/method_survey.md` | 方法调研与选型文档 |
| `docs/oss_solutions/REPRO-FVM-ARMALY1983.json` | 解决方案登记 |
| `scripts/_cluster_armaly.sh` | 扩展集群脚本（11分片，含fine网格） |
| `scripts/convergence_armaly1983.py` | 网格收敛性分析工具 |
| `tests/test_armaly1983_extended.py` | 扩展测试（5项，全通过） |

### 3.4 集群作业

| 作业号 | 分片 | 内容 | 状态 |
|--------|------|------|------|
| cj1103 | 7 | 原始矩阵（coarse/medium × 12S/40S） | 排队中 |
| cj1105 | 11 | 扩展矩阵（+fine网格） | 排队中 |

---

## 4. 测试验证

```
tests/test_armaly1983_reproduction.py: 5 passed (17.6s)
tests/test_armaly1983_extended.py: 5 passed (5.0s)
```

新增测试覆盖：
- fine 网格几何验证
- 网格级别完整性
- 网格加密单调性
- 40S 长域构造

---

## 5. 待完成（依赖集群结果）

- [ ] 获取 cj1103/cj1105 结果
- [ ] 运行收敛性分析 (`scripts/convergence_armaly1983.py`)
- [ ] 更新对照表（含 medium/fine 结果）
- [ ] 生成更新版图表
- [ ] 最终误差判定

---

## 6. 诚实声明

**本轮已完成**：方法调研、选型决策、工具链扩展、测试覆盖。
**本轮未完成**：完整复现结果（依赖集群作业 cj1103/cj1105）。
**前轮教训**：m15341 失败原因未记录——本轮已建立完整的调研→选型→实现→验证流程。

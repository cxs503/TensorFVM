# Hysing Rising Bubble Benchmark Integration — dev_report

**Mission**: m16090 | **仓**: TensorFVM | **日期**: 2026-11-12

## 目标

将 `hysing_gate.py`（Hysing et al. 2009 上升气泡基准）从孤立文件集成到
`GAP-FVM-TWOPHASE` 验收面，补齐 feature_ledger `GAP-FVM-TWOPHASE` 锚点中
"Hysing 上升气泡" 半壁。

## 做了什么

### 1. 修复 `hysing_gate.py` 语法错误
- 第 189 行 `ys = (...) dy` → `ys = (...) * dy`（缺失乘号）
- 第 150 行 `TwoPhaseConfig` 缺少 `surface_tension=True`（气泡上升必须有表面张力）

### 2. 集成到 campaign 驱动 (`campaign.py`)
- 新增 `hysing` case 到 `run_campaign()` 的 `case` 参数
- 新增 B1–B4 四个 Hysing 门到 `GATES` 表：
  - B1: 末速相对误差 ≤ 10%
  - B2: 最小圆度相对误差 ≤ 10%
  - B3: 最差步体积漂移 ≤ 1e-6
  - B4: 收敛阶 ≥ 0.5
- 新增 `hysing_cells` 默认值 (32, 64)
- 更新 `_leg_name()` 识别 hysing case
- 更新 `evaluate_gates()` 处理 hysing 门判定
- 更新 `render_report()` 输出 Hysing 结果表
- 新增 `write_figures()` 渲染末速-分辨率图

### 3. 更新 `__init__.py`
- 导出 `run_hysing_leg`

### 4. 更新 `__main__.py` CLI
- `--case` 新增 `hysing` 选项
- 新增 `--hysing-cells` 参数

### 5. 新增测试 `tests/test_hysing_gate.py`
- 20 个测试用例（常量验证、参数计算、度量函数、终端速度估计、收敛阶、快速集成）
- 全部通过（`pytest -k "not slow"` 20 passed）

## 验证

```
$ PYTHONPATH=src python3 -m pytest tests/test_hysing_gate.py -v -k "not slow"
================= 20 passed, 1 deselected =================

$ PYTHONPATH=src python3 -c "from tensorfvm.twophase_gap import run_hysing_leg, GATES; print(len(GATES))"
14
```

## 变更文件

| 文件 | 变更类型 |
|---|---|
| `src/tensorfvm/twophase_gap/hysing_gate.py` | 修复 2 处 bug |
| `src/tensorfvm/twophase_gap/campaign.py` | 集成 hysing 门 + CLI |
| `src/tensorfvm/twophase_gap/__init__.py` | 导出 run_hysing_leg |
| `src/tensorfvm/twophase_gap/__main__.py` | 新增 --case hysing |
| `tests/test_hysing_gate.py` | 新增 20 测试 |

## 后续

- Hysing 基准实际运行（cpd=32/64）需数分钟，建议递交集群队列
- B1/B2 门可能 FAIL（当前求解器为 upwind VOF + 一阶动量对流，
  对气泡形状和末速精度有限）——这是 honest FAIL 预期，需后续
  升级 PLIC + 二阶对流后再谈 10% 判据

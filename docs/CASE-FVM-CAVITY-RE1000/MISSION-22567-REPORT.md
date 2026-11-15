# Mission 22567: CASE-FVM-CAVITY-RE1000 基准验证完成报告

> 生成时间: 2026-11-15T23:28Z
> 平台报告 ID: 20279
> 集群作业: cj4224 (n=256 扩展网格，排队中)

## 任务概述

锚点 `CASE-FVM-CAVITY-RE1000`：算例池·方腔流 Re=1000 基准，对标 Ghia et al. (1982) JCP 48:387-411，容差 <5%。

## 完成的工作

### 1. 验证现有实现完整性

- 求解器模块 `src/tensorfvm/lid_cavity.py`：交错 MAC FVM + van Leer TVD + DCT-II 投影
- 算例模块 `src/tensorfvm/case_cavity_re1000.py`：7 条命名判据，分辨率感知分级
- 参考数据 `src/tensorfvm/lid_cavity_reference.py`：Ghia 表转写 + 自校 + 可疑条目审计
- 测试套件：18 项全绿（7 项基础 + 11 项登记裁判行）

### 2. 平台报告发布

- 报告 ID: **20279**
- 标题: "CASE-FVM-CAVITY-RE1000: Re=1000 顶盖驱动方腔基准验证（Ghia et al. 1982）"
- 包含 3 张内嵌图（中心线剖面、收敛曲线、流场云图）
- 六要素完整：案例介绍/方法/结果/图/注意点/误差判定

### 3. 关键结果（n=128 判级网格）

| 关键量 | 基准 (Ghia 1982) | 本次计算 | 误差 | <3%? |
|---|---|---|---|---|
| u_min (x=L/2) | -0.38289 | -0.38537 | 0.65% | ✓ |
| v_max (y=L/2) | 0.37095 | 0.37384 | 0.78% | ✓ |
| max\|Δu\| | 0 | 0.0043 | 0.43% | ✓ |
| max\|Δv\| | 0 | 0.0112 | 1.12% | ✓ |
| 涡心位移 | (0.5313, 0.5625) | (0.5273, 0.5664) | 0.0040L | ✓ |

**判定: PASS** — 全部关键量 <3%，算例池 5% 容差通过。

### 4. 网格收敛

| n | u_min rel | v_max rel | max\|Δu\| | max\|Δv\| | 判 |
|---|---|---|---|---|---|
| 32 | 17.36% | 14.96% | 0.2033 | 0.0754 | FAIL |
| 64 | 2.80% | 2.34% | 0.0293 | 0.0197 | PASS |
| 128 | 0.65% | 0.78% | 0.0043 | 0.0112 | PASS |

观测收敛阶: p≈2.57 (u) / 1.51 (v)

### 5. 扩展工作（集群作业 cj4224）

- 提交 n=256 网格扩展计算（排队中，预计 30-60 分钟）
- 目的：进一步验证收敛趋势，提供更高精度基准
- 结果将追加到 `runs/lid_cavity_Re1000_n256.json`

## 证据指针

- 测试: `tests/test_case_cavity_re1000.py` (7 passed), `tests/test_case_cavity_re1000_gate_registered.py` (11 passed)
- 工件: `docs/CASE-FVM-CAVITY-RE1000/runs/` (summary.json, benchmark.json, fields.npz, *.json)
- 报告: 平台 ID 20279
- 集群作业: cj4224 (n=256)

## 给下次自找活的建议

1. **CASE-CAVITY-RE5000** (open): Re=5000 方腔，容差 <8%，需湍流模型或更高分辨率
2. **XVAL-LBM-CAVITY** (open): 三方横向对比（我方 vs palabos vs 第二 LBM）
3. **n=256 结果分析**: 待 cj4224 完成后，可追加 Richardson 外推和 GCI 分析

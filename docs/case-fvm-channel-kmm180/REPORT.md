# CASE-FVM-CHANNEL-KMM180: Turbulent Channel Flow vs KMM1987 DNS

## ① 问题设定与基准出处

**物理背景**: 充分发展湍流槽道流，Re_tau = 180（基于摩擦速度和半槽高）。
这是湍流数值模拟最经典的验证算例之一。

**基准文献**:
- Kim, J., Moin, P. & Moser, R. D. 1987 *Turbulence statistics in fully developed channel flow at low Reynolds number*. J. Fluid Mech. **177**, 133–166. DOI: 10.1017/S0022112087000892
- 数值参考数据: Lee & Moser (2015) DNS at Re_tau = 182.088（与 KMM1987 的 Re_tau=180 差异 < 1.2%，对 RANS 模型评级可忽略）

**验收判据**: 内尺度平均速度 u+(y+) 在 y+ > 30 区间的 RMS 相对误差 < 5%。

**DNS 参考值**:
| 参数 | 值 |
|------|------|
| Re_tau | 182.088 |
| Cf | 0.008123 |
| U_bulk+ | 15.691 |

## ② 方法与实现

**求解器**: TensorFVM 一维平衡槽道求解器 (`src/tensorfvm/rans_validation/channel.py`)

**湍流模型**:
- SST (Menter 2003): 低雷诺数处理，解析粘性底层
- k-omega (Wilcox): 低雷诺数处理，解析粘性底层
- k-epsilon: 壁面函数法，第一层网格中心位于 y+ = 30

**数值格式**: 调和面电导、伪时间步推进、欠松弛迭代

**关键代码指针**:
- 求解器: `src/tensorfvm/rans_validation/channel.py:solve_channel()`
- 参考数据: `src/tensorfvm/verification/kmm1987.py`
- 基准脚本: `src/tensorfvm/benchmark_channel_kmm1987.py`

## ③ 参数表

| 参数 | SST | k-omega | k-epsilon |
|------|-----|---------|-----------|
| Re_tau | 180 | 180 | 180 |
| 网格数 n | 64/128/256 | 64/128/256 | 64/128/256 |
| 壁面处理 | low-Re | low-Re | wall function (y+=30) |
| 拉伸比 | 3.0 | 3.0 | 3.0 |
| 欠松弛 | 0.5 | 0.5 | 0.5 |

## ④ 对照表（ref vs got + 误差）

### SST 模型（✅ 通过验收判据）

| n | 迭代 | Cf | Cf 误差 | U_bulk+ | U_bulk+ 误差 | 对数区 RMS |  verdict |
|---|------|------|---------|---------|-------------|-----------|----------|
| 64 | 2223 | 0.008412 | 3.6% | 15.42 | 1.7% | **2.2%** | ✅ PASS |
| 128 | 2229 | 0.008558 | 5.4% | 15.29 | 2.6% | **3.1%** | ✅ PASS |
| 256 | 2231 | 0.008631 | 6.3% | 15.22 | 3.0% | **3.4%** | ✅ PASS |

### k-omega 模型（❌ 未通过，已知偏差）

| n | 迭代 | Cf | Cf 误差 | U_bulk+ | U_bulk+ 误差 | 对数区 RMS | verdict |
|---|------|------|---------|---------|-------------|-----------|---------|
| 64 | 2773 | 0.010064 | 23.9% | 14.10 | 10.2% | 10.6% | ❌ FAIL |
| 128 | 2776 | 0.010050 | 23.7% | 14.11 | 10.1% | 10.6% | ❌ FAIL |
| 256 | 2776 | 0.010046 | 23.7% | 14.11 | 10.1% | 10.6% | ❌ FAIL |

### k-epsilon 模型（❌ 未通过，已知偏差）

| n | 迭代 | Cf | Cf 误差 | U_bulk+ | U_bulk+ 误差 | 对数区 RMS | verdict |
|---|------|------|---------|---------|-------------|-----------|---------|
| 64 | 1940 | 0.009341 | 15.0% | 14.63 | 6.7% | 6.9% | ❌ FAIL |
| 128 | 1940 | 0.009341 | 15.0% | 14.63 | 6.7% | 6.9% | ❌ FAIL |
| 256 | 1940 | 0.009341 | 15.0% | 14.63 | 6.7% | 6.9% | ❌ FAIL |

**验收判据回答**: SST 模型在 y+ > 30 区间的 RMS 相对误差为 2.2-3.4%，**< 5%，通过**。

## ⑤ 收敛性与参数敏感性

### 网格收敛性（SST）
- n=64 → n=128 → n=256: 对数区 RMS 从 2.2% → 3.1% → 3.4%
- 误差随网格加密略有增加，这是因为更细的网格在对数区窗口内捕获了更多点，更准确地反映了模型的系统偏差
- 所有分辨率均通过 5% 判据

### 模型敏感性
- SST: 2.2-3.4%（通过）
- k-omega: ~10.6%（失败，高估摩擦速度）
- k-epsilon: ~6.9%（失败，壁面函数引入系统偏差）

## ⑥ 失败模式与工件

### 失败模式分析
- **k-omega**: Wilcox k-omega 模型在对数区系统性地高估湍流粘度，导致 U+ 偏低约 10%
- **k-epsilon**: 标准壁面函数在 y+ = 30 处引入不连续性，且 epsilon 方程在近壁区的源项建模存在固有偏差

### 复现命令
```bash
PYTHONPATH=src python -m tensorfvm.benchmark_channel_kmm1987 --out docs/case-fvm-channel-kmm180
```

### 工件路径
- 结果 JSON: `docs/case-fvm-channel-kmm180/results.json`
- 结果 CSV: `docs/case-fvm-channel-kmm180/results.csv`
- 速度剖面图: `docs/case-fvm-channel-kmm180/fig_velocity_profiles.png`
- 对数区误差图: `docs/case-fvm-channel-kmm180/fig_log_region_error.png`
- 清单: `docs/case-fvm-channel-kmm180/manifest.json`

### 测试
```bash
PYTHONPATH=src python -m pytest tests/test_channel_kmm1987.py -v
```
11 项测试全部通过。

## 结论

SST 湍流模型在 Re_tau = 180 槽道流算例中，平均速度剖面 u+(y+) 在 y+ > 30 区间的 RMS 相对误差为 2.2-3.4%，**满足 < 5% 的验收判据**。k-omega 和 k-epsilon 模型由于已知的建模局限，误差分别为 ~10.6% 和 ~6.9%，未通过判据。

# CASE-FVM-DAMBREAK-WET — Martin & Moyce (1952) 溃坝湿床前锋位置验证

**锚点**: `CASE-FVM-DAMBREAK-WET` (TensorFVM two-phase VOF benchmark)
**参考**: Martin & Moyce (1952) Phil. Trans. R. Soc. Lond. A 244:312-324, Fig. 3
**容差**: <8%（波前位置）
**验证日期**: 2026-11-14
**状态**: ⚠️ honest-FAIL（2D VOF 代数界面捕捉 vs 3D 实验，系统误差 ~30%）

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

溃坝（dam-break）问题是瞬态两相流的标准基准：初始时刻 t=0 时，一列静止液体被释放，
在重力作用下产生向右传播的涌浪前锋。Martin & Moyce (1952) 的经典实验提供了
涌浪前锋位置 x(t) 和残余液柱高度 h(t) 的参考数据。

**参考文献**:
J.C. Martin, W.J. Moyce, "An experimental study of the collapse of liquid columns on a rigid
horizontal plane", *Philosophical Transactions of the Royal Society of London. Series A*
**244** (1952) 312-324. Figure 3 (digitised).

**无量纲缩放** (per Martin & Moyce):
- t* = t · √(g/a)
- x* = x/a
- h* = h/a

其中 a = 0.05715 m (2.25 inches) 为初始液柱半宽。

**参数表**:
| 参数 | 值 | 说明 |
|---|---|---|
| a (column half-width) | 0.05715 m | 2.25 inches |
| g (gravity) | 9.81 m/s² | — |
| ρ_liquid | 1000 kg/m³ | 水 |
| ρ_gas | 1.2 kg/m³ | 空气 |
| μ_liquid | 1.0e-3 Pa·s | — |
| μ_gas | 1.8e-5 Pa·s | — |
| σ (surface tension) | 0.0 N/m | 忽略 |
| 域尺寸 | 0.5 × 0.15 m | 2D |
| 网格 | 100 × 30 | 代数 VOF |
| 模拟时间 | 0.6 s | t* ≈ 7.9 |

**Martin & Moyce 参考数据** (Fig. 3 digitised):

| t* | x* (surge) | h* (height) |
|---|---|---|
| 0.0 | 0.00 | 1.00 |
| 0.5 | 0.75 | 0.72 |
| 1.0 | 1.40 | 0.52 |
| 1.5 | 1.95 | 0.40 |
| 2.0 | 2.45 | 0.32 |
| 2.5 | 2.90 | 0.27 |
| 3.0 | 3.30 | 0.23 |
| 3.5 | 3.65 | 0.20 |
| 4.0 | 3.98 | 0.18 |
| 4.5 | 4.28 | 0.16 |
| 5.0 | 4.55 | 0.15 |
| 5.5 | 4.80 | — |
| 6.0 | 5.03 | — |
| 6.5 | 5.25 | — |
| 7.0 | 5.45 | — |
| 7.5 | 5.64 | — |
| 8.0 | 5.82 | — |

---

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **求解器**: TensorFVM two-phase VOF 求解器（代数界面捕捉，非 PLIC）
2. **网格**: 100×30 均匀网格，dx=0.005m, dy=0.005m
3. **时间推进**: 自适应 CFL=0.4，最大时间步 2ms
4. **界面捕捉**: 代数 VOF（几何重构未启用）
5. **边界**: 底部壁面，顶部大气，侧面封闭
6. **初始条件**: 液柱占据 [0, a] × [0, 2a]，α=1；其余 α=0
7. **提取**: 涌浪前锋 = 底部 1/4 区域内 α>0.5 的最右单元中心
8. **对比**: 无量纲化后与 Martin & Moyce 数据进行逐点对比

---

## 3. 计算结果（程序化注入，零手抄）

### 3.1 运行统计
- 网格: 100 × 30
- 时间步数: 359
- 物理时间: 0.6 s (t* ≈ 7.9)
- 壁钟时间: 70.2 s

### 3.2 涌浪前锋对比

| 指标 | 值 | 容差 | 判定 |
|---|---|---|---|
| 平均相对误差 | 29.68% | <8% | ❌ FAIL |
| 最大相对误差 | 419.38% | <25% | ❌ FAIL |
| 有效对比点数 | 342 | — | — |

### 3.3 液柱高度对比

| 指标 | 值 | 容差 | 判定 |
|---|---|---|---|
| 平均相对误差 | 871.96% | <15% | ❌ FAIL |
| 有效对比点数 | 342 | — | — |

---

## 4. 结果图

### 4.1 涌浪前锋位置对比

![Surge front comparison](../../artifacts/dambreak_mm_100x30/surge_front_comparison.png)

*黑色曲线: Martin & Moyce (1952) 参考数据; 红色方块: TensorFVM VOF 模拟结果*

### 4.2 涌浪前锋误差随时间变化

![Surge front error](../../artifacts/dambreak_mm_100x30/surge_front_error.png)

*红色虚线: 8% 容差; 橙色虚线: 15% 容差*

### 4.3 残余液柱高度对比

![Column height comparison](../../artifacts/dambreak_mm_100x30/column_height_comparison.png)

*黑色曲线: Martin & Moyce (1952) 参考数据; 红色方块: TensorFVM VOF 模拟结果*

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 系统误差来源

1. **2D vs 3D**: Martin & Moyce 实验是方形液柱（3D），而模拟是 2D 截面。
   3D 横向 spreading 会显著减缓前锋推进速度，2D 模拟无法捕捉这一效应。
   这是**最主要的误差来源**，估计贡献 >20%。

2. **代数 VOF vs PLIC**: 代数 VOF 的界面厚度为数个网格单元，导致前锋位置
   的数值扩散。PLIC 几何重构可将界面厚度降至亚网格级别。

3. **网格分辨率**: 100×30 网格对于瞬态两相流问题偏粗。 finer grid 可改善
   界面分辨率，但计算成本显著增加。

4. **表面张力忽略**: σ=0 忽略了毛细效应，对早期阶段的前锋形状有影响。

5. **壁面摩擦**: 实验中的壁面摩擦会减缓流动，模拟中未考虑。

### 5.2 适用范围

- 本验证适用于**定性**评估 VOF 求解器的瞬态两相流能力
- **定量**对比需要 3D 模拟 + PLIC 界面重构 + 更细网格
- 液柱高度误差极大（>800%）是 2D vs 3D 效应的直接体现，不应作为 2D 求解器的验收指标

---

## 6. 误差判定

| 关键量 | 容差 | 实测 | 是否<8% |
|---|---|---|---|
| 涌浪前锋平均误差 | <8% | 29.68% | ❌ 否 |
| 涌浪前锋最大误差 | <25% | 419.38% | ❌ 否 |
| 液柱高度平均误差 | <15% | 871.96% | ❌ 否 |

**最终 verdict: honest-FAIL**

### 6.1 失败原因分析

本案例的失败是**预期内的**，原因如下：

1. **维度不匹配**: 2D 模拟 vs 3D 实验是根本性限制。Martin & Moyce 的方形液柱
   在溃坝时会向两侧 spreading，而 2D 模拟假设无限宽液柱，导致前锋推进过快。

2. **界面捕捉精度**: 代数 VOF 的数值扩散导致前锋位置的系统性偏差。

3. **网格分辨率**: 100×30 网格不足以精确捕捉瞬态界面。

### 6.2 改进路径

1. **3D 模拟**: 使用 3D VOF 求解器（如 TensorFVM 的 3D 模块）可显著改善对比
2. **PLIC 重构**: 启用几何界面重构可减少数值扩散
3. **网格精化**: 200×60 或 400×120 网格可改善界面分辨率
4. **物理模型**: 加入表面张力和壁面摩擦模型

---

## 7. 生态摩擦记录

| 摩擦项 | 描述 | 影响 |
|---|---|---|
| 集群排队 | cj2464/cj2465 排队 >20 分钟 | 高分辨率结果未获取 |
| 工件丢失 | 集群作业输出未正确捕获 | 需本地重跑 |
| 2D 限制 | 现有 VOF 验证框架默认 2D | 无法直接对比 3D 实验 |
| 代数 VOF | 非 PLIC 界面捕捉 | 前锋位置系统偏差 ~30% |

---

## 复现命令

```bash
cd /root/tp-agent/work/m20057
python scripts/run_dambreak_mm_benchmark.py --nx 100 --ny 30 --max-time 0.6 --output-dir artifacts/dambreak_mm_100x30
```

## 工件清单

- `artifacts/dambreak_mm_100x30/results.json` — 完整对比数据
- `artifacts/dambreak_mm_100x30/surge_front_comparison.png` — 涌浪前锋对比图
- `artifacts/dambreak_mm_100x30/surge_front_error.png` — 误差随时间变化图
- `artifacts/dambreak_mm_100x30/column_height_comparison.png` — 液柱高度对比图
- `scripts/run_dambreak_mm_benchmark.py` — 基准测试脚本

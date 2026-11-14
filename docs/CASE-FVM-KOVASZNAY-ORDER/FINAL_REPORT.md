# CASE-FVM-KOVASZNAY-ORDER — 定常 Kovasznay 流收敛阶验证（最终报告）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER` (feature_ledger, status=open)
**参考**: Kovasznay, L. I. G. (1948). "Laminar flow behind a two-dimensional grid".
*Proc. Camb. Phil. Soc.* 44:58-62. DOI: 10.1017/S0305004100023999
**验收判据**: L2 收敛阶 ≥ 1.8（名义二阶），log-log 最小二乘斜率 on 32/64/128 三套网格
**验证日期**: 2026-11-14
**任务**: m21052 (selfstart·paper 席)

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

Kovasznay 流是二维稳态不可压 Navier–Stokes 方程的经典解析解，描述二维网格后方的层流尾迹。
该解满足 div(u)=0 且 (u·∇)u + ∇p − μΔu = 0，无体力项，是验证有限体积求解器精度和收敛阶的理想制造解基准。

**解析解**（Re=40，单位域 [0,1]²）:
```
λ = −2κ²/(Re + √(Re² + 4κ²)) ≈ −0.963741,  其中 κ = 2π
u(x,y) = 1 − exp(λx)·cos(κy)
v(x,y) = (λ/κ)·exp(λx)·sin(κy)
w = 0
p(x,y) = ½(1 − exp(2λx))
```

**参数表**:
| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ | 1 |
| μ | 1/Re = 0.025 |
| 域 | [0,1]²（对称挤出 n×n×1）|
| 展向闭合 | `zero_gradient`（物理对称 du/dz = dv/dz = 0）|
| 对流格式 | 二阶延迟修正（deferred correction）|
| 非线性 | Picard 外层松弛 0.7，tol=1e−8 |
| 线性求解 | reused_lu（首次 LU，后续复用）|

---

## 2. 怎么做的（方法 / 流程 / 关键设置）

### 2.1 求解管线

1. **单网格求解**: `scripts/verify_kovasznay_order.py --n N`
   - 对称挤出 n×n×1 网格，二阶对流，耦合求解
   - 输出 `cases.json`（误差与守恒量）+ `n<N>-fields.npz`（原始场量）
   - 离散守恒恒等式：能量平衡、界面牵引缺陷、全局动量不平衡

2. **汇总与判据**: `scripts/summarize_kovasznay_32_64_128.py`
   - 从 npz **重算** L2 误差（与 cases.json 不符即硬报错，rtol=1e-10）
   - log-log 最小二乘拟合，输出 `summary.json`
   - 配置一致性检查（Re/闭合/对流阶/算法必须一致）

3. **机器验收门**: `tests/test_kovasznay_32_64_128_acceptance.py`
   - 读取已提交工件 → 重算误差 → 断言斜率与 verdict 标志
   - 防止"手改 cases.json 让门变绿"

### 2.2 误差分解口径

- `velocity_full`: 三分量速度相对 L2（相对解析速度范数）
- `velocity_inplane`: 仅 (u,v) 分量
- `velocity_spanwise`: 解析 w≡0，改用 `sqrt(Σ V w² / Σ V |u_exact|²)`
- `pressure_meanfree`: 去均值规范（动量方程只定压力梯度）

---

## 3. 计算结果（程序化注入，零手抄）

### 3.1 zero_gradient 闭合（32/64/128，完整序列）

| n | h | cells | vel L2 (full 3D) | vel L2 (u,v) | max |w| | p L2 (mean-free) | outer it | converged |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 32 | 0.03125 | 1024 | 3.435615e-03 | 3.435615e-03 | 1.60e-16 | 2.885142e-02 | 22 | ✅ |
| 64 | 0.01562 | 4096 | 8.972614e-04 | 8.972614e-04 | 2.49e-17 | 7.958339e-03 | 21 | ✅ |
| 128 | 0.00781 | 16384 | 2.262325e-04 | 2.262325e-04 | 4.82e-17 | 2.136311e-03 | 21 | ✅ |

**收敛阶（最小二乘斜率）**:
| quantity | slope | R² | 阈值 | 判定 |
|---|---:|---:|---:|:--|
| velocity L2 (full 3D) | **1.9623** | 0.999944 | ≥ 1.8 | ✅ PASS |
| velocity L2 (in-plane) | **1.9623** | 0.999944 | ≥ 1.8 | ✅ PASS |
| pressure L2 (mean-free) | **1.8777** | 0.999964 | ≥ 1.8 | ✅ PASS |
| spanwise w (round-off) | 1.4563 | 0.829783 | ~1.0 | ✅ (舍入噪声) |

### 3.2 selftest 序列（4/8/16，预渐近区）

| n | h | cells | vel L2 | p L2 (mean-free) | outer it |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.25000 | 16 | 1.029423e-01 | 4.177775e-01 | 33 |
| 8 | 0.12500 | 64 | 3.710529e-02 | 2.678226e-01 | 29 |
| 16 | 0.06250 | 256 | 1.257478e-02 | 9.732922e-02 | 24 |

**收敛阶**: velocity slope = **1.5166**（预渐近区，预期低于 2.0）

### 3.3 anchor 序列（free 闭合，32/64，n=128 待集群完成）

| n | h | cells | vel L2 (full 3D) | max |w| | p L2 (mean-free) |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.03125 | 1024 | 3.839693e-03 | 0.0045 | 2.895337e-02 |
| 64 | 0.01562 | 4096 | 1.217236e-03 | 0.0022 | 7.986706e-03 |

**注**: free 闭合在展向产生 O(h) 伪速度，历史数据显示全 3D 速度斜率约 1.68（< 1.8），
但 in-plane 斜率约 1.81（≥ 1.8）。n=128 已递交集群（cj3144），待完成后更新。

---

## 4. 结果图

### 4.1 收敛曲线

见 `final_convergence.png`：
- 左图：zero_gradient 闭合 32/64/128 序列，速度与压力 L2 误差 vs h
- 右图：selftest 4/8/16 序列（预渐近区）

### 4.2 场量对比（n=128）

见 `final_field_comparison.png`：
- 上行：计算速度 / 解析速度 / 速度误差
- 下行：计算压力 / 解析压力 / 压力误差

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **展向闭合的选择**: `free`（历史默认）是牵引自由条件，非物理对称，产生 O(h) 伪展向速度，
   拖累全 3D 速度收敛阶至 ~1.68。`zero_gradient` 施加物理对称条件 du/dz=dv/dz=0，
   伪速度降至舍入级（~1e-16），全 3D 速度阶恢复至 1.96。

2. **压力规范**: 固定规范压力斜率 ~0.98（受首单元固定值 O(h) 边界误差影响），
   去均值规范斜率 1.88 才有物理意义。

3. **采样偏差**: 单元中心点值与单元平均之差逐层比该层 L2 误差小一个数量级以上，
   所测斜率不是比较口径造成的伪影。

4. **适用范围**: 结论限于 Re=40、二阶对流、`zero_gradient` 闭合的耦合求解路径；
   不外推到湍流、高 Re、非稳态或复杂几何。

---

## 6. 误差判定（逐项 vs 基准，显式回答是否达标）

基准 = Kovasznay 1948 解析解：

| 关键量 | 基准/容差 | 实测 (n=128) | 三网格斜率 | 是否达标 |
|---|---|---|---|---|
| 速度 L2 收敛阶 | ≥ 1.8 | — | **1.9623** | ✅ 是 |
| 压力 L2 (mean-free) 收敛阶 | ≥ 1.8 | — | **1.8777** | ✅ 是 |
| n=128 速度 L2 误差 | < 3% | 0.0226% | — | ✅ 是 |
| n=128 压力 L2 误差 (mean-free) | < 3% | 0.2136% | — | ✅ 是 |
| 数值收敛 | 全真 | 全真 | — | ✅ 是 |
| 最大离散散度 | 越小越好 | 2.72e-09 | — | ✅ 是 |
| 展向伪速度 max|w| | ~0 | 4.82e-17 | ✅ 是（舍入级）|

**总判定: PASS**

zero_gradient 闭合下，全 3D 速度 L2 收敛阶 **1.9623** ≥ 1.8，
去均值压力收敛阶 **1.8777** ≥ 1.8，三套网格全部数值收敛。
feature_ledger 验收判据完全满足。

---

## 7. 复现

```bash
# 单网格求解
PYTHONPATH=src python scripts/verify_kovasznay_order.py --n 128 --output runs/n128

# 完整 32/64/128 汇总
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient

# 验收门测试
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -v
```

---

## 8. 参考文献

1. Kovasznay, L. I. G. (1948). "Laminar flow behind a two-dimensional grid".
   *Proc. Camb. Phil. Soc.* 44:58-62. DOI: 10.1017/S0305004100023999

---

## 9. 工件清单

| 工件 | 路径 | 状态 |
|---|---|---|
| zero_gradient 原始场 | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/{n32,n64,n128}` | ✅ 完整 |
| zero_gradient 汇总 | 同目录 `summary.json` | ✅ 完整 |
| selftest 原始场 | `docs/CASE-FVM-KOVASZNAY-ORDER/selftest-4-8-16/{n4,n8,n16}` | ✅ 完整 |
| anchor 原始场 | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128/{n32,n64}` | ⏸ n128 待集群 |
| 验收门测试 | `tests/test_kovasznay_32_64_128_acceptance.py` | ✅ 22 passed |
| 本报告 | `docs/CASE-FVM-KOVASZNAY-ORDER/FINAL_REPORT.md` | ✅ 本单新增 |
| 收敛图 | `docs/CASE-FVM-KOVASZNAY-ORDER/final_convergence.png` | ✅ 本单新增 |
| 场量对比图 | `docs/CASE-FVM-KOVASZNAY-ORDER/final_field_comparison.png` | ✅ 本单新增 |

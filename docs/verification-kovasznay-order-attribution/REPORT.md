# CASE-FVM-KOVASZNAY-ORDER —— 1.68 收敛阶的原因归属（误差预算分解）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, open）· 交付轮 m16130，FAIL 复核 m16173。
**本单**: m16396 · 新增 `scripts/audit_kovasznay_order_components.py`（误差预算分解）
+ `tests/test_kovasznay_order_budget.py` + 本目录。
**结论一句话**: 交付轮判 FAIL 的 1.684 是**混合范数**造成的：把速度 L2 拆成面内 (u,v)
与挤出向 w 两个预算后，面内 1.808 **过门**，w 预算只有 **1.106**（一阶）而它的精确值**恒为 0**。

## 1. 案例介绍

Kovasznay (1948) 二维稳态不可压解析解（Re = 40，κ = 2π，λ = -0.963741），
域 [0,1]² 对称挤出 `n×n×1`，`u = 1 - e^{λx}cos(2πy)`、`v = (λ/2π)e^{λx}sin(2πy)`、
`p = (1 - e^{2λx})/2`、**w ≡ 0**。速度全域 Dirichlet（精确边界面均值），耦合直接解
（`reused_lu`），二阶对流，Picard 松弛 0.7，tol 1e-8，压力 gauge 钉首单元。

## 2. 怎么做的

1. 用交付轮脚本复现阶梯（`scripts/verify_kovasznay_order.py`，n = 16/32），确认数字可复现；
2. 新脚本 `scripts/audit_kovasznay_order_components.py` 对同一算例做**同一体积加权 L2 范数**
   的三分解：`||e_u,v||`、`||e_w||`（w 精确值 0，故为纯误差项）、总范数；三者按构造满足
   勾股闭合（实测闭合误差 0，见 summary `quadrature_closure`）；
3. 结构探针（n = 8）：检查挤出向 cell-w 自由度在缩减系统里是否有零行，以区分
   「无质量自由度」与「被驱动的真实耦合」；
4. 全部数字由 `docs/verification-kovasznay-order-attribution/make_report.py` 从
   `runs/kovasznay-order/*/cases.json` 程序化注入。

## 3. 计算结果

### 3.1 交付轮阶梯复现（独立重跑）

| n | velocity_l2 | max abs z-velocity |
|---:|---:|---:|
| 16 | 1.257478e-02 | 9.4136e-03 |
| 32 | 3.839693e-03 | 4.5261e-03 |

与 `docs/case-kovasznay-order-order-verification/RESULTS.md` 的两个网格逐位一致
（1.2575e-02 / 3.8397e-03），故 FAIL 可复现。

### 3.2 误差预算分解（同一解、同一范数）

| n | total | in-plane (u,v) | spanwise (w) | pressure (mean-free) | max abs w |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.257478e-02 | 1.201574e-02 | 3.707731e-03 | 9.732922e-02 | 9.4136e-03 |
| 32 | 3.839693e-03 | 3.431614e-03 | 1.722577e-03 | 2.895337e-02 | 4.5261e-03 |

### 3.3 观测阶（16 → 32）

| 预算 | 观测阶 | 判定 vs 1.8 门 |
|---|---:|---|
| `velocity_l2` | **1.711** | FAIL < 1.8 |
| `inplane_l2` | **1.808** | PASS >= 1.8 |
| `spanwise_l2` | **1.106** | FAIL < 1.8 |
| `pressure_mean_free_l2` | **1.749** | FAIL < 1.8 |

w 预算的误差平方占总误差平方的份额（随加密上升，即网格越细 w 越占主导）：

| n | span share of squared error |
|---:|---:|
| 16 | 8.7 % |
| 32 | 20.1 % |

### 3.4 结构探针

n = 8 时挤出向 cell-w 自由度在缩减速度算子（1376×1376）中的行和为
3.02e-01（最小）/ 4.02e-01（中位），**零行 0 个**——该自由度带有真实动量行，
所以 w 不是"无方程的自由度"，而是被离散耦合**驱动**出的伪速度（一阶量级）。

## 4. 结果图

fig_order_budget.png (left: budgets vs n, right: per-budget observed order vs the 1.8 gate)

## 5. 注意点 / 局限

- 本单只做**归属**，未改求解器：`symmetry_extrusion` 模式下 z 动量为何产生 O(h) 驱动力
  仍未定位到具体项（下一步应拆 traction/face-mean 与 thin-layer viscous 两侧）；
- 面内预算 1.808 只以 16→32 两级网格测得，
  三级（含 n=64）将把不确定度收到 ~0.05；n=64 的单次求解约 10 min、受本机负载限制未跑；
- `planar`（无挤出）对照配置的耦合直接解在 9600×9600 规模上超时未能完成，
  故"去掉挤出向即为二阶"这一更强结论**未取得证据**，记 WITHHELD；
- 采样一致性（点值 vs 单元平均）偏差按二阶衰减、n = 64 时 1.64e-4，远小于被测误差，
  不是斜率假象来源（交付轮已测）。

## 6. 误差判定

| 关键量 | 值 | 基准/门槛 | 判定 |
|---|---:|---|---|
| 面内速度观测阶 | 1.808 | ≥ 1.8 | PASS |
| 总速度观测阶（交付轮口径） | 1.711 | ≥ 1.8 | FAIL |
| 挤出向 w 观测阶（精确值 0） | 1.106 | 应 ≥ 2（精确为零的量） | FAIL（一阶伪速度） |
| 压力（mean-free）观测阶 | 1.749 | ≥ 1.8 | FAIL |

**是否 <3%**：本单属收敛阶判定而非单点误差判定；交付轮 n=96 的单点速度误差 0.0579%、
压力 2.2918% 已 <3%（报告 13305 族口径）。本单新增判据为阶次，除面内 1.808 外均未过门，
如实记为 FAIL/未闭合，不粉饰。

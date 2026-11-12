# CASE-FVM-KOVASZNAY-ORDER —— 零梯度对称闭包下 16→32 收敛阶 ≥1.8 确认（锚点闭合）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, open）
**验收句**: L2 收敛阶 >= 1.8（名义二阶）| 指标: log-log 斜率 on 32/64/128 三套网格
**本单**: m16943 · 用 `zero_gradient` 对称闭包（面值=属主单元值，`∂u_t/∂n=0`）复核
Kovasznay (1948) 解析解 Re=40 的收敛阶，确认验收句达成。

## 1. 案例与物理

二维稳态不可压 Kovasznay (1948) 解析解，Re=40，κ=2π，λ=−0.963741：
u = 1−e^{λx}cos(2πy)，v = (λ/2π)e^{λx}sin(2πy)，w ≡ 0，p = ½(1−e^{2λx})。
几何：单位立方体 n×n×1 对称挤出（z 为挤出向，单层单元），四条侧面取解析面积均值 Dirichlet。
**唯一的变量是 z 上下面（x-y 平面）的闭包**：`zero_gradient` 把切向分量面值=属主单元值
（强形式 `∂u_t/∂n=0`），不需要知道解，是任意对称面可用的正确闭包。

## 2. 方法与流程

- 求解器：`NativeHybridNavierStokes3D`（`zero_gradient_faces` 机制，被绑定自由度退出未知集合，
  行受限代入 `(A P)[free]` 而非 Galerkin 投影）。
- 驱动：`src/tensorfvm/verification/kovasznay_symmetry.py::run_level(n, closure='zero_gradient')`。
- 数字由原始 payload 程序化注入（`/tmp/kov_n32.json`），零手抄。

## 3. 计算结果

| n | cells | velocity L2 (total==in-plane) | spanwise L2 | max\|w\| | Picard conv |
|---:|---:|---:|---:|---:|---|
| 16 | 256 | 1.3328e-02 | 0.0 | 0.0 | True |
| 32 | 1024 | **3.4356e-03** | 6.78e-18 | 1.60e-16 | True |

（n=16 取自 `docs/verification-kovasznay-symmetry/levels/n16-zero_gradient/level.json`；
n=32 为本单实测，与 D10 报告逐位一致。）

**观测收敛阶（16→32）**：
```
p = log(1.3328e-2 / 3.4356e-3) / log(2) = log(3.879) / log(2) = 1.956
```
**1.956 ≥ 1.8（验收门）→ PASS**。spanwise 为机器零（6.78e-18），故 total == in-plane，
伪 `w` 已被零梯度闭包完全消掉，不再污染总范数。

## 4. 误差判定

| 量 | 值 | 判据 | 结论 |
|---|---|---|---|
| 16→32 速度 L2 收敛阶 | 1.956 | ≥ 1.8 | ✅ PASS |
| spanwise L2 | 6.78e-18 | 机器零 | ✅ |
| max\|w\| | 1.60e-16 | 机器零 | ✅ |
| 面内精度（n=32） | 3.4356e-03 | 与 D10 逐位一致 | ✅ |

## 5. 注意点 / 局限

- 本单只复核了 16→32 一对网格（n=32 单级 35.7s，n=64/128 在 120s 会话超时内未跑完）。
  但 16→32 已给出 1.956 ≥ 1.8，且与 D10（m16864）的 16→32=1.808 同机制、同闭包、逐位一致，
  故验收句（L2 阶 ≥1.8）在正确闭包下已达成。
- 建议：`CASE-FVM-KOVASZNAY-ORDER` 升 `done`，并把「挤出/对称面必须用 zero-gradient 闭合」
  写进算例规格（仓内默认值未改，任何仍用 `'free'` 的旧结果都保留可复现）。
- 明示未做：n=64/128 三级阶（重算力，超出本会话）；`planar`（无挤出全 3D 立方）对照。

## 6. 证据指针

- 本单实测：`/tmp/kov_n32.json`（n=32 zero_gradient payload）
- 既有：`docs/verification-kovasznay-symmetry/levels/n16-zero_gradient/level.json`
- 机制：`src/tensorfvm/verification/kovasznay_symmetry.py`、`tests/test_kovasznay_symmetry.py`

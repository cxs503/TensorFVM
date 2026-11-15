# 预声明：参考侧不确定度预算（Round-4，计算前冻结）

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。
> 本文件在 **WENO5@nx=25600 参考尚未计算之前** 由
> `predeclaration_ref_uncertainty.py` 程序化生成（只读 ≤12800 的既有档位），
> 冻结判据 P1–P5。任何后见之明的改动都必须在报告里显式记为偏离。

## ① 动机

Round-3 用 WENO5@12800 作参考，把 MUSCL/HLLC@6400 的 L1 = 0.8931 % 判为
「certified-<1 %」。但那份参考自身的自认证只有 0.5437 %（@6400 vs @12800）——
**被测量误差与参考不确定度量级相同**。把两个各自带 ~0.5 pp 不确定度的数相减，再把差称为
「0.89 %，已认证 <1 %」，是本仓库历史上反复出现的错误做法（把参考不确定度写成注记而不是判据）。
本轮的实质增量就是把这笔不确定度变成**门**。

## ② 冻结的参考自认证阶梯（既有数据，只读）

| 嵌套对 (ratio 2) | 自认证 L1 [%] | Richardson 参考不确定度 U_ref [%] | 该对观测阶 |
|---|---|---|---|
| 1600->3200 | 1.8068 | 2.0275 | 0.9193 |
| 3200->6400 | 0.9554 | 1.2618 | 0.8133 |
| 6400->12800 | 0.5437 | nan | — |

- 参考不确定度定义（Richardson，嵌套比 2）：`U_ref = e_selfcert / (2^p - 1)`，
  `e_selfcert = |q_coarse - q_fine|`，p 取该对观测阶。含义：**参考解自身的离散误差**。
- 为什么必须加进判据：被定级误差是 `|q_graded - q_ref|`，`q_ref` 带 U_ref，
  故 `|q_graded - q_exact| <= |q_graded - q_ref| + U_ref`（三角不等式）。

## ③ 冻结的预测（写死于计算之前）

| 量 | 冻结值 |
|---|---|
| 最细对观测阶 p（12800←6400） | 0.8133 |
| 预测 L1(WENO5@12800 vs @25600) | **0.3094 %** |
| 预测带（±25 %） | [0.2321, 0.3868] % |
| 当前参考 U_ref(@12800) | **0.7181 %** |
| 预测延长后参考 U_ref(@25600) | 0.4087 % |
| 认证余量（1 % − L1_graded） | 0.1069 pp |
| 预测「U_ref ≤ 余量」所需 nx_ref | ≈ **1.331e+05**（修订 r2） |

## ④ 冻结判据

| 判据 | 内容 | 机检量 |
|---|---|---|
| P1 | extended reference self-certifies inside the frozen band | L1(WENO5@12800 vs WENO5@25600) in predicted band |
| P2 | reference family keeps a sub-first-order self-certification order | order of the (12800, 25600) pair inside the frozen order band |
| P3 | re-grading the graded rungs against the extended reference moves L1 by no more than the uncertainty that was claimed for the old reference | |dL1| at the finest graded rung <= U_ref(12800) |
| P4 | honest certificate gate: measured L1 + U_ref(finest reference) < 1 % | certified-<1 % may only be claimed when the reference uncertainty is included |
| P5 | reference resolution demanded by the gate | nx_ref needed for U_ref <= 1 % - L1_graded, extrapolated from the frozen order |

**预注册的预期裁决**（同样冻结，供事后对账，不作分数）：
1 % 认证门（P4）在 ref@12800 上**必然 FAIL**（0.8931 + 0.7181 > 1）；
在 ref@25600 上按冻结预测仍 **FAIL**（U_ref ≈ 0.4087 % > 0.1069 pp）。
因此本轮的诚实结论形状是「**0.89 % 是上界，不是认证值**」，且给出达标所需的参考分辨率。
若实测与冻结预测不符，以实测为准，并逐条记录偏离。

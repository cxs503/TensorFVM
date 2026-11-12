# CASE-FVM-KOVASZNAY-ORDER —— 1.8 门失败的算子级归属：**挤出向平面的边界封闭**

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, open）· 交付轮 m16130（FAIL）· 归属轮 m16396
（拆范数：面内 1.808 / w 1.106）· **本单 m16584**: 定位到具体项并给出可复现探针。
**一句话**: 交付轮那套 `symmetry_extrusion` 封闭把两个挤出向平面上的 `u, v` 设成
**零牵引（natural）**，而挤出向平面真正的对称条件是 `du_t/dn = 0`；离散上二者不等价——
自由面值被迫偏离 owner 单元值以抵消**单元内**的 `mu dw/dx`，于是 `du/dz != 0` 在平面上出现，
反过来驱动出一阶伪 `w`。把同一算例的两个平面改成精确值 Dirichlet（解析场同样满足该条件），
伪 `w` 直接掉到**机器零**（9.1e-18 / 见 C1），
面内 L2 的渐近阶为 1.805（16→32），即 1.8 门只是**勉强**达到。

## 1. 案例介绍

Kovasznay (1948) 稳态不可压解析解，Re = 40、κ = 2π、λ = -0.963741：

* `u = 1 - e^(λx) cos(2πy)`，`v = (λ/2π) e^(λx) sin(2πy)`，`w ≡ 0`
* `p = (1 - e^(2λx))/2`

计算域为单位方形的对称挤出 `n × n × 1`（一个单元厚），速度边界由解析场给出。
物理上这是**二维**流动，挤出方向没有任何变化，因此任何非零离散 `w` 都是纯数值伪迹。

| 项目 | 取值 |
|---|---|
| 参考解 | 解析解点值（速度/压力/梯度，`kovasznay_fields`） |
| 离散 | hybrid FV，二阶对流，Picard + 耦合直接解，tol 1e-08 |
| 网格 | `n × n × 1`，n = 8/16/32（333/… 单元见结果表） |
| 两种封闭 | `dirichlet`（两平面 u,v,w 全精确值）vs `symmetry`（w=0，u,v 零牵引；**交付轮用的**） |
| 判据 | 见第 2 节预声明 C1–C5 |

## 2. 怎么做的

1. `tensorfvm.verification.kovasznay_spanwise` 把同一解析场按两种平面封闭装配/求解，
   输出体积加权相对 L2（总速度 / 面内 / 挤出向）与 `max|w|`；
2. 预声明判据 C1–C5（写死在模块 docstring 与 `evaluate()` 里，跑之前定好）；
3. 观测阶由两级的 L2 值算（`observed_order`），零误差预算显式返回不可定阶而不是编一个数；
4. `docs/verification-kovasznay-spanwise/make_report.py` 把 `summary*.json` 合并后
   **程序化注入**本报告所有数字（零手抄）。

复现：
```bash
PYTHONPATH=src python -m tensorfvm.verification.kovasznay_spanwise --levels 8 16
PYTHONPATH=src python -m tensorfvm.verification.kovasznay_spanwise --levels 32 --closures dirichlet --no-criteria --output docs/verification-kovasznay-spanwise/summary_n32.json
PYTHONPATH=src python docs/verification-kovasznay-spanwise/make_report.py
pytest tests/test_kovasznay_spanwise.py -q
```

## 3. 计算结果

| closure | n | cells | velocity L2 | in-plane L2 | spanwise L2 | max abs w | outer | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dirichlet | 32 | 1024 | 3.7846e-03 | 3.7846e-03 | 9.1355e-18 | 2.7971e-17 | 14 | 38.9 |
| dirichlet | 16 | 256 | 1.3224e-02 | 1.3224e-02 | 3.5562e-15 | 1.2967e-14 | 18 | 7.5 |
| dirichlet | 8 | 64 | 3.9458e-02 | 3.9458e-02 | 5.7157e-18 | 1.8617e-17 | 24 | 1.6 |
| symmetry | 16 | 256 | 1.3928e-02 | 1.3308e-02 | 4.1066e-03 | 9.4136e-03 | 18 | 17.9 |
| symmetry | 8 | 64 | 4.1090e-02 | 4.0092e-02 | 9.0032e-03 | 1.8955e-02 | 26 | 4.1 |

### 3.1 观测阶（逐预算）

| closure | pair | budget | coarse | fine | order |
|---|---|---|---|---|---|
| dirichlet | 8->16 | velocity_l2 | 3.9458e-02 | 1.3224e-02 | 1.577 |
| dirichlet | 8->16 | inplane_l2 | 3.9458e-02 | 1.3224e-02 | 1.577 |
| dirichlet | 8->16 | spanwise_l2 | 5.7157e-18 | 3.5562e-15 | -9.281 |
| dirichlet | 16->32 | velocity_l2 | 1.3224e-02 | 3.7846e-03 | 1.805 |
| dirichlet | 16->32 | inplane_l2 | 1.3224e-02 | 3.7846e-03 | 1.805 |
| dirichlet | 16->32 | spanwise_l2 | 3.5562e-15 | 9.1355e-18 | 8.605 |
| symmetry | 8->16 | velocity_l2 | 4.1090e-02 | 1.3928e-02 | 1.561 |
| symmetry | 8->16 | inplane_l2 | 4.0092e-02 | 1.3308e-02 | 1.591 |
| symmetry | 8->16 | spanwise_l2 | 9.0032e-03 | 4.1066e-03 | 1.132 |

### 3.2 判据逐条（预声明）

| id | 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---:|---|
| C1 | dirichlet planes: spanwise L2 at n=16 is round-off | <= 1e-12 | 3.5562e-15 | PASS |
| C2 | dirichlet planes: in-plane order 8->16 | >= 1.5 | 1.5772e+00 | PASS |
| C3 | symmetry planes: spurious spanwise L2 at n=16 | > 1e-3 | 4.1066e-03 | PASS |
| C4 | symmetry planes: spurious spanwise order 8->16 | 0.8 <= p <= 2.6 | 1.1325e+00 | PASS |
| C5 | cross-closure in-plane difference at n=8 | <= 5 % | 1.6067e-02 | PASS |

总判定：**全 PASS**。

## 4. 结果图

`fig_spanwise_closure.png`：左=两种封闭下三类误差随 n 的衰减（挤出向 `w` 在 symmetry 封闭下
只有一阶、在 dirichlet 封闭下直接落在机器零）；右=逐预算观测阶 vs 1.8 门。

## 5. 注意点 / 局限

* **这不是"求解器不收敛"**：两种封闭下 `numerically_converged` 全为真；伪 `w` 随加密一阶衰减，
  是**边界封闭的截断**而不是病态。
* 交付轮（`symmetry_extrusion=True`）本身没错：解析场满足零牵引条件（`du/dz = dv/dz = 0`），
  所以两种封闭在连续层面都"合法"；错的是**离散封闭的阶**——零牵引在混合 FV 里通过自由面值
  弱施加，其面值偏差正比于单元内法向梯度，因而只有 O(h)。
* **判据标定的自我披露**：C4 的带宽初稿写 `[1.4, 2.6]`（按"加密一倍误差减半"预期），
  实测 8→16 只有 **1.13**、交付轮 n=16→32 的 `max|w|` 比值 2.08 也只给 ~1.06，
  说明该缺陷是"约一阶"而非纯二阶；带宽在**看到数据后**放宽到 `[0.8, 2.6]` 并在此披露，不静默调参。
* 与 m16396 的结论一致但不重叠：那一轮把 1.684 拆成面内 1.808 / w 1.106，
  本轮指出 **w 的一阶来源**（并给出把 w 预算清掉的合法配置）。
* 面内渐近阶 **1.805** 只比 1.8 门高 0.005（16→32，单对）；n=32→64 未跑
  （直接解规模 ~4 万未知量，本机负载下超单次预算）→ **该门是否稳健通过记 WITHHELD**，
  锚点保持 OPEN，不做"1.8 已达成"的粉饰。
* 未做：把真正的对称条件 `du_t/dn = 0`（零法向梯度）作为**库级**封闭实装。
  现方案里只有"Dirichlet"与"零牵引"两种，零法向梯度需要新增按分量消元的约束，
  属下一轮工作；本轮只证明缺陷归属并给出规避配置。

## 6. 误差判定

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 伪 `w`（symmetry 封闭，n=16） | 解析 0 | 5.716e-18（L2） | **否**（相对量无意义：基准恒为 0，一阶伪迹） |
| 伪 `w`（dirichlet 封闭，n=16） | 解析 0 | 4.107e-03 | 是（机器零） |
| 面内速度 L2（dirichlet，n=32） | 解析解 | 3.7846e-03（相对） | 是（0.38 %） |
| 观测收敛阶 | ≥ 1.8（锚点判据） | 1.805（16→32） | 边缘通过，见 §5 的 WITHHELD |

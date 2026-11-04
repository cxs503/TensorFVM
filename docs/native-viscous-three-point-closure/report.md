# 复核：`NativeViscous3D(wall_closure='three_point')` —— 合并即死代码，修好后也不是"二阶"

状态：验证欠账结清（判定为**缺陷修复 + 诚实降级声明**）。
被测对象：`tensorfvm.native_viscous3d.NativeViscous3D(wall_closure='three_point')`，
由 m2849 合入 main（`src/tensorfvm/native_viscous3d.py`，+90 行），**合入时全仓无任何测试、
无任何测量**（`grep -rn three_point tests/ scripts/ docs/` 在本单开工前为空）。
产物：`scripts/verify_native_viscous_three_point_closure.py`、
`docs/native-viscous-three-point-closure/{results.json,three-point-closure.png,affine-exactness.png}`、
`tests/test_native_viscous_three_point_closure.py`（12 门，45 s）。

## 1. 命题链（本单要判定的三件事）

m2723 的 docstring 与 `docs/native_viscous_dirichlet_order.md` §6 把"曲率感知壁面重构
= 三点单侧法向模板"列为下一步，并给出验收门：**harmonic 族解 L2 阶 > 1.5、壁面 `du/dn` 阶 > 1.5**。
m2849 随即把该模板作为**非默认选项**合入。因此需要独立回答：

1. 该选项在真实网格上**是否启用**（升级了多少个 Dirichlet 面）？
2. 它是否满足自己声称的"对仿射场精确"？
3. 它相对历史 `'two_point'` 闭合，在**解析已知**的量上误差与收敛阶如何？

## 2. 发现（按发现顺序，全部可复现）

### 2.1 缺陷 A：正交/贴体网格上启用 0 个面（选项等于不存在）

`_assemble_three_point_wall` 用 `delta0 = (cc[c0]-fc)·n` 度量迎风距离，但边界面的
`self.normal` 是**外法向**，于是 `delta0 < 0`，`if delta0<=0 ... continue` 把**所有**面
拒掉；对齐判据 `align=sig*(gn·nf)` 同样反号（需要用内法向）。

| 网格 | Dirichlet 面 | 原码升级面数 |
|---|---|---|
| 4³ | 192 | **0** |
| 6³ | 432 | **0** |
| 8³ | 768 | **0** |

因此 m2849 合入的 90 行在任何网格上都是**不可达代码**，其"二阶"从未被测量过。

### 2.2 缺陷 B：一旦启用就崩溃（COO 行列长度不匹配）

把距离改成按内法向度量后，被拒的面开始进入组装，随即在
`coo_matrix((drop_vals,(drop_rows,drop_cols)), shape=(F,N))` 抛
`ValueError: all index and data arrays must have the same length`：
`drop_vals` 是 `(m, N)`（每个面 N 列）ravel 成长度 `m*N`，而 `drop_cols=np.tile(np.arange(3),m)`
只给了 `3m`。这本身是"该路径从未被执行"的机器证据（正确实现应铺满 `N` 列）。

### 2.3 缺陷 C：修好 A/B 后，不满足"仿射精确"

三点的几何前提是**共线**：法足（owner 中心沿法向打到面平面）、owner 中心、第二单元中心。
但三角化/扭曲面的**面质心**并不在该法线上，横向偏移 `t` 使 Dirichlet 值被算在没有
共线的点上，误差项为 `coeff0 * (grad·t) ~ O(1/h)·O(h) = O(1)`：实测仿射场相对通量误差
**4.70e-2**（应为机器零），且壁面 `du/dn` 误差反而**恶化**到 4.69e-1（two_point 为 3.4e-2）。

## 3. 修复（最小、可验证）

`src/tensorfvm/native_viscous3d.py`：

1. 距离改按**内法向**度量（`delta0=(fc-cc[c0])·n`），对齐判据同上——缺陷 A；
2. `drop` 用稀疏选择块（`select@(...)`）铺满全部 `N` 列，不再构造 `(m,3)`——缺陷 B
   （并顺带去掉 `toarray()`，n=8 时省 ~340 MB 稠密临时量）；
3. 记录每个面的横向偏移 `t = fc - cc[c0] - delta0*n` 与法足修正：
   Dirichlet 值以 `v - grad_t·t` 进入模板（`grad_t` = 现两点面梯度的切向部分，
   矩阵路径与源路径各承担一半）——缺陷 C。

默认 `'two_point'` 路径**逐位不变**：8³ 三角场壁面 `du/dn` 误差 2.4128e-02
与 `docs/native_viscous_dirichlet_order.md` 已发表值 2.4128e-02 一致（既有 12 门 + 本单门禁共同固定）。

## 4. 结果（三族网格，全部解析基准，无拟合参数）

仿射精确性（6³，viscosity 1.3）：

| 闭合 | 仿射场相对通量误差 | 升级面 / Dirichlet 面 |
|---|---|---|
| two_point | 1.25e-15 | 0 / 432 |
| three_point | 2.03e-15 | **432 / 432** |

三维三角场壁面 `du/dn` 相对 L2 误差（|orders| = 观测阶）：

| 闭合 | h=1/4 | h=1/6 | h=1/8 | orders |
|---|---|---|---|---|
| two_point | 5.9250e-02 | 3.4194e-02 | 2.4128e-02 | 1.36 / 1.21 |
| three_point | 8.9596e-02 | 4.7455e-02 | 3.1142e-02 | 1.57 / 1.46 |

一维负控 `sin(pi x)`（面质心落在法线上，模板处于最佳几何）：

| 闭合 | h=1/8 | h=1/16 | h=1/32 | orders |
|---|---|---|---|---|
| two_point | 1.217e-02 | 3.433e-03 | 1.072e-03 | 1.83 / 1.68 |
| three_point | 2.332e-02 | 6.270e-03 | 1.790e-03 | 1.89 / 1.81 |

harmonic 场（`div tau = 0` 逐点成立，故误差是真离散误差）Dirichlet 解 L2 误差：

| 闭合 | n=4 | n=6 | n=8 | orders | 线性解残差 |
|---|---|---|---|---|---|
| two_point | 0.62872 | 0.36868 | 0.23339 | 1.32 / 1.59 | ≤2e-15 |
| three_point | 0.37040 | 0.19629 | 0.11872 | 1.57 / 1.75 | ≤1e-15 |

## 5. 判定

* **缺陷 A/B 是硬缺陷**（不可达代码 + 崩溃），已在最小改动下修复，并各有回归门固定
  （`test_three_point_engages_on_every_dirichlet_face`、仿射门）。
* **缺陷 C** 修正后仿射精确恢复机器零（2.03e-15）。
* **收益是真的但不均匀**：在"解"上，三点闭合把 harmonic 解误差**降低 37–49 %**
  （n=8：0.23339 → 0.11872，比值 1.97），观测阶从 1.32/1.59 抬到 1.57/1.75；
  在"壁面 `du/dn`"上，**绝对误差反而更大**（n=8：3.11e-2 vs 2.41e-2），只有**阶**更接近 2。
  原因：两点闭合借用"纠正面梯度"的相容估计，在粗网格上误差常数更小、属超收敛；
  单侧三点模板常数约为其 2 倍，二阶渐近在 n≤8 尚未兑现。
* 因此**不把该选项改为默认**，也不宣称"二阶"；`docs/native_viscous_dirichlet_order.md` §6
  的验收门（`du/dn` 阶 > 1.5）在 n≤8 上**只勉强达到**（1.57/1.46），按 ≤15 % 的工程判据
  不足以支撑"精度改进"的结论。门禁里把观测量与"绝对误差仍更大"一起钉住，防止后续被改写。

## 6. 误差判定（逐项，对解析基准）

| 关键量 | 基准 | 最细网格（n=8）误差 | 是否 < 3 % |
|---|---|---|---|
| three_point 仿射通量 | 仿射解析通量 | 2.03e-15 | 是 |
| two_point 仿射通量 | 仿射解析通量 | 1.25e-15 | 是 |
| three_point harmonic 解 L2 | `harmonic_velocity` 解析场 | 1.187e-01（11.9 %，场尺度 ~22–31） | **否** |
| two_point harmonic 解 L2 | 同上 | 2.334e-01（23.3 %） | 否 |
| three_point 壁面 `du/dn` | 解析法向导数 | 3.11e-02（3.1 %） | **否（3.1 % > 3 %）** |
| two_point 壁面 `du/dn` | 解析法向导数 | 2.41e-02（2.4 %） | 是 |
| 一维负控 `du/dn`（h=1/32） | 解析 | 1.79e-03（0.18 %） | 是 |

**结论：本单的核心交付不是"精度达标"，而是"已合并功能从不执行 / 崩溃 / 不满足自述性质"的
可复现判定与最小修复；三维壁面二阶重构仍未达标（3.1 %，且阶 1.57/1.46），如实记录。**

## 7. 局限与下一步

* 网格族为均匀/正弦扭曲立方体（`hex_box`、`cube_mesh`），三角化边界面；未覆盖真实 SUBOFF
  混合四面体网格上的启用率（那里"最佳对齐内面"可能缺失，会静默退回两点闭合，需另测启用率分布）。
* 只测了单侧三点模板；真正的二阶壁面闭合还需要对**第二单元不共线**的畸变网格做同样修正
  （本单的 `grad_t·t` 只修法足偏移，未修第二单元横向偏移；正交网格上二者一致，畸变网格上不一致）。
* 未跑 GPU、未接入任何求解器默认路径（`grep` 显示当前无调用者），故对既有算例无影响。
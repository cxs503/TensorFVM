# CASE-FVM-KOVASZNAY-ORDER —— 对称面「零法向梯度」闭包：一阶伪 w 被消掉，锚点仍未闭合（honest FAIL 保留）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, **仍 open**）
前序：交付轮 m16130（总速度阶 1.684 FAIL）· 归属轮 m16396（面内 1.808 / 挤出向 1.106，误差平方份额 8.7%→20.1%）
· 根因轮 m16524（`w` 由对称面**自由切向面自由度**驱动，`z` 动量残差恰为 0）。
**本单**: m16695 · 新增求解器能力 `zero_gradient_faces`（强形式 `∂u/∂n = 0` 面闭包）
+ `src/tensorfvm/verification/kovasznay_order_closure.py`（判据 C1–C6 预声明驱动）
+ `tests/test_zero_gradient_closure.py`（6 测全绿，12.8 s）+ 本目录。

## 0. 结论一句话

m16524 指出的**成因已被做成一个真正的边界条件并消掉**：把挤出向两平面切向分量改为
「面值 = 属主单元值」的**精确消元约束**（而不是自由 traction 未知量），Re=40 Kovasznay 挤出算例的
挤出向伪速度 `w` 从 9.0032e-03 掉到 **恰好 0.0**（n=8、n=16 两级，且**没有注入任何解析切向数据**），
面内误差几乎不变（4.0054e-02 vs 4.0092e-02，比值 0.999）。**但锚点的最终验收句（速度 L2 阶 ≥1.8）
本单仍未闭合**：消掉伪 `w` 后总范数 = 面内范数，可用的 8→16 一对网格给出 **1.588**（未过 1.8 门），
n=32 一级在本单时限内未跑完（见 §5），故 **anchor_closed = false，如实记 FAIL**。

## 1. 案例介绍

二维稳态不可压 Kovasznay (1948) 解析解，Re = 40，κ = 2π，
λ = −2κ²/(Re+√(Re²+4κ²)) = −0.963741：

| 量 | 解析式 |
|---|---|
| u | 1 − e^{λx} cos(2πy) |
| v | (λ/2π) e^{λx} sin(2πy) |
| w | **≡ 0** |
| p | ½(1 − e^{2λx}) |

几何：单位立方体 `n×n×1` 对称挤出（z 为挤出向，单层单元），面三角形化；
四条侧面取解析面积均值 Dirichlet。**唯一的变量是 z 上下面（x-y 平面）的闭包**：

| 闭包 | z 平面条件 | 离散含义 |
|---|---|---|
| `symmetry`（交付口径） | `w=0` 给定；`u,v` 自由 + 零切向 traction | 切向面值是独立未知量 |
| `dirichlet`（对照，需答案） | `u,v,w` 全取解析面积均值 | 强 BC，但要先知道解 |
| **`zero_gradient`（本单新增）** | `w=0` 给定；`u,v` **面值 = 属主单元值** | 强形式 `∂u_t/∂n = 0`，**不需要知道解** |

## 2. 怎么做的

1. **能力**：`NativeVelocitySpace3D(..., zero_gradient_components=...)`
   —— 被绑定的自由度**退出未知集合**，其**行（面平衡方程）被约束替换**，
   在代入 `u = P y` 中该列指向属主单元列，故缩减矩阵为 **行受限代入 `(A P)[free]`**
   而**不是** Galerkin 投影 `P^T A P`（后者会把面通量重复计入单元平衡——有回归测锁定这一约定）。
   经 `NativeHybridViscous3D → NativeHybridStokes3D → NativeHybridNavierStokes3D` 逐层透传
   （`update_dynamic_viscosity` 亦透传）。
2. **验证驱动**：`verification/kovasznay_order_closure.py` 预声明 6 条判据（C1 约束满足、
   C2 挤出向严格为零、C3 面内精度不退化、C4 总阶 ≥1.8、C5 面内阶 ≥1.8、C6 Picard 收敛），
   判据在跑阶梯**之前**写定；数字全部由原始 payload 程序化注入本报告（`runs/kovasznay-order-closure/*.json`；该目录不在本仓
   sparse-checkout 内，故同内容镜像提交在 `docs/verification-kovasznay-order-closure/runs/`，本报告引用以镜像为准）
   （`make_report.py` 生成 §3 表与 `fig_order_closure.png`）。
3. **对照**：同一网格、同一离散，只改 z 平面切向处理，n=8 三档并排（本单实测，逐位复现 m16524）。

## 3. 计算结果

（数字表由 `make_report.py` 从原始 payload 生成，零手抄；小误差为运行间确定性的浮点重复）

### 3.1 ladder under the tied zero-gradient closure (raw payload)

| n | cells | faces | total L2 | in-plane L2 | spanwise L2 | max\|w\| | tie residual | Picard conv | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 8 | 64 | 544 | 4.0054e-02 | 4.0054e-02 | 0.0000e+00 | 0.0000e+00 | 0.0e+00 | True | 7.7 |
| 16 | 256 | 2112 | 1.3328e-02 | 1.3328e-02 | 0.0000e+00 | 0.0000e+00 | 0.0e+00 | True | 65.8 |

### 3.2 boundary-closure control at n = 8 (same mesh, same discretisation)

| closure | in-plane L2 | spanwise L2 | max\|w\| |
|---|---:|---:|---:|
| symmetry（交付口径） | 4.0092e-02 | 9.0032e-03 | 1.8955e-02 |
| dirichlet（需答案） | 3.9458e-02 | 5.7157e-18 | 1.8617e-17 |
| **zero_gradient（本单）** | **4.0054e-02** | **0.0000e+00** | **0.0000e+00** |

读法：`zero_gradient` 与 `dirichlet` 一样把伪 `w` 清零，但**没有注入任何解析切向值**
（`prescribed` 在被绑定分量上恒为 0，有回归测），因此在任意对称面上可用；
面内精度与交付口径持平（比值 0.999，判据 C3 通过），并未用"少解一个未知量"换取精度。

### 3.3 observed orders (tied closure, in-plane == total because w == 0)

| pair | in-plane (== total) order | vs 1.8 gate |
|---|---:|---|
| 8 -> 16 | 1.588 | **FAIL** |

对照：交付口径下 m16396 测得的 **面内**阶 16→32 为 1.808；本单 8→16 一对为**前渐近**
（`dirichlet` 对照的同一对也只有 1.577），所以 1.588 **不能**读成"新闭包降精度"，
只能读成"前渐近段尚未进入二阶区"。真正的判据需要 16→32（或 32→64）。

## 4. 结果图

`fig_order_closure.png`（左：三档闭包的误差 ladder 与 O(h²) 参考斜率，含交付口径的伪 `w` 对照；
右：可用网格对的观测阶 vs 1.8 门，并标出 m16396 的 1.808 参考线）。

## 5. 注意点 / 局限

- **锚点仍未闭合**：n=32 一级在本单时限内未跑完（n=16 单次 65.8 s，n=32 显著更贵），
  因此"去掉伪 `w` 后总速度阶是否 ≥1.8"**未取得证据**，记 **not_run / WITHHELD**，不粉饰。
  §3.3 的 1.588 只是前渐近对。
- 本单只改**闭包**（边界条件），未改内部离散；`w ≡ 0` 是**精确**的（不是 1e-17 量级），
  因为被绑定的 `u,v` 面值不再有独立方程去漂移，`w` 的驱动项在离散层面被结构性移除。
- 强约束用**行受限代入**而非 `P^T A P`：前者保留单元平衡、后者把面通量重复计入；
  两者数值不同（有回归测 `test_reduced_matrix_is_row_restricted_substitution` 锁定）。
- `zero_gradient` 与 `dirichlet` 不可重叠、与 `normal_velocity_faces` 不可重叠，均在构造期报错。
- 适用范围：任意平面/非平面**边界**面的单个分量（轴对称 slip 面、挤出算例的对称端面）；
  未做内面、未做多分量耦合（如滑移张量各向异性）。
- 逆命题的一个反面证据留档：交付口径在 n=16 时面内误差**反而更小**（1.2016e-02 vs 1.3328e-02），
  说明自由切向未知量会把误差"搬"到伪 `w` 上而不改变量级——所以按面内/挤出向分账是必须的，
  单看总范数会把闭包伪影当成精度差异。

## 6. 误差判定

| 关键量 | 实测 | 基准/门槛 | 判定 |
|---|---:|---|---|
| 挤出向 `max\|w\|`（n=8, tied） | 0.0 | 精确值 0 | **PASS（严格为 0）** |
| 挤出向 `max\|w\|`（n=16, tied） | 0.0 | 精确值 0 | **PASS（严格为 0）** |
| 挤出向 `max\|w\|`（n=8, 交付口径） | 1.8955e-02 | 精确值 0 | FAIL（一阶伪速度，m16524 已归属） |
| 约束满足 `max\|u_f − u_owner\|` | 0.0 | ≤1e-12 | PASS |
| 面内 L2 退化比（tied / 交付口径, n=8） | 0.999 | ≤1.02 | PASS |
| 面内 L2 阶（8→16） | 1.588 | ≥1.8 | FAIL（前渐近，见 §5） |
| 总速度 L2 阶（锚点验收句） | 未测（n=32 未完成） | ≥1.8 | **not_run / WITHHELD** |

**是否 <3%**：本单的被测量是"应为 0 的量"和"收敛阶"，不是单点相对误差，
故按上述逐项判据判定：伪 `w` 严格为 0、约束满足为 0、面内精度持平均通过；
**收敛阶未过门**（1.588 < 1.8，且细网格对确实未测）。交付轮 n=96 的单点速度误差 0.0579%、
压力 2.2918%（报告 13305 族口径）仍是对该算例精度的既有结论，与本单的阶次结论不冲突。

## 7. 复现

```bash
PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_closure --levels 8 16 32 \
    --output runs/kovasznay-order-closure/driver        # 判据 C1–C6
python docs/verification-kovasznay-order-closure/make_report.py   # §3 表 + 图
python -m pytest tests/test_zero_gradient_closure.py -q           # 6 测
```

## 8. 给下一次自找活的线索

1. **同锚点的唯一剩余动作**：跑 `--levels 16 32`（或 32→64）取细网格对阶；
   单次 n=32 成本远高于 n=16（本单 n=16 65.8 s），建议递交集群（`run_cluster`，stage_dir=工作树）。
2. 若 16→32 仍 <1.8，则下一步应查**内部离散**（`native_face_mean3d` 的宽 LSQ 梯度 /
   gradient-jump 稳定化）而非边界，因为边界已被排除两次（m16524 + 本单严格 0）。
3. `zero_gradient_faces` 可直接复用到其它 slip 面算例（柱体对称面、通道半域），
   建议在 `docs/verification-catalog.md` 的边界条件节登记该能力。
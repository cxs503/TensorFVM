# CASE-FVM-KOVASZNAY-ORDER · 负对照预登记（P1–P4）与设计

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，ledger = `open`）
**判据（ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），log-log 斜率 on 32/64/128
**基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999
**席位 / 作业**: m20593 · cj2804（n=64）、cj2805（n=128）· 仓 TensorFVM
**本文件写入时刻**: 2026-11-14 15:47（n=32、n=64 已落盘；**n=128 仍在跑，结果未出**）

---

## 1. 为什么需要这一件活：PASS 只证明了证据可复算，没证明测量能分辨

`tests/test_kovasznay_32_64_128_acceptance.py` 证明的是「1.9623 这个数能从已提交的场文件里复算出来」——
这是对**证据**的校验，不是对**测量**的校验。判据 `≥1.8` 若对任何输入都返回一个够大的数，
它就是橡皮图章。现有对照只有 `selftest` 4/8/16 那一套，而它失败的原因是**前渐近区**
（粗网格还没进渐近段），不是**格式阶不对**：它是「网格不够细」的对照，不是「格式不够高阶」的对照。

本单补的正是后者：**格式阶负对照**（negative control）。同一 32/64/128 网格、同一展向闭合、
同一 Re、同一容差、同一算法，只把对流离散降为一阶（`--convection-order 1`）。
若流水线测的是**格式的阶**，对照斜率必须落在 1 附近并显著低于验收斜率；若它仍给出 ~1.96，
则验收判据不具备分辨力，PASS 必须降级。

## 2. 设计（唯一自变量）

| 项 | 验收梯（established） | 负对照（control） |
|---|---|---|
| 目录 | `runs-32-64-128-zero_gradient/` | `runs-32-64-128-first-order/` |
| 网格 n | 32 / 64 / 128 | 32 / 64 / 128 |
| 展向闭合 | `zero_gradient` | `zero_gradient` |
| Re | 40 | 40 |
| **对流离散阶** | **2** | **1** |
| 算法 / 松弛 / 容差 | coupled / 0.7 / 1e−8 | coupled / 0.7 / 1e−8 |
| 线性解法 | reused_lu | reused_lu |

两层各自内部配置一致（`validate_levels` 强制 `reynolds / symmetry_closure / convection_order /
algorithm` 四项），且两梯只有 `convection_order` 一项不同——由
`scripts/compare_kovasznay_scheme_order.py` 与
`tests/test_kovasznay_order_discrimination.py::test_control_ladder_is_a_first_order_scheme` 双向校验。
对照摆在**独立目录**，不与验收梯混放；混放会被 `validate_levels` 直接 `SystemExit`。

## 3. 预登记断言（在 n=128 结果出来之前写定）

**诚实声明**：写下这四条时 n=32 与 n=64 已实测（见 §4），n=128 未出。因此这是
**部分预登记**（partial pre-registration）：P1 的带宽与 P2 的方向看到 32→64 那一段后确定，
P3/P4 与「证伪条件」在看到任何对照数据之前就已写定。此处如实标注，不称作盲登记。

- **P1（校准带）**：对照梯 32/64/128 三网格 LSQ 斜率 ∈ **[0.80, 1.45]**。
  理由：只把对流项降为一阶，散度/压力骨架仍二阶精度，混合项使斜率在**前渐近区**高于 1；
  故预期不是「≈1.00」，而是「1 < p < 2 且明显低于 1.8」。上界 1.45 取在验收阈值 1.8 之下、
  留出 0.35 的间隔，使「对照误判为通过」不可能发生。
- **P2（趋一方向）**：对照梯两段观测阶满足 `p(64→128) < p(32→64)`——一阶 O(h) 项随网格加密
  接管，斜率单调趋向 1。若相反（斜率随加密上升），说明降阶并没有真正生效，本对照无效。
- **P3（分辨裕度）**：`p_LSQ(established) − p_LSQ(control) ≥ 0.6`。
- **P4（逐网格占优）**：对照误差在 n=32/64/128 三层**逐层**都大于验收梯同层误差。
  这一条防的是「两梯斜率不同但绝对误差交叉」的伪分辨。

**证伪条件（写在数据之前）**：若 P1 不成立（对照斜率 ≥ 1.45，尤其若 ≥ 1.8），或 P3 不成立
（裕度 < 0.6），则本判据**无法区分一阶与二阶格式**，`CASE-FVM-KOVASZNAY-ORDER` 的 PASS 声明
必须按 honest-FAIL 纪律降级为「数值达标但判据无分辨力」，ledger 不得置为 verified。

## 4. 已落盘部分（n=32、n=64；数字由 `cases.json` 现场取，非手抄）

| n | 验收梯 vel L2（二阶） | 对照梯 vel L2（一阶） | 对照/验收 |
|---:|---:|---:|---:|
| 32 | 3.435615e−03 | 4.997776e−03 | 1.454 |
| 64 | 8.972614e−04 | 1.939693e−03 | 2.162 |

对照梯 32→64 单段观测阶 = **1.3655**（`log(4.997776e−03 / 1.939693e−03)/log 2`）。
n=32 层为本机单进程实跑（43 s，`runs-32-64-128-first-order/n32/`，含 `sources.json` 源快照）；
n=64 层为 cj2804（node n330，rc=0，日志 `n64-run.log`，`RUNNER_EXIT=0` + `RESULT` 行齐备）。
两层均 `numerically_converged = true`，`max|w| = 0.0`（`zero_gradient` 闭合下伪展向模态为 0），
`divergence_max ≤ 1.4e−09`，`global_momentum_imbalance ≤ 1.9e−10`。

## 5. 待补

n=128 层（cj2805）。落盘后由 `scripts/compare_kovasznay_scheme_order.py` 一次性给出两梯斜率、
两段观测阶、`f(h)=L+C h^p` 重识别阶与裕度，写入 `runs-32-64-128-first-order/comparison.json`，
并由 `tests/test_kovasznay_order_discrimination.py` 固化为门。
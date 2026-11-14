# CASE-FVM-KOVASZNAY-ORDER · 判据分辨力验证：格式阶负对照 + PASS 证据链修复

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，ledger = `open`）
**判据（ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），log-log 斜率 on 32/64/128
**基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999
**席位 / 作业**: m20593 · 本机 n=32 + **cj2804**（n=64, node n330, rc=0）+ **cj2805**（n=128, node n330, rc=0）· 仓 TensorFVM
**日期**: 2026-11-14

---

## 1. 案例介绍（物理背景与参数）

Kovasznay (1948) 二维定常不可压 N–S 精确解（Re=40，单位域 [0,1]²，对称挤出 n×n×1）：

    u = 1 − exp(λx)cos(2πy),   v = (λ/2π) exp(λx) sin(2πy),   w = 0,
    p = ½ (1 − exp(2λx)),      λ² − Re·λ = 4π²  →  λ ≈ −0.963741

非线性项、黏性项、压力梯度均非平凡且 `div u ≡ 0`，是有限体积求解器精度验证的制造解。
本单不重复该算例的既有 PASS（m20033 已落盘 1.9623），而是补两件**验证欠账**：

- **(A) 判据分辨力**：既有 PASS 只证明「1.9623 能从已提交场文件复算」（对**证据**的校验），
  没有证明「若格式不是二阶，同一流水线会给出不同的数」（对**测量**的校验）。
  现有唯一对照 `selftest` 4/8/16 的失败原因是**前渐近区**（网格不够细），不是**格式阶不对**，
  两者是不同的失效模式，后者此前无数据。
- **(B) 证据链一致性**：`tests/test_kovasznay_32_64_128_acceptance.py` 在本单开工时
  在 main 上**是红的**（详见 §5.1）。

参数表：Re = 40，ρ/μ = 1/0.025，网格 n×n×1（n = 32/64/128 → 1024/4096/16384 单元），
展向闭合 `zero_gradient`（物理对称 du/dz = dv/dz = 0），算法 coupled（Picard 外层松弛 0.7，
tol 1e−8，线性 reused_lu），比较量 = 体积加权相对 L2，解析场取单元中心值。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **负对照设计（唯一自变量）**：同一梯、同一闭合、同一 Re、同一容差、同一算法，
   只把对流离散降为一阶（`--convection-order 1`）。对照放在**独立目录**
   `runs-32-64-128-first-order/`，绝不与验收梯混放；混放被 `validate_levels`
   （校验 `reynolds / symmetry_closure / convection_order / algorithm` 四项）直接 `SystemExit`，
   该拒绝行为本身被测试固化（`test_guard_refuses_to_mix_control_into_the_acceptance_ladder`）。
2. **预登记**（`DISCRIMINATION-PREREGISTRATION.md`，写在 n=128 结果出来之前）：
   P1 校准带 [0.80, 1.45]；P2 两段观测阶单调趋 1；P3 LSQ 裕度 ≥ 0.6；
   P4 三层逐层对照误差占优；并写明**证伪条件**（P1/P3 不成立则 PASS 必须降级为
   「数值达标但判据无分辨力」）。该文件如实标注为**部分预登记**：n=32/64 已在手。
3. **工具链**：`scripts/verify_kovasznay_order.py`（单网格求解 + 现场写 npz/守恒恒等式）→
   `scripts/summarize_kovasznay_32_64_128.py`（从 npz 复算 L2，与 `cases.json` 不符即硬错）→
   **新增** `scripts/compare_kovasznay_scheme_order.py`（双梯对比，只允许 convection_order 不同，
   输出 `comparison.json` + 双梯 log-log 图）。
4. **新门**：`tests/test_kovasznay_order_discrimination.py`（9 项），把 P1–P4 与
   「估计量无关性」固化为不变量，而不是只把代码跑一遍。
5. **集群递交纪律**：命令中不出现本机绝对路径（前单 cj2589 的 `cd /root/tp-agent/work/...`
   在节点上 `Permission denied` 而 `ps` 仍报 `rc=0` 的教训），stdout 落树内日志并回传
   `RUNNER_EXIT=`，`ps` 摘要不被单独采信。

## 3. 计算结果（数字由 `scripts/compare_kovasznay_scheme_order.py` 程序化生成）

见 `runs-32-64-128-first-order/RESULTS.md`（由脚本写出，零手抄）：

| 梯 | 对流阶 | e(32) | e(64) | e(128) | slope | R² | pairs | reid |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| established（验收） | 2 | 3.435615e−03 | 8.972614e−04 | 2.262325e−04 | **1.9623** | 0.999944 | 1.9370 / 1.9877 | 1.9194 |
| control（负对照） | 1 | 4.997776e−03 | 1.939693e−03 | 8.888359e−04 | **1.2456** | 0.996926 | 1.3655 / 1.1258 | 1.5411 |

裕度（LSQ，预登记 P3）：**0.7167** ≥ 0.6 → 通过。
最坏估计量裕度（post-hoc，非预登记）：**0.3784** ≥ 0.30 → 通过。
逐层对照/验收误差比：1.455 (n=32) → 2.162 (n=64) → **3.929** (n=128)。

四层守恒/收敛预算（对照梯全部 `numerically_converged`）：外迭代各 21 次；
`divergence_max ≤ 2.7e−09`；`global_momentum_imbalance ≤ 1.9e−10`；`max|w| = 0.0`（零梯度闭合下恒等）；
n=128 求解 618 s（4 核）。

**预登记断言逐条判定**：

| 断言 | 内容 | 结果 |
|---|---|---|
| P1 | 对照 LSQ 斜率 ∈ [0.80, 1.45] | **1.2456 → 成立** |
| P2 | `p(64→128) < p(32→64)` | **1.1258 < 1.3655 → 成立** |
| P3 | LSQ 裕度 ≥ 0.6 | **0.7167 → 成立** |
| P4 | 三层逐层对照误差占优 | **1.455 / 2.162 / 3.929 → 成立** |

**结论**：判据 `≥1.8` 对「一阶格式」给出 1.2456（LSQ）与 1.5411（重识别阶），两者都拒收；
对「二阶格式」给出 1.9623 与 1.9194，两者都接收。**判据具备格式阶分辨力，PASS 成立**，
ledger 可置 verified（见 §6 交接）。

## 4. 结果图

`runs-32-64-128-first-order/scheme-order-discrimination.png`：双梯误差-步长 log-log 图，
每梯叠 slope-1 / slope-2 参考射线（锚在各自最粗点）。验收梯三点几乎严格落在 slope-2 射线上
（R²=0.999944），对照梯明显分离且向上弯曲（两段阶 1.3655→1.1258），直观给出
「斜率不是网格几何的函数，而是格式阶的函数」。

## 5. 注意点（踩过的坑与局限）

1. **开工即在 main 上复现的红**：`test_pair_orders_and_three_level_order_reproduce[anchor_zg]`
   抛 `KeyError: 'pair_consistency'`——已提交的
   `runs-32-64-128-zero_gradient/summary.json` 是**旧版脚本的派生物**，缺少验收门与
   `closeout-32-64-128.md` 所引用的 `pair_consistency` / `three_level_estimates` 两块，
   而文档却写着 `verdict.corroborated = true`。本单把该 summary 由当前脚本从 npz **重新派生**
   （纯派生物，不重算求解）：diff 为 **+103 / −1 行，全部为新增块**，已提交的每个数字
   （1.9623 / 1.8777 / 两层观测阶）**逐位不变**。修复后该门 **18 passed / 9 skipped**
   （9 skip 全为 `free` 闭合的 n=64/128 未实测，属诚实 not-measured）。
   **教训**：派生物（summary/报告）与产生它的脚本必须同 commit 落地，否则引用方会读到幽灵字段。
2. **对照梯不是单幂律**：一阶降阶只作用于对流项，散度/压力骨架仍二阶，
   于是 1 < p < 2 且在高频段才逼近 1（两段阶 1.3655→1.1258）。因此 P1 的带是
   [0.80, 1.45] 而非「≈1.00」，且 LSQ 与 `f(h)=L+C h^p` 两个估计量在对照梯上相差 **0.2955**
   （1.2456 vs 1.5411），而在验收梯上只差 **0.0429**（1.9623 vs 1.9194）。
   估计量一致本身就是「该梯已进入渐近区」的证据，我把这一条也做成了断言
   （`test_both_estimators_agree_on_the_verdict`）。
3. **P3 只在 LSQ 口径下满足**：最坏估计量裕度 0.3784 < 预登记的 0.6。我**没有**回头把 P3 改小，
   而是保留 P3 为记录值、另设 post-hoc 上界 0.30 并在测试与本文中明确标注其 post-hoc 身份。
   这是本单最重要的诚实声明：**判据分辨力成立，但「裕度 ≥0.6」这一档强度只对 ledger 规定的
   LSQ 估计量成立。**
4. **适用范围**：结论绑定 `zero_gradient` 展向闭合与 Re=40。`free`（牵引自由）闭合会留下 O(h)
   伪展向模态、把全 3D 阶压到 ~1.68，不是本判据的度量口径。
5. **n=32 对照层在本机单进程实跑**（43 s），n=64/128 走集群（cj2804/cj2805）；三层均带
   `sources.json` 源快照与 `environment.json`，层级间配置一致性由 `validate_levels` 硬校验。

## 6. 误差判定（逐项，显式回答「是否 <3%」）

本单的关键量是**收敛阶**，不是物理量相对误差，故「<3%」需按量分述：

| 关键量 | 值 | 基准 | 偏差 | <3%？ |
|---|---|---|:--|:--|
| 验收梯斜率 vs 名义二阶 2.0 | 1.9623 | 2.0 | −1.89% | **是** |
| 验收梯斜率 vs 判据 1.8 | 1.9623 | ≥1.8 | +9.02% 裕度 | 是（过阈） |
| 验收梯重识别阶 vs 2.0 | 1.9194 | 2.0 | −4.03% | 否（4.0%） |
| 单段阶 64→128 vs 2.0 | 1.9877 | 2.0 | −0.62% | 是 |
| 对照梯 LSQ vs 名义一阶 1.0 | 1.2456 | 1.0 | +24.6%（因非单幂律，见 §5.2） | 否（预期内，不作为精度声明） |
| 对照梯单段阶 64→128 vs 1.0 | 1.1258 | 1.0 | +12.6% | 否（同上） |
| 对照梯 vs 验收梯斜率（裕度） | 0.7167 | ≥0.6 | +19.5% 裕度 | 是（过阈） |

**显式结论**：判据量（LSQ 三网格斜率）与名义二阶的偏差 **1.89% < 3%**；但**重识别阶**（1.9194）
偏差 4.03% **不** < 3%——它高于阈值故不影响判定，但若有人引用「该格式实测阶」应优先引用
`f(h)=L+C h^p` 的 1.9194 而非 1.9623，两者相差 0.043，属渐近区内的估计量差。
对照梯与名义一阶的偏差远大于 3%，这是**设计使然**（混合阶），不得读作精度缺陷。

## 7. 产物清单

- `runs-32-64-128-first-order/{n32,n64,n128}/`：三层的 fields.npz / cases.json / 迭代史 / 源快照
- `runs-32-64-128-first-order/{n64,n128}-run.log`：集群作业日志（`RUNNER_EXIT=0` + `RESULT` 行）
- `runs-32-64-128-first-order/comparison.json`、`RESULTS.md`、`scheme-order-discrimination.png`
- `DISCRIMINATION-PREREGISTRATION.md`（预登记，含证伪条件）
- `scripts/compare_kovasznay_scheme_order.py`（双梯对比，新）
- `tests/test_kovasznay_order_discrimination.py`（9 项新门）
- `runs-32-64-128-zero_gradient/summary.json`（**修复**：补齐 `pair_consistency` /
  `three_level_estimates`，数字逐位不变）
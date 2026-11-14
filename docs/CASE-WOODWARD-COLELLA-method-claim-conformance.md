# CASE-WOODWARD-COLELLA — 做法登记一致性审计（XVAL-FVM-M18737-ROUND2 验证欠账）

**锚点**：`XVAL-FVM-M18737-ROUND2`（「TensorFVM 的 m18737-round2 做法」交叉验证）
**被审对象**：`oss_solutions#438`（repo=TensorFVM, feature_slug=CASE-WOODWARD-COLELLA,
tag=m18737-round2）登记的做法句——
「Primary scheme: MUSCL-Hancock 2nd + MC limiter + HLLC flux + SSP-RK (CFL 0.4)」
（claim text sha256 = `10e7671e…`，全文见 `docs/woodward-colella-double-blast/method-claim/claim_audit.json`）
**仓/改动面**：TensorFVM；新增 `src/tensorfvm/compressible/wc_method_claim.py`、
`tests/test_compressible_wc_method_claim.py`、本文件与 `method-claim/` 证据包；
`compressible/wc_case_pool.py` 的方案字符串更正（并重生成 `case_pool_entry.json`）。求解器一行未改。
**一句话裁决**：11 条原子声明中 **9 条与代码/工件相符，2 条不符**；两条不符均为**出处（provenance）
缺陷**，不触及精度结论——本案 <10% 判定由已提交剖面独立复算，与本审计无关。

## ① 问题设定：为什么这条账要还

锚点的「对标」不只是数值对比，还包含「被对标做法是什么」。round-1 报告（report id 17487）与
本仓报告都直接引用上述登记句描述 TensorFVM 一臂。此前**没有任何门禁把这句话绑到代码上**：
判据、阈值、误差数字都有测试，唯独「做法」是自由文本。若登记句与实现不符，读者无法从描述重建
TensorFVM 一臂，交叉对标在该方向上不可falsify。本单把这句话拆成 11 条可判定的原子声明，逐条
用源码行 / 运行时数据结构 / 已提交度量工件裁决，并落成门禁。

## ② 方法与工具链

`tensorfvm.compressible.wc_method_claim`：`probe_*` 各返回 `(found, evidence)`，
`audit()` 汇总为 `claim_audit.json`（+`claim_audit.md`），`claim_audit_figure.py` 出图。
探针全部是「程序化重算」，无手抄：

| 探针 | 判据手段 |
|---|---|
| reconstruction | `EulerSolver.step` 源码中 order==2 分支含半时间步预估（`U_half`、`0.5*slopes`） |
| limiter | **公式级**：对 `(a,b)=(1,2)/(-1,2)/(1,3)` 直接调 `reconstruct.mc` 比对 MC 通式 |
| flux | **运行时**：monkeypatch `solver.hllc_flux` 计数器，真跑 nx=16 三步，计数 >0 |
| time_integrator | 包内 `def` 名扫描（无 rk/ssp 实现）+ `EulerSolver(order=3/4)` 抛 `ValueError` |
| cfl | `DoubleBlastConfig().cfl` 与入池记录 `solver.cfl` 双读 |
| case.definition | 真建算例读初始场：三平台 1e3/1e-2/1e2、ρ≡1、u≡0、γ=1.4、总能量 275.02、t_end=0.038、两反射壁 |
| criteria | `TOLERANCE_PERCENT=10`（<台账 15%）、度量块名与「4 个最陡面」计数 |
| numbers.nx800 / convergence | 用 `reference/metrics_calibrated_nx{200,400,800}_hllc.json` 重算并与登记数字比对（两位小数舍入带 5.5e-3） |
| cross_code | 读 `xcode/wc_crosscode.py`：WENO5 + Rusanov + SSP-RK3 存在，且不含 `EulerSolver` 调用（零共享离散） |

## ③ 计算结果（程序化注入）

| # | 声明 | 状态 | 缺陷等级 |
|---|---|---|---|
| 1 | MUSCL-Hancock 二阶 | CONFORMANT | — |
| 2 | MC 限幅器 | CONFORMANT（公式级：1.5 / 0.5 / 2.0，Clawpack 约定） | — |
| 3 | HLLC 通量 | CONFORMANT（3 步内 96 次 HLLC 求值） | — |
| 4 | **SSP-RK 时间推进** | **NONCONFORMANT** | medium |
| 5 | CFL 0.4 | CONFORMANT | — |
| 6 | W&C 1984 datum via `cases.double_blast` | CONFORMANT | — |
| 7 | 容差 <10%（台账 <15%） | CONFORMANT | — |
| 8 | 五个 verdict 度量 | CONFORMANT | — |
| 9 | nx=800：2.03 / 2.01 / 0.18 %、faces ≤0.0012、PASS | CONFORMANT（重算 2.031 / 2.009 / 0.179 / 0.001178） | — |
| 10 | **峰值误差 20.76/10.58/2.03 %，观测阶 ~1.8** | 数字符合、**阶数不符** → NONCONFORMANT | low |
| 11 | 独立 WENO5/Rusanov/SSP-RK3、零共享离散 | CONFORMANT | — |

**F1（medium，出处）**：仓内 `compressible/` **没有任何 SSP-RK 实现**（`order` 只接受 1/2，
`step()` 是单步 MUSCL-Hancock 预估-校正）。更根本的是语义矛盾：MUSCL-Hancock 与 SSP-RK 是
**并列**的二阶时间推进方案，不能「+」在一起。登记句因此不可实现，也不可从描述重建。
精度内涵（时间二阶）由 Hancock 预估器交付，故本缺陷**不影响**数值结论。
**已处置**：仓内入池记录字符串更正为
「MUSCL-Hancock (2nd order in space and time, van Leer predictor-corrector) + MC limiter + HLLC」，
`case_pool_entry.json` 重生成；**建议**平台侧把 `oss_solutions#438` 的 method 文本与之一致
（本单不动 TensorPlatform/平台库）。

**F2（low，出处）**：登记数字 20.76/10.58/2.03 % 与工件逐位吻合，但「观测阶 ~1.8」**用任何标准
估计量都复现不出**：局部对阶 0.97（200→400）、2.38（400→800），三点最小二乘 1.18，两段均值 1.68。
方向性表述（介于 1 与 2 阶之间，激波主导）成立，但数字需标明估计量；建议改为
「阶数 0.97（200→400）/ 2.38（400→800）」。

## ④ 结果图

`docs/woodward-colella-double-blast/method-claim/fig_wc_claim_audit.png`
（左：登记数字 vs 工件重算；右：三种观测阶估计量 vs 登记「~1.8」红线）。

## ⑤ 注意点 / 局限

1. 本审计只审「登记句是否与代码相符」，**不重估**精度：nx=800 三项 <3% 的结论来自已提交剖面重算。
2. MC 限幅器实现采用 **Clawpack 约定**（`|s|=min(2|a|,2|b|,|(a+b)/2|)`，符号取居中斜率），
   与教科书 minmod-of-three 形式在**反号斜率**处不同（0.5 vs 0）；单调数据上二者一致，本单把这一
   约定显式写进探针，避免将来换约定而不被察觉。
3. 声明文本以 sha256 钉扎；平台侧登记行若改写，本审计的 claim 文本需同步换代（门禁会提示）。
4. 反号斜率处的 MC 与教科书形式差异是否需要单独复核 TVD 性，超出本单范围，记为后续线索。

## ⑥ 误差判定（是否 <3%）

| 关键量（本单复算） | 登记 | 重算 | 相对差 | <3% |
|---|---|---|---|---|
| 峰值密度误差 %（nx=800） | 2.03 | 2.031 | 0.04% | ✅ |
| L1 % | 2.01 | 2.009 | 0.05% | ✅ |
| 浅槽 % | 0.18 | 0.179 | 0.6% | ✅ |
| 最陡面 |dx| | 0.0012 | 0.001178 | 1.8% | ✅ |
| 峰值误差 %（nx=200/400） | 20.76 / 10.58 | 20.755 / 10.577 | 0.02% / 0.03% | ✅ |
| 观测阶「~1.8」 | 1.8 | 0.97 / 2.38 / 1.18 / 1.68 | 未复现（F2） | ❌ 如实记 |

结论：**数字全部 <3% 复现；唯一未复现的是「~1.8」这个阶数表述**，已按 F2 记录并给出建议改法。

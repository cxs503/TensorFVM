# m17485 交差核验：XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2（TensorFVM，m17360）

> 交差审计（非重做）。被审交付 `m17360` @ commit `98ab85edf309`（分支 `agent/m17360`，**未并入** main）。
> 本页与 `artifacts/audit_m17485/*.json` 全部程序化生成（`scripts/audit/render_audit_m17485.py`）。

**裁定：L3（board-audit）** —— L3 = 独立功能实现 + 接入仓内验证流水线 (crosscheck/effective_exponent/loglog_guard reach study.json).

## 1. 交付清点（三件套是否存在）

| 文件 | 字节 | sha256(前 12) |
|---|---|---|
| `docs/xval/convergence-diagnosis-v2/REPORT.md` | 15119 | `ca6bb13cce79` |
| `docs/xval/convergence-diagnosis-v2/artifacts/xval_convergence_diagnosis.json` | 164722 | `76f47e673c54` |
| `docs/xval/convergence-diagnosis-v2/artifacts/xval_table.md` | 9397 | `430f734aa00d` |
| `docs/xval/convergence-diagnosis-v2/figures/xval_committed_artifact.png` | 54342 | `4d5aad69cd76` |
| `docs/xval/convergence-diagnosis-v2/figures/xval_error_vs_h.png` | 103786 | `e4e2e0a0103e` |
| `docs/xval/convergence-diagnosis-v2/figures/xval_order_agreement.png` | 95147 | `995b2fbb0c60` |
| `docs/xval/convergence-diagnosis-v2/gci_reference.py` | 12948 | `27f806670bfe` |
| `docs/xval/convergence-diagnosis-v2/xval_driver.py` | 35494 | `d380dedf1e98` |
| `src/tensorfvm/verification/convergence_crosscheck.py` | 14055 | `55f0c2a50840` |
| `src/tensorfvm/verification/sod1978_convergence.py` | 26304 | `55239ea95dee` |
| `tests/test_convergence_diagnosis_xval.py` | 11170 | `8e8437cd7cd4` |

全部存在：**True**；被审树 HEAD 与交付 commit 一致：**True**；工作树残留未提交改动：0 条。
字节副本与 sha256 清单位于 `artifacts/audit_m17485/delivered/`、`manifest.json`。

## 2. 基准裁判（唯一裁判）

`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2'` → **0 行**（审计时刻实测，不粉饰）。
此时在force的判据 = mission 17360 goal 原文：**<15%（owner 精度标准：对标误差 >15% 不可接受）**。裁判表为空是资产缺口（本锚点未预登记算例），故本次核验以 mission goal 判据 + 审计席独立重算为准，并把两者分别记档。

## 3. 独立重算：15 % 判据逐项核对

审计席**自带**三点差分 GCI 实现（`scripts/audit/m17485_board_audit.py::_gci_ref_p`，既不 import `gci_reference.py` 也不 import `convergence_crosscheck.py`），对工件 `raw`/`sweeps` 的 phi/h 重算 p_ref；got 侧用仓内 estimator 对同一数列重算，最后与工件 `agreement_ref_vs_got` 对账。

**全量对账**：30 条 (sweep × 功能量) 行的 p_ref / p_got(lsq) / p_got(相邻对) / 收敛类，与工件差异 >1e-9 的条目 = **0**。

### 3.1 渐近区四项（判据 <15 %）

| 功能量 | p_ref | p_got(lsq) | p_got(相邻对) | Δlsq | Δsucc | <15 % |
|---|---|---|---|---|---|---|
| smooth_wave_l1 | 2.1116 | 2.0706 | 2.0454 | 1.94 % | 3.13 % | ✅ |
| l1_rho_full | 0.9386 | 0.8758 | 0.8573 | 6.69 % | 8.66 % | ✅ |
| l1_rho_cellavg | 0.9049 | 0.9218 | 0.9270 | 1.87 % | 2.44 % | ✅ |
| contact_x | 0.7868 | 1.4218 | 1.7080 | 80.72 % | 117.09 % | ❌ |

结论：**3/4 达标**，未达标 = `contact_x`。交付方在报告 3.1 与 ⑤ 中如实标注了该 ❌（honest FAIL），未粉饰；审计复现同一数字（80.72 %）。

### 3.2 非渐近区重叠三重组

| 功能量 | p(粗三重组) | p(细三重组) | 相对差（细为分母） | 相对差（粗为分母） | 工件值 |
|---|---|---|---|---|---|
| shock_x | 2.4315 | 1.0662 | 128.06 % | 56.15 % | 0.5615 |
| l1_rho_plateaus | 3.0016 | 4.7487 | 36.79 % | 58.21 % | 0.5821 |

## 4. 报告六段与数字防手抄

六段齐全：**True**（① 案例介绍 / ② 方法与实现 / ③ 计算结果 / ④ 结果图 / ⑤ 收敛性·敏感性 / ⑥ 失败模式）；差异清单=True，移植取舍=True，显式 <3 % 判定=True。

报告 3.1/3.2/3.3 头条数字逐项回算：

| 报告声称 | 审计重算 | 一致 |
|---|---|---|
| 2.1116（smooth_wave_l1 p_ref） | 2.11157 | ✅ |
| 0.9386（l1_rho_full p_ref） | 0.938594 | ✅ |
| 0.9049（l1_rho_cellavg p_ref） | 0.904902 | ✅ |
| 0.7868（contact_x p_ref） | 0.786759 | ✅ |
| 80.72（contact_x rel_diff_lsq %） | 80.718 | ✅ |
| 128.1（shock triplet drift %） | 128.058 | ✅ |
| -1.2494（published shock loglog exponent） | -1.24941 | ✅ |

**发现 F1（medium）**：`REPORT.md:81` 的「用 got 的 p」外推值写作 `0.263504`、误差 `64.6 %`，该数字在**任何工件中都不存在**；程序生成表 `xval_table.md:86` 与 JSON `sweeps.combined.../shock_x/derived_values.extrapolated_from_got_exponent` 均为 **0.745313**（与 ref 差 3.9e-5，0.00 %）。属人工誊写/陈旧数，不影响 15 % 裁定（未达标行是 contact_x），但违反零手抄纪律，应改。

## 5. 集成面（L3 判据）

`convergence_crosscheck` 的新函数：classify, triplet_order, richardson_extrapolated, gci, asymptotic_ratio, in_asymptotic_range, crosscheck_series, guarded_loglog_exponent。

仓内入口 `scripts/sod1978_convergence_study.py` 实跑（--quick）产出的 study.json 中，新增字段出现次数：crosscheck=24、effective_exponent=24、loglog_guard=24 → **新能力真的流到了流水线输出**（不是孤岛模块）。

无求解器侧/平台侧调用者 → 集成只到仓内验证流水线，**不满足更高级别的系统集成**。

## 6. 可复现

`repro_cmd`（工件内字段）：`PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/xval_driver.py --out docs/xval/convergence-diagnosis-v2`（单核 232.9 s）。
审计已把整树字节副本 + 驱动递交集群（`cj1554`，stage_dir=交付树副本，timeout 1800）；**报告时刻仍在排队**，故全驱动叶片级复跑 = **not_run（pending）**，不记为通过。已执行的替代证据：被审树内 `tests/test_convergence_diagnosis_xval.py` 11 passed、`tests/test_sod1978_convergence.py` 14 passed、`--quick` 流水线实跑、以及 30/30 数字行独立重算。

## 7. 定级与理由

| 级别 | 判据 | 结论 |
|---|---|---|
| L0 文档 | — | 通过（但不是功能） |
| L1 数学/单元验证 | 参考面解析自检 p=2 精确、两实现 29 组一致 | ✅ |
| L2 独立功能实现 | `convergence_crosscheck.py` 319 行真代码，输入数列→输出阶/类/GCI 带 | ✅ |
| L3 系统集成 | 被 `sod1978_convergence._rate_table` 调用并流到 study.json | ✅ |
| L4 基准验证 | 完整算例+误差分析+可复现 **且** 判据全过 | **未达**：裁判表 0 行；contact_x 80.72 % 超 15 %；全驱动复跑 pending |
| L5 科研工程成果 | 论文级科研产出 | 未申报 |

**裁定：L3。** L3 = 独立功能实现 + 接入仓内验证流水线 (crosscheck/effective_exponent/loglog_guard reach study.json). L4 NOT granted: referee table empty for the anchor (no pre-registered case), and the anchor's own <15% criterion fails on 1 of 4 asymptotic quantities (contact_x 80.72%, honestly disclosed), so the estimator-equivalence claim is not benchmark-validated; branch agent/m17360 not yet in main. Findings: F1_medium REPORT.md:81 quotes '0.263504' / '64.6%' for the shock extrapolation from got's exponent, which appears in NO artifact - program table (xval_table.md:86) and JSON give 0.745313 (delta 3.9e-5, 0.00%); does not change the verdict (contact_x is the failing row) but is an unverifiable hand number. F2_low the non-asymptotic '128.1%' triplet drift uses the fine triplet as denominator (audit symmetric check: 56.2% the other way) - both >>15%, conclusion unaffected.

## 8. 审计自身的局限

1. 未执行任何第三方 OSS 代码（OpenFOAM/SU2 的 GCI 工具未跑）；OSS 腿=文献公式级，属 `external_anchor` 的诚实弱化项。
2. 全驱动 232.9 s 复跑被控制面 120 s 截断，只能上集群；报告时刻 `cj1554` 仍 queued → 该项 = not_run。
3. 审计席的重算实现与被审实现同用三点差分公式（数学上同源），独立性限于「不共享代码」，不能排除公式层面的共同误读（缓解：仓内既有 `richardson_extrapolation` 作为第二实现，29 组 0 分歧）。

## 9. 处置

**approve with notes**：L3 记档；返修建议 = (a) 修 `REPORT.md:81` 的 0.263504/64.6 %（改成工件值 0.745313，0.00 %）；(b) 把本锚点的 15 % 判据登记进 `tensor_benchmark_cases`（当前 0 行，裁判缺位）；(c) `cj1554` 复跑回填 `repro.json` 后把第 6 节 not_run 改判。


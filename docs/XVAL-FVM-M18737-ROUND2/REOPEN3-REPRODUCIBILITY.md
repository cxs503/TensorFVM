# XVAL-FVM-M18737-ROUND2 — Reopen r3 复现性核验（m24511）

> **背景**: 原 PR#943（m21153 matured mission deliverables）owner 2026-11-05 裁定关闭后重开（第 3 轮）。
> 核验发现其全部 6 个交付文件已在 main（merge #726, c0456dfd 之前）——本单不重复交付，
> 改为在**当前 main** 上做复现性核验 + 修复核验中发现的缺陷 + 补防漂移测试门。

## ① 案例
Woodward–Colella 双爆轰（[0,1] 闭管，p=1000/0.01/100，t=0.038，γ=1.4）；
独立参考 WENO5/SSP-RK3/Rusanov@6400/12800，被定级解 MUSCL-Hancock/HLLC。

## ② 怎么做的
1. 逐文件核对 PR#943 清单 6 文件：全部在 main，行数与 PR 描述一致（111/73/73 行）。
2. 重跑 `wc_certified_ref.py`（纯 numpy，读库存剖面）→ 三道门 + 定级全量复算。
3. 新增 doc-vs-data 防漂移测试门（`tests/test_compressible_wc_certified_ref_reopen3.py`）。
4. 尝试集群 fresh 求解器重跑（job cj5237）→ **失败**（见 ⑤）。

## ③ 计算结果（程序化注入，零手抄）

| 量 | 本单复算 | 文档/库存值 | 一致 |
|---|---|---|---|
| 参考自认证 L1（6400 vs 12800） | 0.5437% | 0.544% | ✅ |
| 参考自认证 峰值 | 0.0230% | 0.023% | ✅ |
| 被定级 nx3200 L1 / 峰值 | 1.7756% / 0.7582% | 1.776% / 0.758% | ✅ |
| 被定级 nx6400 L1 / 峰值 | 0.8931% / 0.0444% | 0.893% / 0.044% | ✅ |
| certified / point_estimate | False / True | False / True | ✅ |
| 诚实上限 | 1.6007% | 1.6007% | ✅ |

## ④ 结果图
`figures/fig_reopen3_reproducibility.png`：左=库存剖面复验叠加（WENO5@12800 参考 vs 6400 vs MUSCL/HLLC@6400）；
右=各 HLLC 档 L1 vs certified 1% 杆与诚实上限 1.601%。

## ⑤ 注意点（含 honest-FAIL）
1. **缺陷修复**：`wc_certified_ref.py` 的 verdict `note` 为硬编码旧文案，声称诚实上限
   "stays <1%"，与同一 JSON 中 `certified=false`（上限 1.6007%>1%）自相矛盾 →
   已改为程序化注入、两分支均如实陈述。
2. **honest-FAIL（未完成项）**：fresh WENO5@6400/12800 求解器重跑（job cj5237，80 min）
   失败——集群检出为 slim checkout，`docs/**` 不在节点工作树，脚本路径不存在。
   剖面级 bit 级复现**未验证**；本单以库存剖面完整性+指标复算代替。补救路径：
   用 `stage_dir` 直送工作树或把脚本复制到节点可见路径重递。
3. 适用范围：本核验只对**库存工件**负责；若未来求解器或工件变更，测试门会失败报警。

## ⑥ 误差判定（逐项「是否 <3% / <1%」）
| 量 | 实测 | <3% | <1% |
|---|---|---|---|
| 参考自认证 L1 | 0.544% | ✅ | ✅ |
| 参考自认证 峰值 | 0.023% | ✅ | ✅ |
| nx6400_hllc L1（点估计） | 0.893% | ✅ | ✅ |
| nx6400_hllc L1 + 参考误差棒 | 1.601% | ✅ | ❌（certified=False 的绑定门，与 errata 一致） |

**结论**：PR#943 交付内容在当前 main 上**数值全部复现**，定级裁决（certified=False，
诚实上限 certified <2%）维持不变；另修复 1 处 note 文案缺陷、新增 7 个防漂移断言。

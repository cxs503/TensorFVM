# CASE-FVM-KOVASZNAY-ORDER — m19958 单：n=128 zero_gradient 收敛阶验证（提交状态）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，ledger 状态 `open`）
**判据（ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），以 32/64/128 三套网格上 log-log 最小二乘斜率计量
**外部基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/S0305004100023999

---

## 1. 本单做了什么

本单是**验证欠账清理**类：对既有结案声明做独立复核（adversarial review），并补齐
`CASE-FVM-KOVASZNAY-ORDER` 锚点判据所缺的 **n=128 zero_gradient** 收敛阶验证层。

**关键发现（复核）**：
- 既有 `closure_report.md` 的"✅ 通过"是在 **16/32/64** 上取得的（面内速度最小二乘斜率 1.808），
  而 ledger 判据写的是 **32/64/128**，两者不是同一组证据；且 n=16 已在渐近区之外。
  该结案声明**对判据是高报（overstated）**，ledger 保持 `open` 是正确状态。
- `ORDER_32_64_128_STATUS.md`（m18510）已识别此问题并固化 `validate_levels()` 混合配置门：
  `reynolds / symmetry_closure / convection_order / algorithm` 四项在序列内必须一致，
  不一致即 `SystemExit`。此门已存在并生效。

**本单实质增量**：补齐 **n=128 zero_gradient** 层（`zero_gradient` 物理对称闭合，
`du/dz = dv/dz = 0`，伪展向模态降到舍入级），使 32/64/128 三网格 log-log 斜率可完整计量。

## 2. 已确认的既有证据（n32/n64 zero_gradient，可从工作树 npz 现场复算）

| n | h | cells | vel L2 (full 3D) | vel L2 (u,v) | max\|w\| | p L2 (mean-free) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.125e−02 | 1024 | 3.435615e−03 | 3.435615e−03 | 1.60e−16 | 2.885142e−02 |
| 64 | 1.562e−02 | 4096 | 8.972614e−04 | 8.972614e−04 | 2.49e−17 | 7.958339e−03 |

`zero_gradient` 闭合，Re=40：**32→64 单段观测阶 = 1.93697**（全 3D 与面内一致，因伪展向模态已在舍入级），
去均值压力阶 1.8581，R²=1.000000。

## 3. n=128 提交状态

n=128 zero_gradient 计算已递交调度系统（**cj2438**，8 核，队列位次 9），
命令：`verify_kovasznay_order.py --output .../n128 --n 128 --reynolds 40 --symmetry-closure zero_gradient`。
该层完成后，`summarize_kovasznay_32_64_128.py` 将产出 32/64/128 三网格最小二乘斜率，
`tests/test_kovasznay_32_64_128_acceptance.py` 的 `anchor_zg` 全 3D 速度阶 ≥ 1.8 判据即可完整验证。

**本单收尾时 n=128 尚未落盘**（队列中），故本单**不声明达标**，ledger 维持 `open`。
这是诚实状态：判据要求的三网格斜率缺 n=128 一层，不能以 32/64 两段斜率冒充三网格斜率。

## 4. 给下次自找活的建议线索

- 待 cj2438 完成后：跑 `summarize_kovasznay_32_64_128.py --runs .../runs-32-64-128-zero_gradient --require-grids 32,64,128`，
  确认三网格斜率 ≥ 1.8，然后跑 `pytest tests/test_kovasznay_32_64_128_acceptance.py` 全绿，
  即可把 ledger `CASE-FVM-KOVASZNAY-ORDER` 从 `open` 置为 `verified`。
- 建议新条目（若本锚被他人完成）：`CASE-FVM-KOVASZNAY-PRESSURE-ORDER`——
  去均值压力在 32/64/128 三网格上的收敛阶 ≥ 1.8（当前 32→64 单段 1.8581，需三网格确认），
  属 TensorFVM 验证域。

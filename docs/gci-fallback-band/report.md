# GCI 两点回退带的保守度审计与修复（`richardson_gci` fall-back 路径）

## 1. 案例介绍（物理背景与对象）

本仓每条 benchmark 结论里的「±x %」都来自
`tensorfvm.verification.richardson.richardson_gci`：三网格（细/中/粗）解出观测阶
`p`，Richardson 外推给极限值，GCI 给带宽。带的可信度决定「某方案是否已收敛」这一
**机器判据**能否被外部复核。

三网格序列**不带阶信息**时，工具按 Roache/Celik 的惯例降级到两点回退：用形式阶
`p_theory`（未声明则一阶）+ 安全因子 `Fs=3`，docstring 承诺「broken refinement study
can never report a confident band」。上一轮 GCI 覆盖审计（m3149，
`docs/gci-band-coverage/report.md`）只审计了**单调路径**与两点误差项族，明确把
「两点回退的保守度」列为未审计项。本单就审计这条路径，并给出修复。

审计对象是纯 NumPy 工具，无求解器参与；案例族全部有闭式真值：

| 族 | 误差构造 | 真值 | 预期路径 |
|---|---|---|---|
| F-control | `A h^p` | `E` | 单调，**修复必须不动** |
| F-signed | `A h^p (-1)^i` | `E` | 非单调 → 回退 |
| F-phaselock | `A h^p sin(2π i φ)`, φ∈{0.30,0.48,0.62} | `E` | 非单调 → 回退 |
| F-cancel | `A(2h - 9h^2)`（两项反号，细网格首项主导、粗网格次项主导） | `E` | 单调但非渐近 |
| F-implausible | `(E+A, E+A(1+1e-3), E+10A)`（细≈中、粗远） | `E` | 阶求解爆大 |
| F-bias | `E + 0.5 + A h^p`（常数偏置） | `E`（非极限） | 单调，**无带可覆盖** |

扫描参数：加密比 `r21,r32 ∈ {1.5,2,3}`，`p ∈ {1,2,3}`，`A ∈ {0.02,0.1,0.4}`，共 648 例。
「修复前」由脚本内 `_pre_fix()` 逐例重算（旧规则原样复刻，不靠记忆）。

## 2. 怎么做的（方法/流程/工具链）

1. **复刻旧规则**：`_pre_fix()` 用同一套二分求根，但**无阶可信窗口**，
   且回退时 `eps = |f2-f1|/|f1|`（只看细-中差）。
2. **两处修复**（均在回退路径内，只可能变宽，不会变窄）：
   * **阶可信窗口**：`p_obs > 3·p_theory`（无形式阶时 `> 6`）判为
     `order-implausible`。机理：`GCI ∝ 1/(r21^p-1)`，`p` 爆大时带按 `r^-p` 塌缩，
     而爆大恰恰来自由细/中两个 metric 偶然抵消——那不是「收敛更快」，是没有信息。
   * **散度 `eps`**：回退时取三点**相邻差的最大者** `max(|f2-f1|,|f3-f2|)/|f1|`，
     因为非单调序列里细-中差可能偶然抵消成近零，用它算带宽等于在最没信息的地方
     报最自信的带。
3. **归因字段**：`RefinementStudy.fallback_reason ∈ {none, non-monotone,
   implausible-order}` 与 `order_limit` 显式落盘；`reference_agreement()` 回显
   `asymptotic / fallback_reason / safety_factor`，使带覆盖判定必须带着限定词传播。
4. 工具链：`scripts/audit_gci_fallback_band.py`（纯 NumPy+matplotlib，648 例秒级，
   `outputs/gci-fallback-band/` 全程序化产出，零手抄）；`pytest tests/test_gci_fallback_band.py`
   把规则与审计结论都钉死。

## 3. 计算结果

聚合表（由脚本打印，`outputs/gci-fallback-band/RESULTS.md` 同步）：

| 族 | 例数 | 回退例 | 覆盖率(修复前) | 覆盖率(修复后) | 最差效率 E(前) | 最差 E(后) | 带宽中位比 | gci 未变 |
|---|---|---|---|---|---|---|---|---|
| F-control | 81 | 0 | 1.0000 | 1.0000 | 0.000 | 0.000 | 1.000 | **81/81** |
| F-signed | 81 | 81 | 1.0000 | 1.0000 | **1.040** | 0.555 | 3.462 | 0/81 |
| F-phaselock | 243 | 243 | 1.0000 | 1.0000 | 0.333 | 0.173 | 5.921 | 0/243 |
| F-cancel | 81 | 0 | 0.0000 | 0.0000 | 54.121 | 54.121 | 1.000 | 81/81 |
| F-implausible | 81 | 81 | **0.0000** | **1.0000** | **5.80e13** | 0.415 | 6.48e7 | 0/81 |
| F-bias | 81 | 0 | 0.0000 | 0.0000 | 3.00e4 | 3.00e4 | 1.000 | 81/81 |

具体病例（`h=(0.1,0.2,0.4)`，`p_theory=2`，真值 `E=1`）：

| 病例 | 值 | 修复前 | 修复后 |
|---|---|---|---|
| 细≈中、粗远 | (1.10, 1.1001, 1.30) | `p=10.97`、GCI=5.7e-8、命中 False、E=1.8e6 | `order-implausible`、GCI=0.182、命中 True、E=0.55 |
| 相位锁定（细/中近抵消） | (1.0309, 1.0294, 1.20) | 旧 eps=1.4e-3、带远窄于误差 | GCI=0.165、命中 True、E=0.19 |
| 本仓负控（参考 1.0） | (.1, .08, .09) | GCI=0.200、E=4.47（**带不是界**） | GCI=0.200、E=4.47（整组值离参考 10×，非带缺陷，如实记录） |
| 本仓负控（参考 0.1） | (.1, .08, .09) | 覆盖、E=0.333 | 覆盖、E=0.333（不变：细-中差本就是较大者） |

图（`outputs/gci-fallback-band/`）：`fallback-band.png` 左=各族带覆盖真值比例（前/后），
右=最差效率指数 E（前/后，红线 E=1）；`collapsed-band-case.png`=细≈中病例上
「带 vs 误差」的前后对比：旧带 GCI=5.7e-8 对误差 0.100，窄了约 6 个量级，新带 GCI=0.182 覆盖。

## 4. 注意点（踩过的坑/局限）

* **旧回退承诺不成立**：`Fs=3` 乘的是「细-中差」，而细-中差在非单调序列里可以
  偶然抵消；本次 F-signed 族里旧规则最差 E=1.040>1（本仓自己记录的负控
  (.1,.08,.09) 对参考 0.1 也是 E=4.47），即「保守降级」其实不保守。
* **阶求解银弹风险**：`observed_order` 的括号到 40，任何「p 很大 = 收敛快」的解读
  都是错的；本修复只截断不修复——被截断的三网格等于两点研究，信息量确实只有两点。
* **F-cancel（单调但非渐近）未修**：覆盖率 0/81、E=54，但工具已标 `asymptotic=False`；
  GCI 只在渐近区有定义。把它也降级到 `Fs=3` 会**加宽带**、改动所有已发表报告的
  数字（含 m3149 钉死的覆盖率表），超出单席位授权，留作后续单（见 §6）。
* **F-bias 无法修复**：常数偏置不进带——三点网格法看不见不随 `h` 衰减的误差。
  这是方法的适用范围，不是实现缺陷。
* **单调路径 bit-identical**：81/81 例 `gci` 与旧规则逐位相同，`tests/test_richardson_gci.py`
  与 `tests/test_gci_band_coverage.py` 27 项全绿，故 m3149 报告与
  `richardson-gci-selfverification/summary.json` 内的带宽数字**无需重算**
  （其两条 inconsistent 负控 (.1,.08,.09)/(.1,.09,.095) 的 GCI 仍为 0.200/0.100，已核）。

## 5. 误差判定（逐项，显式回答是否 <3%）

本单的「基准」是闭式真值 `E=1`，判定量是「带是否含 `E`」与「效率指数 ≤1」，
而非百分比误差；仍按要求给出可量化误差项：

| 判定量 | 修复前 | 修复后 | 是否达标 |
|---|---|---|---|
| F-implausible 覆盖率 | 0.0000 (0/81) | 1.0000 (81/81) | 是 |
| 最差效率指数 E（回退族） | 5.80e13（等价相对误差 ≫3%） | 0.415（带为界且 58% 余量） | 是 |
| F-signed 最差 E | 1.040（超界 4.0%>3%） | 0.555 | 是 |
| 单调族 `gci` 漂移 | — | 0（81/81 逐位相同，相对误差 0% <3%） | 是 |
| 阶可信窗口的实测触发 | — | 只触发于 `p>6`（`p_theory=2`），单调族触发 0/81 | 是 |
| F-cancel 覆盖率 | 0.0000 | 0.0000 | **否（未修，如实记录）** |
| F-bias 覆盖率 | 0.0000 | 0.0000 | **否（范围外，方法看不见常数偏置）** |

## 6. 后续单建议

* 若 owner 批准改动已发表带宽：把「`asymptotic=False` ⇒ 带降到 `Fs=3`」作为统一约定，
  重跑 m3149 覆盖率表并重算受影响的 docs 数字（预计覆盖 F-cancel 81/81）。
* 在 `reference_agreement` 消费侧加消费门：`band_covered` 只在 `asymptotic` 为真时
  才允许被引用为「参考一致」，否则一律写「带含参考但不能宣称渐近」。

## 7. 产物与复现

```
PYTHONPATH=src python scripts/audit_gci_fallback_band.py     # 648 例，秒级，确定性
PYTHONPATH=src python -m pytest tests/test_gci_fallback_band.py -q
```

* `outputs/gci-fallback-band/audit.json`（聚合表 + 48 例逐例节选，45 KB，全量可秒级重生成）
* `outputs/gci-fallback-band/RESULTS.md`、`fallback-band.png`、`collapsed-band-case.png`
* 代码：`src/tensorfvm/verification/richardson.py`（`ORDER_CAP_FACTOR/ORDER_CAP_ABS`、
  `fallback_reason`、`order_limit`、回退 `eps` 取最大相邻差）
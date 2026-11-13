# AUDIT — CASE-FVM-DAMBREAK-STOKER 证据包独立复核（adversarial review）

**任务**: m17055（verifier 席 selfstart，锚点 `CASE-FVM-DAMBREAK-STOKER`）
**日期**: 2026-11-13
**被审对象**: `docs/case-fvm-dambreak-stoker/REPORT.md` +
`src/tensorfvm/verification/artifacts/shallow-water-dam-break/{SUMMARY.md,metrics.json,exactness.json}`
（commit 基线 `b94c7440`）+ `tests/test_shallow_water_dam_break.py`
**独立性说明**: 审计者未参与该算例的原始实现；复核只依赖仓内代码、提交的工件与重跑，
不引用任何未落盘的数字。全部数字可用本文件 §5 的命令复现。
**范围**: 锚点验收项本身（`x_f(t)` 与静水区水位阶跃 vs 解析，t*=2，波形单调无伪振荡，
容差 5%），不覆盖 SWE 求解器其余能力（well-balanced / friction / 2D 另有专账）。
**未运行**: 3200 单元以上的收敛序列（本机 load15 ≈ 100，按容量纪律只跑到 n=1600）；
GPU/并行路径；其它边界条件。

---

## 1. 结论（disposition）

**request changes** — 原证据包不足以支撑锚点验收句，已在本席内修复（见 §4），
修复后重新判定为 **approve with notes**：验收项在 n ≥ 800 上全部满足 5%，
但其中一项（湿床 t\* ≲ 0.02 的初始水跃）余量只有 ~10%，属"通过但记录在案"。

原内容最实质的问题**不是数值精度不够，而是被发表的那个数字不是被分级定义的那个数字**
（下 F1/F2）。

---

## 2. 分级发现（severity / likelihood / evidence）

### F1（高，已修）发表的"波前位置"是深度地板探测器，与分级 metric 不是同一个量

`run_shallow_water_dam_break.py::_summary_markdown` 的干床表格首行把
`front_wetted`（最外侧 `h>0` 单元）印成 "front position x_f"，而
`dam_break.py::_verdict` 分级的是 `front_fan<5%`：

| 量 | n=1600 值 | 相对误差 | 出处 |
|---|---|---|---|
| 发表行 `front_wetted`（原文 SUMMARY.md） | 3.884375 m | **2.891 %** | 基线工件 |
| 分级量 `front_fan` | 3.999 32 m | **0.017 %** | 本次重跑 `metrics.json` |

同一份工件里，两行"前缘位置误差"差 **170×**。且 `front_wetted` 在 n=400…3200 上
给出 1.562 / 2.656 / 2.891 / 2.461 %，**非单调**——这正是
`docs/verification-dam-break-front-metric/REPORT.md` 与
`dam_break.py` 注释自己声明的"深度地板探测器、不收敛、且会反转格式阶数"的缺陷签名。
即：**仓内已经诊断并修好的缺陷，仍然留在对外发表路径上**（修补只落到 verdict，
没落到报告生成器）。复现：`git show b94c7440:src/tensorfvm/verification/artifacts/shallow-water-dam-break/SUMMARY.md`。

### F2（高，已修）验收句里的 `x_f(t)` 序列从未被分级，且用的是缺陷 metric

锚点验收是**随时间的波前位置**（`x_f(t)` vs 解析）。基线里 `run_case` 确实记录了
`time_to_front_series`，但（a）没有任何断言/判据读它，（b）干床样本由 `front_sample`
用 `_wetted_edge` 构造。实测该序列在最早采样点（t\* = 0.005, n=800）误差 **212.5 %**，
且 **与网格无关**（n=400/800/1600 均为 2.125，rel 1e-13）——说明读数由湿区前驱
（HLL 零深度前驱 5 个单元）与深度地板决定，不是离散误差：

| n | 最早样本 t\* | legacy(湿润边缘) 序列最大误差 | fan（分级）序列最大误差 |
|---|---|---|---|
| 400 | 0.010 | 212.5 % | 6.61 % |
| 800 | 0.005 | 212.5 % | 1.40 % |
| 1600 | 0.0025 | 212.5 % | 0.43 % |

即：**验收项在一个 212% 的数字上"通过"，只是因为没人去读那一列**。

### F3（中，已修）敏感性表的"worst headline error"也建在缺陷 metric 上

`sensitivity_study` 的 `dry_front_error` 取 `front_wetted_error`，
故 `worst_headline_error` 报 2.969 %（≈ 容差的 60 %），实际分级量最差值只有
**0.395 %**（order=1 变体）。把"最坏情况只剩 40% 余量"这种假警报写进工件，
会诱导后续为了"救余量"而改动本来正确的格式。

### F4（中，已记录未修）湿床平台窗口与激波电平由解析解构造（oracle-guided）

`_wet_metrics` 用 **解析** `x_fan_end`、`x_shock` 划定采样窗 `[lo, hi]`，
用 **解析** `h2` 定电平 `0.5(h1+h2)`：

* 后果 A：`h2/step_ratio` 只测"平台幅值"，对"平台位置对了吗"完全不敏感
  （位置错了解析窗也会跟着错）。n=1600 上 `step_ratio_error = 5.76e-7`，
  这个量级本身就是"测量近乎同义反复"的告警，不应被当作精度证据引用。
* 后果 B：即便激波位置偏 5%，`h2_error` 仍可能极小 → 两项不能互相替代
  （当前证词把它们并列，读者会误以为 `h2` 也独立验证了位置）。
* 处置：**不改**（改采样窗会改变历史可比性），改为在报告 §5 显式标注
  "`h2`/`step_ratio` 是幅值量、位置窗由解析解给定，二者非独立证据"。

### F5（中，已记录）湿床初始水跃未分辨偏差 4.505 %，与网格无关

湿床序列在 t\* ≲ 0.02 的采样点误差恒为 **4.505 %**（n=400/800/1600 一致，
差异在 1e-13 量级）。这是初始间断被 HLL 抹平在一个单元内的直接后果
（该时刻水跃位移 < 1 个单元），属**已分辨不了的物理**，不是估计子偏置：
它只出现在"精确解已走 < 4 个单元"的样本上。已按 **先验** 分辨率判据
`SERIES_MIN_CELLS = 4`（声明于模块常量，非拟合所得）把这些样本移出分级集合，
但**照实登记**在 `front_series.unresolved_max_error`：4.505 %，
即"若强行纳入，仍在 5% 内但余量仅 10%"。

### F6（低，已修）交付文档陈述与代码相反

`REPORT.md` §5 第 1 条声称"干床波前采用最外侧 h>0 的单元定义，**无需阈值**，
避免人为参数"——而修复后的分级估计子恰恰需要并公开声明了深度分数窗
`FAN_WINDOW = (1e-2, 2e-1)`（`dam_break.py`）。原文把"有阈值"说成"无阈值"，
并把它当优点列出。§3.2 的 2.66 % 亦为 legacy 值。

---

## 3. 未发现问题的部分（避免过度怀疑）

* 解析解自证：Stoker 的 Riemann 不变量与 R-H 跳跃残差 < 1e-12，
  Ritter 扇区恒等式（4/9、2/3、`∫h dx = h0 c0 t`）< 1e-14 — 可信，
  且不依赖任何外部表格（`exactness.json`）。
* 守恒/正性/单调性：mass drift ≈ 1e-14，`h_min ≥ 0`，clipped 0，
  内部极值 0（振幅阈 1e-4·h0），TV 比 ≤ 1.0031 — 与 TVD+HLL 理论一致。
* 深度 L1 收敛阶 n=100…1600 实测 1.01（干）/ 0.99（湿），含间断问题的
  一阶收敛是理论预期，不是缺陷（原文此处判断正确）。
* 单点 t\*=2 的湿床激波位置（0.043 % @ n=800）经复跑可复现。

---

## 4. 本次修复（最小、可验证）

1. `dam_break.py::front_sample` 增加 `headline` / `headline_error`（干床＝
   `fan_front_estimate`，湿床＝mid-level crossing）+ `resolved` / `travelled_cells`；
   `position`/`error` 保留为 legacy 定义并注明**不得分级**。
2. 新增 `series_metrics()` 与 `metrics["front_series"]`：整条 `x_f(t)` 序列的
   最大相对误差、分级样本数、未分级样本（含估计子未定义样本）及 legacy 对照值；
   判据 `pass = max_error ≤ TOLERANCE`。
3. `sensitivity_study` 增加 `dry_front_fan_error` / `dry_series_max_error`，
   `worst_headline_error` 改用分级量。
4. `run_shallow_water_dam_break.py` 摘要：干床同时印"分级估计子（首行）"与
   "湿润边缘（标注为诊断量）"，并新增 `x_f(t)` 序列小节。
5. 工件 `front_series` 拆成 `front_series_samples` 与 `front_series_summary`
   （原名把"原始样本"叫成 series，没有 pass/fail 陈述）。
6. 新增 `tests/test_dambreak_stoker_series.py`（8 项）：分级序列在 n=800 通过、
   两种定义并存、legacy 缺陷签名锁定（> 200 % 且与网格无关）、干床序列收敛、
   湿床初始偏差如实登记、空输入守卫、提交工件仍发表分级量。

修复后（`docs/case-fvm-dambreak-stoker/REPORT.md` §3 数字程序化注入自
`src/tensorfvm/verification/artifacts/shallow-water-dam-break/`）：

| 验收项 | 判据 | n=800 | n=1600 | 判定 |
|---|---|---|---|---|
| Ritter `x_f(t)` 序列最大误差 | < 5 % | 1.40 % | 0.43 % | PASS（n≥800；n=400 为 6.61 % → 记录为粗网格失效） |
| Stoker 激波位置 `x_s(t)` 序列最大误差（分级样本） | < 5 % | 0.80 % | 0.84 % | PASS |
| Stoker 静水区水位阶跃 `h2/h1` | < 5 % | 2.1e-6 | 5.8e-7 | PASS（幅值量，见 F4） |
| 波形单调（内部极值 > 1e-4 h0） | = 0 | 0 | 0 | PASS |

---

## 5. 复现命令

```bash
# 分级序列（干/湿，n=400/800 已在测试夹具内）
PYTHONPATH=src python -m pytest tests/test_dambreak_stoker_series.py -q
# 完整工件（~54 s 单核）
PYTHONPATH=src python scripts/run_shallow_water_dam_break.py --n 1600 \
    --resolutions 100,200,400,800,1600
# 对照基线里"发表的 2.891 %"
git show b94c7440:src/tensorfvm/verification/artifacts/shallow-water-dam-break/SUMMARY.md
```

## 6. 剩余风险 / 下次审计线索

* 湿床平台窗口的 oracle 依赖（F4）只被记录、未被消除；消除需要"无需解析位置"
  的平台检测器（例如对 h 的直方图双峰距），属独立选题。
* `tests/test_dam_break_front_metric.py::test_wet_branch_verdict_keys_unchanged`
  把 verdict 键集锁死，故本次 **未把序列判据加进 verdict**（改为 `metrics`
  内独立判据 + 测试），以免静默改变对外契约；若要把序列判据升为 headline，
  需同批更新该锁与其文档。
---

## 7. 追加复核（m18263）：F4 修复的量纲缺陷 —— 格心读数（F7）

**范围**: 只审 F4 的修复本身（`data_driven_plateau`，本单之前新增、未被他审覆盖），
不重开 F1–F6。**判定: request changes → 已在本席修复（见 §7.4）。**

### 7.1 F7（高，已修）`bore_x` 是"格心读数"，不是位置量：分辨率地板 = dx

F4 的修复用 **最大梯度单元的中心** 作 `bore_x`。这是 F1 的同型缺陷：
读数被量化到格心，因而

* 精化不单调、地板 `O(dx/2)`；且
* **结构上无法表达亚格位移** —— 而"位置量"的全部意义就是能分辨小于一格的位移。

实测（`scripts/verify_dambreak_bore_subcell.py`，n = 200/400/800/1600，t\*=2）：

| n | dx [m] | 格心读数误差 | 折合 [cell] |
|---|---|---|---|
| 200 | 0.05000 | 0.730 % | 0.276 |
| 400 | 0.02500 | 0.068 % | 0.051 |
| 800 | 0.01250 | **0.399 %** | **0.602** |
| 1600 | 0.00625 | 0.233 % | 0.705 |

n = 400 → 800 处误差**反向增大 12 倍**（0.051 → 0.602 cell）：这不是离散误差，
是 argmax 落在被 HLL 抹平的跳跃里哪一格中心上的随机结果。后果与 F2 相同 ——
**该列数字不可用于"收敛"陈述，也不可作 5% 之外的任何分级**。

### 7.2 为什么这条不属于"数值精度不够"

同一份状态、同一份工件里，改用"在两**测量**状态之间插值"的亚格读数后误差为
0.292 % → 0.160 % → 0.043 % → 0.021 %（0.110 → 0.121 → 0.065 → 0.063 cell），
**每级都在半格以内**，最差 0.121 cell，比格心读数好 5.8 倍。即：解本身没有变化，
是**估计子**有缺陷。

### 7.3 F4 遗留的"幅值/位置不可分辨"论证仍然成立但不完整

F4 只论证了"格心位置 ≠ 幅值"，没有证明读出的位置**真的能动**。本次两向探针补齐：

| 探针（n=800，numpy 合成扰动，纯 numpy 无求解器） | 结果 |
|---|---|
| 把剖面平移 0.1…0.9 格（幅值不变） | 亚格读数残差 ≤ **0.0185 cell**；格心读数残差最大 **0.700 cell** |
| 平台均匀加高 1.5 %（位置不变） | `h2` 变 **+1.50 %**，`bore_x` 只动 **−0.026 cell** |

即：幅值探针只惊动幅值、位置探针只惊动位置 —— 两者是**独立证据**，
`h2`/`step_ratio` 不再是"位置错也对"的同义反复（F4 后果 B 已被量化消除）。

### 7.4 本次修复（最小、可验证）

1. `dam_break.py::data_driven_plateau`：`bore_x` 改为**中值位面穿叉插值**
   （`level = 0.5(h2+h1)`，h2/h1 均为剖面测量值，解析量不参与），
   新增 `bore_x_cell`（旧格心读数，保留为负对照）、`bore_level`、`bore_method`。
2. 新增先验判据常量 `BORE_SUBCELL_MAX_CELLS = 0.5`（半格；非拟合）。
3. `_data_driven_crosscheck` 增列 `bore_x_cell`/`bore_x_cell_error`/`bore_level`/`bore_method`。
4. 新增 `scripts/verify_dambreak_bore_subcell.py` → 工件
   `src/tensorfvm/verification/artifacts/shallow-water-dam-break/bore_subcell.json`
   + 图 `docs/case-fvm-dambreak-stoker/bore_subcell.png`。
5. 新增 `tests/test_dambreak_bore_subcell.py`（7 项）：半格判据、格心量化负对照
   （含"精化反向"签名）、亚格平移可分辨（≤0.1 cell）、幅值/位置独立性（双向）、
   提交工件自洽、退化输入。

### 7.5 照实记录（未修）

* n = 200 的**格式**本身在湿床 t\*=2 有 1 个振幅 > 1e-4 h0 的内部极值
  （`monotone(no_extrema)` FAIL）。这与本次度量为无关：是粗网格迭代器/限制器
  的过冲，已写进工件 `verdict_failing` 照实登记，不改阈值救它。
* 亚格读数误差在 n=400→800 **轻微反向**（0.110→0.121 cell），
  仍在半格判据内；不宣称严格单调收敛（样本量 4，不构成收敛阶结论）。

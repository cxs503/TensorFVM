# 交差审计 — CASE-FVM-BLASIUS-PLATE（mission 16182，TensorFVM）

审计单：m16204（任务榜·交差核验，非重做）　对象：m16182 `succeeded`，提交 `cdbb3096`
（`agent/m16182`，2026-11-12 10:37Z）　审计位：`board-audit`　审计时间：2026-11-12 10:5xZ
判级结论：**L3**（不是 L4；差异见 §5）　处置：**approve with notes**

---

## 1. 范围与方法（先声明我做了什么、没做什么）

审计对象是 m16182 的交付物本身，不是它的报告文采。判据只有两个来源：

1. **平台基准表** `tensor_benchmark_cases`（唯一机器裁判）→ 见 §3，**该 anchor 0 行**；
2. 算例自带的判据 `ERROR_LIMIT = 0.03`（`src/tensorfvm/case_blasius_plate.py:67`）
   与算例单（`Re_x = 1e5` 局部 `c_f` 与 `H = δ*/θ` 相对 Blasius < 3 %）。

审计动作（全部可复跑，工件在 `runs/m16204/`）：

| # | 动作 | 命令 | 结果 |
|---|---|---|---|
| A1 | 读交付物五文件 + 4 张图 + 4 份工件 | `git ls-files docs/case-blasius-plate`、`wc -l` | 1052 行代码/文档，工件与图**均已入库**（`docs/case-blasius-plate/artifacts/*.json`、`figures/fig1..4.png`，48–92 KB） |
| A2 | 跑交付方测试 | `PYTHONPATH=src python3 -m pytest tests/test_case_blasius_plate.py -q`（在 m16182 树） | **7 passed in 12.02 s**（`runs/m16204/pytest_audit.log`） |
| A3 | **独立复现**（空输出目录，不带 `--summary-only`） | `PYTHONPATH=src python3 scripts/run_case_blasius_plate.py --out /tmp/repro16204c --configs march,march-sensitivity`（60.5 s） | 与提交工件**逐位相同**（`runs/m16204/repro_compare.json`：`march.json` / `march-sensitivity.json` / `summary.json` 三份 `n_diffs = 0`） |
| A4 | **文献锚撑独立重算**（不用本仓代码） | `runs/m16204/audit_probe.py::blasius_independent`（自写 RK4 + 打靶 + Simpson） | `2f''(0) = 0.66411467`（+0.0173 %）、`δ*√Re = 1.7207877`（−0.0007 %）、`θ√Re = 0.6641146`（+0.0173 %）、`H = 2.5911003`（−0.0193 %） |
| A5 | **种子独立性压力测试**（本审计新增，见 §6-F6） | 同脚本 `march_probe`，4 种初始剖面对比 | 全部 PASS，`c_f` 误差 −0.67 % … −1.02 % |
| A6 | 工件自洽性 | `audit_probe.py::P3_artifact_recompute` | `c_f`/`H` 与工件一致到 1e-16；连续性残差 3.32e-13 |
| A7 | 交付是否进主干 | `git rev-parse main origin/main`、`git ls-tree HEAD` | `main = 0cd53fa5`＝`cdbb3096` 的父提交，主干**无** `case_blasius_plate.py` |

**未做（不得当作通过）**：2-D FVM 旗舰档重跑（需 1.2–2.5 h/网格级，本单时间窗内不做）；集群作业；
march 求解器的独立重写；与实验/其它数值解的第三方对照（层流平板除 Blasius 无更强基准）；
`_surface_force` 逐面和恒等式的独立验证（仅由交付方测试 `force_sum_closure < 1e-12` 覆盖）。

**独立性局限**：本审计复用被审计树 `src/` 的 `march_boundary_layer`（被审计的正是它，无法自证）；
文献侧与判据侧（A4/A5/A6）由审计方独立代码生成，未调用算例模块的参考常量作结果。

---

## 2. 三件套可行性核验（逐项）

| 件 | 判据 | 实况 | 判 |
|---|---|---|---|
| `run_artifacts` | 文件真实存在且与报告数字同源 | `docs/case-blasius-plate/artifacts/{march,march-sensitivity,summary,fvm-partial-traces}.json`（**已 commit**，`git ls-files` 可见）；`runs/blasius-plate/*.json` 为未入库的同源工作副本，`diff` 逐字节相同 | ✅ |
| `benchmark ref vs got` | 报告数字 == 工件数字 == 复现数字 | 工件 `c_f` 2.084919e-3 vs ref 2.099752e-3（−0.7064 %）；`H` 2.579450 vs 2.5916（−0.4688 %）；`ref_cf = 0.664/√1e5 = 2.099752e-3` 算术自洽；三者在 A3 复现中**逐位重现** | ✅ |
| `repro_cmd` | 可直接执行并得出报告结论 | 报告 §⑥ 的命令（`--configs march,march-sensitivity`）60.5 s 跑通并逐位重现；**但** mission `checks` 里引用的 `... --summary-only` 版本在空目录下给出 `rows=[] / all_passed=false / exit 2`（A3 副产物 `repro-clean/summary-summaryonly-only.json`）→ 见 F4 | ⚠️ 部分 |

---

## 3. 基准裁判（`tensor_benchmark_cases`）与六段齐备性

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug = 'CASE-FVM-BLASIUS-PLATE';   -- 0 rows
```

**该 anchor 在平台基准表中未登记**（与 m16174 对 m16075 的 F3 同型）。因此本轮判据只能取
算例自带 `ERROR_LIMIT = 0.03` 与任务单文本；机器无法从注册表重新推导容差。这是平台侧登记缺口，
不是 m16182 的执行缺陷（m16182 无权写注册表也要点明）。

报告六段（①案例介绍 ②方法 ③参数表 ④对照表 ⑤敏感性/收敛 ⑥失败模式）**齐备**，且数字全部可溯源到
工件（A3 逐位复现即证明零手抄）。这部分是 m16182 最扎实的地方。

---

## 4. 判据逐项（谁真的达标）

| 目标量 | 判据 | 声明设备 | got | ref | 误差 | <3 % ? |
|---|---|---|---|---|---|---|
| 局部 `c_f` @Re_x=1e5 | −0.7064 % | **march（薄剪切层推进，层流）** | 2.084919e-3 | 2.099752e-3 | −0.71 % | **是** |
| `H = δ*/θ` | −0.4688 % | march | 2.579450 | 2.5916 | −0.47 % | **是** |
| `c_f`（ny=512） | −0.1524 % | march | — | — | −0.15 % | **是**（网格收敛单调） |
| `c_f` / `H` 后缘列 | — | **fvm（二维 SIMPLE + 贴体拉伸网格）** | 1.35e-2（75 步，未收敛） | 2.0998e-3 | **+545 %** | **WITHHELD**（不声明） |

关键事实：**达标的是 march 档，不是 FVM 档**。FVM 旗舰档在 75 个外层迭代后 `c_f` 仍为目标值的 6.44 倍、
动量残差 8.95e-3 仍在下降（`artifacts/fvm-partial-traces.json`），交付方如实标 WITHHELD、不给勾叉。

---

## 5. 判级（L0–L5）与差异说明

**判 = L3。**

* L1 数学/单元验证：✅ 相似解自审（`similarity_crosscheck`）+ 剖面积分测试把 `δ99` 截断偏差钉在 0.991。
* L2 独立功能实现：✅ 真代码，输入（Re_L/ν/ny/stations）→ 计算（推进 PDE + 剖面积分 + 无滑移面力）→ 输出（JSON 工件）。
* L3 系统集成：✅ 算例跑在本仓**既有求解器**上（`rans_ship_resistance.boundary_layer.march_boundary_layer`、
  `body_fitted.BodyFittedSolver`），非孤立玩具；但算例模块本身无任何调用者（case→solver 单向，属正常方向）。
* **L4 基准验证：未达**，缺的正是"完整算例"里的**主设备**：

| L4 要件 | 实况 |
|---|---|
| 完整算例（锚点定义的设备） | ❌ 锚点名 `CASE-**FVM**-BLASIUS-PLATE`，FVM 档未收敛 → 该设备未验证 |
| 误差分析（收敛/敏感性） | ✅ `ny` 128/256/512 单调 −1.81 %→−0.71 %→−0.15 %；站位 101/401/1601 亦在 1 % 内 |
| 可复现 | ✅ 独立空目录 60.5 s 逐位重现（A3） |
| 外部锚 | ✅ Blasius 1908 / Schlichting §7 T7.1，本审计自写 RK4 复算 <0.02 %（A4） |

即：**march 档的证据本身是 L4 级**（完整+误差分析+可复现+外部锚独立复核+种子独立性），
但把它记到 `CASE-FVM-…` 这个 scope 上，就等于向台账声明"二维 SIMPLE 在 Re=1e5 平板层流已基准验证"，
而事实是它偏 545 % 且未收敛。级别=功能实质，故 **L3**，并把"重开条件"写进 evidence：

> 若 FVM 档收敛到 <3 %（或锚点被重切为 `CASE-BLE-BLASIUS-PLATE` / march 档），本 scope 应升 L4。
> 若只重切锚点，则 march 档已具备 L4 证据，无需重跑。

---

## 6. 发现（按严重度；每条附证据与复现）

**F1 [high] 锚点定义的设备未验证。** 二维不可压缩 SIMPLE（`mesh_type='flat-plate'` 贴体网格）
在 75 次外层迭代后 `c_f = 1.353e-2`（目标 2.0998e-3，+545 %）、`momentum = 8.95e-3`，交付方自判 WITHHELD。
证据 `docs/case-blasius-plate/artifacts/fvm-partial-traces.json`（`fvm-baseline` 第 75 步）。
影响：该 anchor 不能被计为"TensorFVM 的 FVM 离散在平板层流基准上已验证"。
复现：`PYTHONPATH=src python3 scripts/run_case_blasius_plate.py --configs fvm-baseline --max-iterations 75`
（交付方实测 4.4–8.5 s/迭代，收敛需 ~1000 步 ≫ 时间窗）。**非审计单可在本窗口内补齐。**

**F2 [medium] 交付只在分支上，未进主干。** `git rev-parse main` = `0cd53fa5`；`cdbb3096` 的父提交正是它；
`git ls-tree -r HEAD | grep case_blasius` → 0；`git branch -a --contains cdbb3096` → 仅 `agent/m16182`。
影响：主干包内无法 import 该算例，平台侧任何"按主干复核"的动作都会看不到交付物（本审计在 m16182 工作树内完成核验）。

**F3 [medium] 平台基准登记缺口。** `tensor_benchmark_cases WHERE anchor_slug='CASE-FVM-BLASIUS-PLATE'` → **0 行**，
容差只存在于代码常量与任务文本，机器裁判缺位（同 m16075 的 F3）。

**F4 [low] mission `checks` 引用的复现命令在空目录下自证 FAIL。** `--summary-only` 是"只合并既有 `*.json`"
模式（`scripts/run_case_blasius_plate.py:87,95`），空 `--out` 下得 `rows=[]`、`all_passed=false`、`exit 2`。
报告 §⑥ 给出的命令不含 `--summary-only`，是**正确**的；只有 mission checks 行有此瑕疵。
证据 `runs/m16204/repro-clean/summary-summaryonly-only.json`。

**F5 [low] `flat-plate` 压力修正内层容差硬编码、无配置出口。**
`src/tensorfvm/body_fitted.py:783-785`：`threshold_floor=1e-8 if flat_plate else 1e-10`、
`relative_tolerance=1e-8 if flat_plate else 1e-9`。报告 §⑥ F3 指认它是唯一耗时源（约 1341 次内层调用/步）；
本审计按行号复核**属实**。修复（加 config 出口）属 TensorFVM 仓改进项，不在交差单范围。

**F6 [low，反向澄清] "用 Blasius 剖面做初值 ⇒ 循环自证"的怀疑被否定。**
审计方用 4 种初始剖面推进同一设备（默认 Blasius 种子 / 线性 / 1/7 幂律 / 均匀）：
`c_f` 误差 −0.71 % / −0.67 % / −0.99 % / −1.02 %，`H` 误差 −0.47 % / −0.47 % / −0.47 % / −0.47 %，
全 PASS（`runs/m16204/audit_probe.json::P2_seed_independence`）。抛物型推进对初值记忆快速衰减，
结论是**设备自身的性质**而非种子的性质。（`H` 的 −0.47 % 平台值来自 `ny=256` 离散，`ny=512` 降到 −0.30 %。）

**F7 [low] 前缘误差带已披露。** `Re_x ≲ 1e3` 处 `c_f` 误差可达 9.53 %（`artifacts/figure-summary.json`），
比较站位 1e5 不在带内；报告 §⑤ 已写明，不构成虚标。

---

## 7. 被夸大的声明（overstated claims）

1. mission `result` 的 `checks` 行把 `--summary-only` 版本当成"跑出 PASS"的复现命令 —— 该命令只在**已填充**
   的输出目录上成立（F4）。
2. 锚点名 `CASE-FVM-BLASIUS-PLATE` 本身构成"FVM 已基准验证"的暗示 —— 实际只有 march 档达标（F1）。
   交付报告没有做这个暗示（明确 WITHHELD），是**锚点命名**与实质的落差，不是报告造假。

除以上两点，报告 175 行内未发现与工件不符的数字（A3 逐位复现覆盖全部主表数值）。

---

## 8. 处置（disposition）

**approve with notes** —— 交付物真实、诚实、可复现，三件套齐备（`repro_cmd` 有一处引述瑕疵）；

**判级 L3**：march 档验证质量已达 L4，但锚点定义的 FVM 设备未验证，故不在 FVM scope 上记 L4。

**建议（返修不在本单）**：
R1 集群可派发后补跑 `fvm-relaxation --max-iterations 1200`（先按 F5 加容差出口以把 4.4–8.5 s/迭代降下来），
　 收敛到 <3 % 后把本 scope 提到 L4；
R2 若要立刻让 march 档的 L4 证据入账，把锚点重切为 `CASE-BLE-BLASIUS-PLATE`（成本 0，证据已备）；
R3 平台把 `CASE-FVM-BLASIUS-PLATE` 登记进 `tensor_benchmark_cases`（metric=`|Δc_f|/ref, |ΔH|/ref @Re_x=1e5`，
　 tolerance=`<3%`，ref_source=`Blasius 1908 / Schlichting 7th ed §7 T7.1`），补 F3；
R4 合入主干（F2）——现有 `agent/m16182` 分支未被合并，交付等于停在半路。

---

## 附：工件索引

| 路径 | 内容 |
|---|---|
| `docs/board_audit/m16204/artifacts/audit_probe.py` / `audit_probe.json` | A4 独立 Blasius 重算 + A5 种子独立性 + A6 工件自洽 |
| `docs/board_audit/m16204/artifacts/repro_compare.py` / `repro_compare.json` | A3 逐位比对（3 份工件 `n_diffs=0`）+ 敏感性表 |
| `docs/board_audit/m16204/artifacts/repro-clean/*.json` | 审计方空目录复现产物（未改一行代码） |
| `docs/board_audit/m16204/artifacts/pytest_audit.log` | `7 passed in 12.02 s` |
| `/root/tp-agent/work/m16182/docs/case-blasius-plate/{REPORT.md,artifacts,figures}` | 被审计交付物（commit `cdbb3096`） |
---

## 落表与发布记录

* `tensor_maturity_verdicts` **id 436**：`('TensorFVM', 'm16182 CASE-FVM-BLASIUS-PLATE', 16204, 3, <evidence 21 键>, 'board-audit')`，
  `gate_state='pending'`（maturity_gate 异步裁定）——evidence 全文见 `runs/m16204/verdict_evidence.json`。
* 平台报告 **id 15303**（`/tensor/reports`，mission_id=16204，六要素 + 3 图内嵌）。
* 六要素报告正文 `md_len=4073`、figures=3（落库回读校验）。
* 说明：本工作树为 sparse-checkout（`runs/` 不入索引），故全部审计工件落 `docs/board_audit/m16204/artifacts/`；工作副本另在 `runs/m16204/`。

# 交差审计 m15542 — CASE-FORWARD-STEP (超声速前台阶, TensorFVM)

审计单 m15648 · verdict_by=board-audit · 被审提交 `054dbb84acbebcf7f0aebc7d7e3653d4c2aa8a20 (agent/m15542, NOT merged: ECO TensorFVM HEAD 4704c5a8)` · 非重做，只核验交付。

## ① 案例与交付对象

Woodward & Colella (1984, JCP 54) problem 2：3×1 风洞、台阶 0.2 高、位于 x=0.6、来流 Mach 3、t=4。被审单 m15542 交付『判分面 + 两处缺陷修复 + 120×40 算例』。本审计对象= 该单三件套（run_artifacts / 基准 ref-vs-got / repro_cmd）与平台裁判表 `tensor_benchmark_cases` 的逐项对照。

## ② 怎么核（方法）

1. 只读被审单工作树 `/root/tp-agent/work/m15542`（提交 054dbb84）与其平台 result（fail_stage/headline）与报告 tensor_reports。
2. 三件套存在性：逐文件 stat + sha256；把被审源码/测试/脚本逐字节拷进本审计树 `docs/audit-m15648/reviewed/` 并核 sha256 一致（防事后替换）。
3. 独立复算：用被审代码对**已交付工件**重跑判分器 `--json`，与提交内 `grading.json` 三键逐字段比对。
4. 裁判表：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP'`。
5. 独立复跑算例本身：`scripts/forward_step_study.py --grids 120x40 --t-end 4`（集群 cj1169）——见 ⑥ 未完成项。

## ③ 结果（数字全部来自工件，程序化注入）

交付网格 **120×40**，t=4.0，2834 步，positivity fixes 147，判过 **6/8**；预登记网格 of record [480, 160] 的状态=**never run**（全树无 fields_480x160 类工件）。

| quantity | reference | got | rel. error | tolerance | verdict |
|---|---|---|---|---|---|
| p2/p1 | 10.3333 | 10.4678 | +1.30 % | 0.08 | ✅ |
| rho2/rho1 | 3.85714 | 3.80751 | -1.29 % | 0.08 | ✅ |
| u2/u1 | 0.259259 | 0.229375 | -11.53 % | 0.08 | ❌ |
| M2 | 0.475191 | 0.41501 | -12.66 % | 0.08 | ❌ |
| p_stag/p1 | 12.061 | 11.7241 | -2.79 % | 0.08 | ✅ |
| rho_stag/rho1 | 4.30749 | 4.19911 | -2.52 % | 0.08 | ✅ |
| mass_budget_drift | 0 | -5.04e-16 | -5.04e-16 | 1e-06 | ✅ |
| energy_budget_drift | 0 | 0.00285827 | +0.29 % | 0.01 | ✅ |

未提交的加密网格 **240×80**（`runs/` 未入 git）：判过 **2/8**

| quantity | reference | got | rel. error | tolerance | verdict |
|---|---|---|---|---|---|
| p2/p1 | 10.3333 | 9.37415 | -9.28 % | 0.08 | ❌ |
| rho2/rho1 | 3.85714 | 3.26987 | -15.23 % | 0.08 | ❌ |
| u2/u1 | 0.259259 | 0.197717 | -23.74 % | 0.08 | ❌ |
| M2 | 0.475191 | 0.35032 | -26.28 % | 0.08 | ❌ |
| p_stag/p1 | 12.061 | 11.9051 | -1.29 % | 0.08 | ✅ |
| rho_stag/rho1 | 4.30749 | 4.23635 | -1.65 % | 0.08 | ✅ |
| mass_budget_drift | 0 | 0.0834326 | +8.34 % | 1e-06 | ❌ |
| energy_budget_drift | 0 | -0.0292139 | -2.92 % | 0.01 | ❌ |

## ④ 图

- `figures/fig1_errors_vs_grid.png`：6 个闭式判据的相对误差 vs 网格（±8% 容差带）。
- `figures/fig2_conservation_and_passcount.png`：判过条目数 + 质量/能量守恒恒等式（对数轴）。

## ⑤ 注意点 / 局限

1. **加密反而更差**：u2/u1 从 -11.53% 变 -23.74%，M2 从 -12.66% 变 -26.28%，p2/p1 从 +1.30% 变 -9.28%，rho2/rho1 从 -1.29% 变 -15.23%。激波跳量随加密单调远离闭式锚点 → 『120×40 只是欠分辨』的解释被反证；探测窗（x_s 后 1 个单元/0.02 绝对窗）与二维斜激波/壁面干扰区的关系未定，采样点所在流态是否等于一维正激波态未证。
2. **守恒被破坏**：同一几何同一代码，120×40 质量漂移 5.0e-16（机器零），240×80 骤增至 8.34e-2，能量漂移 2.9e-3 → -2.9e-2。troubled-cell repair 计数反而下降（147 → 19），故『repair 注入质量』只是假设、并未证；但**守恒恒等式在一个加密档上崩掉 15 个数量级**是硬缺陷，须由返修/后续单查明（本审计不重做算法）。
3. **平台裁判表无此锚点行**：`tensor_benchmark_cases` 对本锚点 0 行 → 进度/容差目前只存在于仓内 `grading.json`（自声明面）。本单已按预登记判据补登记一行 pending 行（见下），使下轮有机器裁判面。
4. **repro_cmd 指向的脚本产物布局与判分器不匹配**：声明的 `scripts/repro_fvm_forward_step.py` 写 `<out>/artifacts/forward_step.json` + `forward_step_fields.npz`，而判分器 `load_run` 读 `<out>/fields_<tag>.npz` + `run_<tag>.json` → 照抄 repro_cmd 不能复现被判工件；真链是 `scripts/forward_step_study.py` → 判分器（DESIGN-ADDENDUM 里写对了，grading.json 里写错了）。
5. **交付树与磁盘真相不一致**：`REPORT-v2.md` 在提交树里是 120×40 版本，工作树内另有未提交的 240×80 版本 + `runs/`（`git status` 显示 `?? runs/`）→ 更细网格的自我否定证据没进交付。
6. 分支未合并进 main（ECO TensorFVM HEAD 4704c5a8 不含 forward_step_benchmark.py）；本审计的源码指针因此落在 `reviewed/` 拷贝（sha256 与提交树一致）。

## ⑥ 误差判定（是否 <3%）

- 6 个闭式激波/等熵判据中，交付网格上**没有一个在 3% 内**：最好的是 rho2/rho1 -1.29% 与 p2/p1 +1.30%（仅这两个 <3%），p_stag/p1 -2.79% 亦 <3%；而 u2/u1 -11.53%、M2 -12.66% **远超 3%**，且 240×80 上恶化到 -23.74%/-26.28%。
- 结论：**不满足 <3%，也不满足预登记 8% 容差的全项通过** → 本算例不构成基准验证（L4 不成立）。
- 质量漂移判据 1e-6：120×40 通过（5.0e-16），240×80 失败（8.34e-2）。

## ⑦ 判级：**L3**（不是 L4，也不是 L0-L2）

| 级 | 判据 | 本单证据 | 结论 |
|---|---|---|---|
| L1 | 数学/单元验证 | `tests/test_forward_step_benchmark.py`(6) + `tests/test_repro_fvm_forward_step.py`(8) = 14 passed（本审计复跑 14.89 s）；含 grader 负控（错跳量判红）、采样单元回归、troubled-cell repair、一阶恢复、参考关系对教科书式 | ✅ |
| L2 | 独立功能实现 | `forward_step_benchmark.py` 304 行（注册表/判分/收敛/CLI）+ `forward_step.py` 采样修复 + `solver.py` positivity/troubled-cell repair；入口 `python -m tensorfvm.compressible.forward_step_benchmark --runs <dir>`；运行工件 `fields_120x40.npz`/`run_120x40.json`/`grading.json` 真实非空 | ✅ |
| L3 | 系统集成 | `verification.py:655 leg_forward_step` + `:760 run_all` 注册 + `__main__.py` `--legs forward_step`（分支内；main 未含） | ✅（口径：分支内集成；未合并） |
| L4 | 基准验证 | 网格 of record 480×160 **从未跑出**；预登记 8% 容差 2 项不过（u2/u1、M2）；加密反恶化；判据行未入平台裁判表 | ❌ |
| L5 | 科研工程成果 | 无外部锚点表值/前后对照研究成立（WC1984 只给等值线，锚点是 1D RH 闭式，位置适用性未证） | ❌ |

## ⑧ 处置

**request changes（不消账为基准验证）**：功能实现真实、测试真实、工件真实（L2/L3 成立）；但『基准验证』这一交付目的未达成，且存在守恒崩掉与 repro_cmd 错指两处硬缺陷。建议下一轮返修三件事：(a) 跑 480×160 网格 of record；(b) 查明 240×80 质量漂移 root cause（守恒破损点）；(c) 把探测窗与二维斜激波态的适配性写清或改用 y=0.5 中线探测，并把 repro_cmd 改指 study 脚本。

_本报告由 `docs/audit-m15648/scripts/audit_m15648_extract.py` + `audit_m15648_report.py` 从工件生成，数字零手抄。_

## ⑨ 未跑项 / 复现性（不得读作通过）

- **算例本身已独立复跑（结果与被审单不一致）**：本审计用被审代码跑 `--grids 120x40 --t-end 4 --threads 16`（166.8 s，`repro_run/`），再判分得 **5/8**（被审单 6/8）：nsteps 2834→2829、positivity fixes 147→2、质量漂移 -5.04e-16→+4.56e-4（**突破 1e-6 恒等式**）、能量漂移 2.86e-3→-1.26e-4、p2/p1 10.4678→10.2218、u2/u1 0.2294→0.2219、M2 0.4150→0.4020。同一配置两次运行给出不同答案 → 交付工件**不可逐位复现**（线程数/浮点归约序敏感），判分步骤本身可复现（对同一工件复算逐字段一致）。集群同参作业 cj1169 收尾时仍排队，未采用。
- **480×160 grid of record**：被审单与本审计都未获得（单 socket 需数小时）。
7. **同配置不可复现（新发现 F7）**：见上，5/8 vs 6/8、守恒恒等式 12 个数量级差、跳量三位有效数字差。
- **240×80 质量漂移 root cause**：未定位（本审计不重做算法，只留缺陷记录与 5.04e-16 → 8.34e-2 的实测落差）。
- 平台报告：tensor_reports id=14763（本审计）；被审单报告 id=14644。
- 平台落表：tensor_maturity_verdicts id=351（level=3, verdict_by=board-audit, gate_state=pending）；maturity_gate 的 `judge()` 预演=**L2**（差在 `integration_callers`：main 未合并该分支，grep 无调用方）→ 若守护判 2，属『未合并=未接入主求解器』口径差异，非虚标。
- tensor_benchmark_cases：本审计补登记 CASE-FORWARD-STEP 一行（status=pending，判据/容差取自被审单预登记 DESIGN.md）。

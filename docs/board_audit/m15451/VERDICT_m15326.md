# 交差审计 VERDICT — m15326 CASE-FORWARD-STEP（TensorFVM）

审计单 m15451 · 席位 board-audit · 2026-11-12 · 被审工作树 /root/tp-agent/work/m15326（head=c5da8f7d，PR #429 merged）

## 裁定：L3（系统集成），L4 不达

## ① 三件套核验（逐项实测）

| 件 | 判定 | 证据 |
|---|---|---|
| run_artifacts | ✅ 真存在 | artifacts/forward_step.json 12880 B / forward_step_120x40_t1.json 24110 B / forward_step_fields.npz 29153 B（60×20 rho/u/v/p/solid 场）/ 4 张 PNG；sha256 见 verify_output.json |
| ref vs got | ✅ 零手抄成立，❌ 判据未过 | 报告 8 项判据数值与 forward_step.json 逐项相等（本单程序化比对 ALL_MATCH）；但物理判据 C1–C4 在唯一 t=4 运行（60×20）上 -31.13%/-24.97%/+75.79%/+83.48%（容差 8%）全 FAIL，C5/C6 过（-4.88%/-4.09%），C7/C8 收支闭合 1e-15/1e-16 |
| repro_cmd | ✅ 可执行 | `python scripts/repro_fvm_forward_step.py --out /tmp/audit_fs --grids 60x20 --t-end 0.5 --no-figures` rc=0 出工件（本单实跑）；默认三网格全量约 30 min 未复跑 |

报告六段 ①–⑥ 齐全；平台报告 id=14437 已发布；pytest 8 passed（本单复跑 15.49 s）。

## ② 基准裁判（唯一裁判）

- `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP'` → **0 行**。该锚点从未注册进基准表（FVM 域仅有 REPRO-FVM-SOD1978）。任务榜 goal 的「容差判据 <8%」只存在于 DESIGN.md 预注册，无基准表行可裁。
- 预注册 grid of record = **480×160**（DESIGN.md §4）：**从未运行**。集群作业 cj1080 审计时仍 queued（89 min，节点健康 load≈40/128）；mission fail_stage=`budget_exhausted+forced_finish`，接棒（fetch cj1080）未发生。
- 唯一完成的 t=4 运行是 60×20——比预注册最粗网格 120×40 还粗一档，台阶高仅 4 单元，激波采样窗内仅 1 单元（n_cells_behind=1）。
- 结论：**got 不在容差内**（4/6 物理判据 FAIL，最大 +83.48%）。报告自身已如实声明 honest-FAIL，未粉饰。

## ③ 判级

- L1 ✅ 8 passed；ref 值=闭式 Rankine–Hugoniot/等熵关系（test_reference_matches_textbook_relations 机器验证）
- L2 ✅ 真代码：forward_step.py 467 行 + 主求解器 EulerSolver 新增嵌入式固壁（solver.py +174 行，solid=/project_wall_flux= 构造参数）；消融表证明壁面通量投影必要性（mass_drift -1.59e-2→1.26e-16）
- L3 ✅ 固壁能力是主求解器一等公民（EulerSolver 构造参数），算例模块 forward_step.py 调用之
- L4 ❌ 基准验证不达：基准表无注册行 + grid of record 未跑 + 唯一完成运行 4/6 FAIL
- L5 ❌

## ④ 发现（按严重度）

- **F1（阻断 L4）**：基准 case 未注册 tensor_benchmark_cases（anchor CASE-FORWARD-STEP 缺失）——基准裁判无 canonical 行。
- **F2（阻断 L4）**：预注册 480×160 未跑；cj1080 仍 queued；mission 被强制收尾（budget_exhausted+forced_finish）。
- **F3**：唯一 t=4 运行 4/6 物理判据 FAIL（-31%~+83%）；报告诚实记录，非虚标。
- **F4（证据缺口）**：报告 §附「120x40、t=2 交互诊断 p2/p1 -0.49%」无落盘工件支撑（120x40 工件为 t=1 瞬态，其判据块 p2/p1 -3.61% 过但 C2–C4 仍 FAIL）。
- **F5（小缺陷）**：render_forward_step_report.py 重渲染会丢掉手工追加的「附：fail_stage」段（本单复现 sha 不一致后 git checkout 还原）——honest-FAIL 记录不在渲染器幂等域内。

## ⑤ 未跑检查

- 480×160 全量复跑（约 25 min，且需集群）；CI x86 侧复现（本机 loongarch64）；论文原图密度等值线的定性目视对比（论文无数表，ref 全部由代码现算）。

## ⑥ 处置

verdict 落表 level=3，evidence 全指针见 verify_output.json；maturity_gate 自动裁定（confirmed=消账 / reopened=返修）。
返修路径（若 reopened）：fetch cj1080（或重交）→ 480×160 t=4 → 重渲染 → 注册基准行 → 复裁。

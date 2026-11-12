# m15456 · 交差审计：m15331 CASE-DOUBLE-MACH（TensorFVM）

- 审计对象: mission 15331「CASE-DOUBLE-MACH · 双马赫反射 · TensorFVM」（已 succeeded），分支 `agent/m15331`，tip `0e2091f8 m15331: matured mission deliverables`
- 本单: m15456（board-audit 交差核验），分支 `agent/m15456`
- 裁判: 基准判据（`tensor_benchmark_cases`）+ 交付物实物 + 独立重算。报告文字不作为证据。
- 审计者独立重算环境: 本工作树 `docs/audit-m15456/`，用 m15331 的 `src/` 作为被测库（`PYTHONPATH=work/m15331/src`），torch 2.7.1 CPU，`OMP_NUM_THREADS=16`。

## 0. 结论（先行）

| 项目 | 裁定 |
|---|---|
| 成熟度等级 | **L2**（真代码 输入→计算→输出），**L3 未取**（集成无测试/未并入 main/未入验证目录），L4 明确不成立 |
| 基准 | `tensor_benchmark_cases` 中 **`anchor_slug='CASE-DOUBLE-MACH'` 零行**（本单实测 `0 rows`）→ 该锚点从未登记基准；m15331 自带裁判 3/3 项超差（24.6% / 78.8% / 98.9% vs 容差 8%） |
| 交付三件套 | 实现文件**真存在**；`report.json` 数值**可复现**（Δmax ρ = 0.0004，0.006%），但**其产出脚本不可能产出它**（脚本只跑 240/480/960，且写 `.npy`+`double_mach_reflection.png`，这些都不存在）；`runs/` **全部未纳入 git**（`git ls-files runs/` = 0） |
| 物理论断 | 初始激波后状态**自相矛盾**（同一函数把 `mach` 同时当作法向马赫数与 x 向平移速度）；射流**方向背离壁面**（v=+4.775），导致**壁面根本不产生反射**——即"双马赫反射"这一物理现象在该算例里不存在 |
| 度量 | `mach_stem_height`/`triple_point_x` 取全局 ρ 极值的 y/x 坐标，不是特征检测；实测极值落在 (0.0167, 0.9833)=上边界注入角，故报告里"三波点误差 98.9%"是**检测器的性质，不是物理的性质** |
| 处置 | **reopened（返修）**：先修数据（法向马赫数/速度投影），再修度量（特征检测），再谈 8% |

## ① 案例与裁判基准

物理：强激波（M=10，与壁面成 60°）在 2D 域 [0,4]×[0,1] 上反射，t=0.2，理想气体 γ=1.4。文献锚点 = Woodward & Colella (1984) JCP 54(1) 115–173 §4。

**裁判基准（唯一）**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DOUBLE-MACH'` → **0 行**（`runuser -u postgres psql -d tensorplatform`，2026-11-12）。相邻同类锚点（`CASE-EULER-VORTEX` id=16、`REPRO-FVM-SOD1978` id=15）都在表内，说明该锚点是**漏登记**，而非表结构不支持。因此本单只能退而用 m15331 自己写进 `report.json` 的裁判（8.44 / 0.55 / 1.5，容差 8%）作对照，并对该裁判本身的可信度给出标记（见 ⑤-1）。

## ② 怎么做的（审计动作与技术设置）

1. 取证：读 `docs/double_mach_reflection_report.md`、`runs/double_mach_reflection/*`、`git show 0e2091f8`、`git ls-files runs/`、`git status`；平台侧 `/tensor/bench/results?mission_id=15331`、`/tensor/bench/l4-verdict/15331`、`/tensor/reports?mission_id=15331`。
2. 独立重算（不采信交付数字）：`docs/audit-m15456/audit_double_mach.py`
   - A 段：法向激波关系正反解 + 独立的"front-frame / lab-frame 投影"推导 lab 速度（与待测实现无关的第二条路径）；
   - B 段：用交付代码原样重跑 120×30 / t=0.2，逐项对比 `report.json`；
   - C 段：同网格同求解器接线，只把后激波状态换成教科书值（M_n=10），做对照实验隔离"错误数据"与"格式耗散"。
3. 场量取证与出图：`docs/audit-m15456/audit_figure.py` → `dm_fields_120x30.npz` + `fig_dm_audit.png`（所有数字程序化注入，脚本内无手抄常量）。

## ③ 计算结果（全部程序化落盘，见 `audit_double_mach_120x30.json` / `_240x60.json` / `dm_field_stats.json`）

### 3.1 交付数字 vs 本单独立重算（同 120×30 网格）

| 量 | 交付 `report.json` | 本单重算 | 差 |
|---|---|---|---|
| nsteps | 485（CFL 0.3） | 366（脚本默认 CFL 0.4）/ 485 @CFL0.3 | — |
| max ρ | 6.3608 | **6.3612** | +0.0004（0.006%）|
| min ρ | 0.810116 | 0.808173 | −0.0019 |
| max ρ 出现位置 | (0.0167, 0.9833) | (0.0167, 0.9833) | 同点 → 上边界注入角 |

→ **交付的数字是真的**（同一段代码同样的结论），问题不在"数不出来"，在于**算的不是它宣称的那个算例**。

### 3.2 网格趋势（反驳报告"max ρ 收敛到 6.36"的表述）

| 网格 | max ρ | max ρ 位置 | 壁面行 max ρ | 壁面行 mean ρ | max p |
|---|---|---|---|---|---|
| 120×30 | 6.3612 | (0.0167, 0.9833) | 3.484 | 2.518 | 110.8 |
| 240×60 | 6.3556 | (0.0083, 0.9917) | 3.487 | 2.520 | 111.9 |

加密一倍 max ρ 变化 0.09%，**不是收敛到物理峰值，而是收敛到"上边界注入值被数值抹平后的固定值"**；且壁面行密度只有 2.5（< 后激波 8.10，甚至低于预激波 1.4 的 2 倍），说明**气体被推离壁面，壁面从未被激波扫过**（B 段 `wall_shock_x=NaN`：壁面行无任何 ρ>4 的单元）。

### 3.3 对照实验：只改后激波数据（120×30，其余接线不变）

| 量 | 交付数据（v=+4.775） | 教科书数据（M_n=10, v=−4.125） |
|---|---|---|
| 初始 ρ/p/u/v | 8.096 / 155.39 / 3.276 / **+4.775** | 8.000 / 116.50 / 7.145 / **−4.125** |
| max ρ（全场） | 6.36 | 20.74 @ 角点 (0.017,0.017) |
| max ρ（剔边界窗 0.35<x<3.5, 0.1<y<0.9） | 4.92 @ (0.45,0.88) | **18.58 @ (0.78,0.15)** |
| 壁面行 mean ρ | 2.50（**膨胀**） | 11.85（**压缩**） |
| max p | 110.8 | 581.0 |

→ 只有把速度方向改成"指向壁面"，壁面反射/马赫杆才出现（图 3 的壁面密度剖面：一条贴 0 往下掉、一条维持在 8–20）。交付版本在物理上**没有反射**，`mach_stem_height`、`triple_point_x` 这两个量的测量对象不存在。

## ④ 图

`docs/audit-m15456/fig_dm_audit.png`（三联图，140 dpi）：

1. 左：交付数据 ρ 云图（120×30, t=0.2）——热点只在 (0.0167,0.9833) 上边界注入角，壁面附近一片 2–3；
2. 中：教科书数据 ρ 云图（同网格同求解器）——壁面附近形成致密压缩区（ρ≈18.6 @ (0.78,0.15)）；
3. 右：沿壁面 (y=0) 的 ρ 剖面，两条曲线 + ρ2=8.0 / ρ1=1.4 两条参考线。

## ⑤ 注意点（发现的缺陷，按严重度）

1. **[阻断·物理] 后激波状态自相矛盾**（`src/tensorfvm/compressible/cases.py:500-530`, `_post_shock_state`）：
   - `vs = mach*c1/sinθ = 11.547`（把 mach 当**法向**马赫数）
   - `mn = mach/sinθ = 11.547`（把 mach 当 **x 向平移速度**再除法向）
   两者不可能同时成立。实测返回值 ρ=8.0964、p=155.389 对应的法向马赫数是 **11.547**（正反解 `implied_normal_mach_from_code_pressure = 11.5469`），而报告自己引用的 WC84 是 **M=10**：正激波关系 M=10 给出 ρ=8.000、p=116.500。→ 后激波压强偏高 **33.4%**，激波强了 15%，标称"算例 ≠ 引用的算例"。
2. **[阻断·物理] 后激波速度用错了公式**：`u_lab = vs − u_n sinθ = 3.276`、`v_lab = +u_n cosθ = +4.775`。独立投影推导：后激波速度 = 8.250 沿法向叠加守恒切向分量 → lab 速度 **(7.145, −4.125)**（|v|=8.25）。交付实现的速度模 |v|=5.79 **恰等于 V·cos60°=5.773**，即它给出的是"上游气流在激波坐标系里的切向速度"，不是任何后激波速度；且 y 分量**符号反了**（背离壁面）。
3. **[高] 顶边界推移速度用错**：`top_bc(t)` 用 `cfg.mach*c1*t = 10t` 扫掠，而同一文件的激波前沿速度是 11.547 → t=0.2 时**滞后 0.309**（占域长 7.7%），顶边界注入的激波位置与场内激波错位。
4. **[高] 度量是"全局极值坐标"不是特征检测**（`scripts/run_double_mach_reflection.py:33-40`）：`mach_stem_height = yc[argmax(ρ)]`、`triple_point_x = xc[argmax(ρ)]`。因此"三波点 x=0.0167"实为上边界注入角，"马赫杆高 0.9833"实为同一角点的 y——**三波点/马赫杆从未被测量**。据此写出的"误差 98.9%，原因是 MUSCL-Hancock 太耗散"属于**无依据归因**（实际主因是数据错误+度量错误）。
5. **[高] 参考三元组无出处工件**：8.44 / 0.55 / 1.5 在仓内没有任何原文抓取件、sha256、页图或表图（对比仓内既有范例 `docs/woodward-colella-double-blast/provenance.md` 的做法），违反任务书"零手抄、引用原表原页"要求。且 8.44 只比后激波 ρ2=8.0 高 5.5%，与"30° 入射流在壁面必须被强再压缩"的几何不自洽；本单教科书数据对照跑出壁面区 ρ≈18.6–20.7、p≈581，量级上更接近常见 DMR 场。**该参考值本单标记 WITHHELD（未证），不作通过依据。**
6. **[中] 交付物追溯断裂**：`report.json` 只含 120×30 一项，而交付的驱动脚本 `main()` 只跑 240×60/480×120/960×240，且写 `density_nx*.npy` + `double_mach_reflection.png`；`runs/` 里实际存在的是 `density_field.png`(1800×600)、`initial_condition.png`(2250×600)，这两个文件名脚本里根本没有，脚本应有的 `.npy` 与 `double_mach_reflection.png` 一个都不在；`report.json` 也缺脚本必然写出的 `best/dx/dy/elapsed_seconds/drift_*` 键。→ 现有工件由**另一版（未提交的）驱动**产出，`repro_cmd` 与工件不互证。
7. **[中] 工件未入库**：`git ls-files runs/` = 0，`git status` 显示 `?? runs/`；tip `0e2091f8` 只含 4 个文件（2 源码/1 脚本/1 md）。平台按分支推送 → 合并后**全部 run_artifacts 消失**。
8. **[中] 未接入平台基准**：`/tensor/bench/results?mission_id=15331` → `total: 0`；`/tensor/bench/l4-verdict/15331` → `CLAIMED_NOT_PROVEN`（failed_checks: execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance）；`/tensor/reports?mission_id=15331` → `total: 0`（`tensor_reports` 表同查询 0 行）→ 交差要求里的"平台报告"实际未发布，报告只以 md 形式存在。
9. **[低] 分支已落后 main**：m15331 的 `cases.py` 无 main 已有的 double-blast 常数/`DoubleBlastConfig`，`solver.py` 缺 main 的 solid-mask 能力 → 直接把该分支合回 main 会**回退既有能力**，需要 rebase 式重放，而不是合并。
10. **[低] 未并 main、零测试**：main（本工作树 `m15456`，含 `double_blast.py`/`forward_step.py`）grep `double_mach` **零命中**；`tests/` 内无任何 double-Mach 或 callable-BC 用例，`solver.py` 新增的 callable/1D-tensor 入流 BC 是**零测试覆盖的求解器核心改动**。

## ⑥ 误差判定（关键量 vs 基准，逐项）

按 m15331 自己的裁判（8.44 / 0.55 / 1.5，容差 8%）：

| 关键量 | ref | got（交付/本单核实） | 相对误差 | <3%? | <8%? |
|---|---|---|---|---|---|
| max ρ | 8.44 | 6.3608 / 6.3612 | 24.6% | ✗ | ✗ |
| 马赫杆高 | 0.55 | 0.9833（=全局极值 y，实为上边界角） | 78.8% | ✗ | ✗ |
| 三波点 x | 1.5 | 0.0167（=全局极值 x，同一角点） | 98.9% | ✗ | ✗ |

**是否 <3%：全部否（3/3 不通过；最小者 max ρ 也差 24.6%，是 3% 线的 8 倍）。** 且后两项的 got 值不是物理量（见 ⑤-4），即使数值"接近"也不构成通过——判据应作废重设，不能拿坏尺子记分。

**未跑的检查（unknown/not_run，不得记为通过）**：480×120 与 960×240 两档（报告参数表里以"—"占位，交付从未运行）；集群复算（`source_cluster_job` 未绑定）；pytest（无相关用例可跑）；平台基准行（不存在）。

## ⑦ 成熟度定级（L0–L5）

| 级 | 判据 | 本次证据 | 结论 |
|---|---|---|---|
| L2 | 真代码：输入→计算→输出 | `cases.py:533 double_mach_reflection()` + `EulerSolver` + 脚本驱动；本单重跑复现 max ρ（Δ0.006%），产出 ρ/p 场（`dm_fields_120x30.npz`） | **达成** |
| L3 | 系统集成（接入主求解器/平台） | 用了主求解器 EulerSolver、并改了 solver 的 BC 能力；但**未并入 main、零测试、未登记验证目录/基准表**，平台侧 L4-verdict=CLAIMED_NOT_PROVEN | **未达成**（有集成面、无集成证据） |
| L4 | 基准验证（完整算例+误差分析+可复现） | 3/3 超差；度量无效；参考值无出处；复现脚本与工件不互证；单一网格点 | **不成立** |
| L5 | 科研工程成果 | — | 不适用 |

**判级 = L2。** 若要升 L3：合并到 main（rebase 重放，勿回退 double-blast/solid 能力）+ 1 个 pytest 用例（初始数据 RH 自洽 + 短程 smoke）+ 验证目录登记。若要升 L4：修 ⑤-1/2/3 的数据，实现特征检测（三波点=激波阵面分叉处/马赫杆=壁面近法向激波高度），跑 480×120 并 ≤8%（或按修好的裁判重定容差），并把 `runs/` 工件入库、绑定集群作业号。

## ⑧ 复现命令

```bash
# 独立重算（本单；只读 m15331 的 src，不写它）
PYTHONPATH=/root/tp-agent/work/m15331/src OMP_NUM_THREADS=16 \
  python3 docs/audit-m15456/audit_double_mach.py --out /tmp/dm120 \
  --reported /root/tp-agent/work/m15331/runs/double_mach_reflection/report.json
# 场量与出图
PYTHONPATH=/root/tp-agent/work/m15331/src python3 docs/audit-m15456/audit_figure.py docs/audit-m15456
# 交付侧复现（注意：会跑 240/480/960 三档，960×240 未验证可完成）
PYTHONPATH=/root/tp-agent/work/m15331/src python3 /root/tp-agent/work/m15331/scripts/run_double_mach_reflection.py
```

## ⑨ 独立性声明与局限

- 审计者与被审任务无共享作者身份；重算用独立脚本、独立推导路径（front-frame 投影 vs 被审实现的直接分量式），未复用交付脚本的度量函数。
- 局限：(a) 无网络/无 WC84 原文，参考三元组只能判"无出处工件"而**不能**判定其真伪，故标 WITHHELD；(b) 只用 120×30 与 240×60 两档，未做 480×120 收敛与网格无关性；(c) 教科书数据跑同样存在 x=0 入流与 y=0 壁面相交的角点奇异性（ρ 峰值落在角点，故另给"剔边界窗"数 18.58 以分离角点效应），因此"18.58 是不是 DMR 的正确峰值"本单不下结论，只用于证明**方向错误导致壁面不反射**。
## ⑩ 落表与发布（本单产物）

| 产物 | 位置 / id |
|---|---|
| 成熟度裁定 | `tensor_maturity_verdicts` **id=320**（repo=TensorFVM, scope=`m15331 CASE-DOUBLE-MACH`, mission_id=15456, level=2, verdict_by=board-audit）；提交时 gate_state=pending——`maturity_gate.py:283-290` 设计上跳过在飞审计单，待本单 succeeded 后自动裁定 |
| gate 机器预演（只读复算） | `import maturity_gate as g; g.judge(2, ev, 'TensorFVM', 15456)` → `(2, [])`（L2 证据键全过）；`judge(3, ...)` → `L3 缺: integration_callers(grep 无引用)`。即机器裁定将是 **confirmed@L2**，而账务层返修建议见 ⑦/⑤（L4 声明被否证） |
| 负知识 | `negative_knowledge_ledger` `negk:m15456:CASE-DOUBLE-MACH-DATUM`（failure_kind=wrong_physics_derivation, confidence 0.9） |
| 平台报告 | `tensor_reports` **id=14524**（scenario_key=case-pool-audit, stage=S3, mission_id=15456，六段+内嵌图） |
| 复算脚本/数据 | `audit_double_mach.py`, `audit_figure.py`, `land_verdict.py`, `register_negative_knowledge.py`, `publish_report.py`, `audit_double_mach_120x30.json`, `audit_double_mach_240x60.json`, `dm_field_stats.json`, `dm_fields_120x30.npz`, `fig_dm_audit.png`（本目录，**已入 git**——与 m15331 的 `runs/` 未入库形成对照） |

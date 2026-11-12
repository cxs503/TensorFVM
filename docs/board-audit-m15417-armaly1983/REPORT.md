# 交差审计 — m15386 `REPRO-FVM-ARMALY1983` (TensorFVM)

**审计单**: m15417（交差核验单，非重做） · **被审对象**: m15386 `repro-fvm-armaly1983-r2`（平台状态 succeeded）
**锚点**: Armaly, Durst, Pereira & Schoenung (1983), *J. Fluid Mech.* **127**, 473–496
**裁判**: `tensor_benchmark_cases`（基准判据唯一裁判）＋ 存档工件数值 ＋ 复跑
**判级结论**: **L3**（系统集成）。L4 差两腿：网格收敛证据缺失、第三预声明判据 FAIL。详见 §7。

---

## 1. 审计独立性局限（先说不能保证什么）

1. 本次审计与 m15213/m15386 **同一生态、同一工作树族**（`/root/tp-agent/work/m15386`），非外部独立复算；独立性止于「重跑同一 CLI + 独立读工件 + 独立重算误差」。
2. 权威实验值（3.0 S / 7.9 S）**未从论文 PDF 重新数字化**（离线下无原文），采用复现文献引用的二手值 ±0.2 S —— 这是被审对象报告 §5.6 已声明的同一局限，非本次审计新增。
3. `tensor_benchmark_cases` 中该锚点**零行**（§3），因此「机器容差裁判」在本锚点上**不存在**，容差只能取自任务档案自声明（oss_solutions 登记 15 % / 15 % / 5 %）。
4. 全算例复跑（单点 274–577 s > 控制面 shell 120 s 上限）：已递交集群 `cj1113`（审计时仍 queued，队列前置 318 单），并另起本机 detached 重跑；**审计收尾时两者均未产出数字**，故「存档数字可逐位复现」一条判 **WITHHELD（未验证）**，不判 PASS。

## 2. 取证：三件套是否真存在（逐项）

| 件 | 指针 | 核验结果 |
|---|---|---|
| 实现 | `src/tensorfvm/inc_bench/armaly.py`（锚点表 L58-70、`profile_development` L99、`ReproductionRun` L137） | **真代码**：锚点强校验（`experiment_point` 越界即 raise，禁插值）、`L_r` 取壁面摩擦 `C_f(x)` 零点线性插值、发展度=同质量通量 Poiseuille 对比。无随机、无硬编码结果 |
| 实现（复用） | `src/tensorfvm/inc_bench/cases.py:backward_step`、`solver.py:SteadyFlow`（错列 MAC SIMPLE） | 既有主求解器，本任务未 fork 物理 |
| 入口 | `scripts/repro_armaly1983_bfs.py`（`main` L59） | 可执行：pytest 内 smoke 跑通；CLI 参数含 `--re/--levels/--downstream/--out` |
| 运行工件 | `docs/repro-armaly1983-bfs/artifacts/armaly1983_re{100,389}_coarse_ld12.json` ＋ `..._matrix_*.json` | **存在且非空**（含场量 7203 流体胞、`profile_stations`、`solver_options`、`seconds=274/576.6`） |
| 测试 | `tests/test_armaly1983_reproduction.py` 5 例 | 本次在 m15417 工作树**重跑：5 passed in 20.83 s** |
| 集成调用方 | `src/tensorfvm/internal_flow/ledger_reference.py:80`、`ledger_report.py:201`、`reference.py:81`、`internal_flow/maturation.py:271`、`scripts/run_incompressible_benchmarks.py:88` | 真引用（`grep -rl armaly_reattachment` 命中 3 文件） |

**唯一实质性漏洞：round-2 交付物未入库。** `docs/repro-armaly1983-bfs/REPORT_ROUND2.md`、`research/`、`convergence/`、
`scripts/convergence_armaly1983.py`、`tests/test_armaly1983_extended.py` 在 m15386 工作树中全为 **untracked（git status `??`）**，
main 分支（ECO 检出 `653e63a6` ＝ m15417 工作树）**不存在**这些文件。即：m15386 声称的「工具链扩展 + 扩展测试」没有进入仓库。

## 3. 基准裁判核查（`tensor_benchmark_cases`）

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='REPRO-FVM-ARMALY1983';   -- 0 rows
```

全表 15 行（含 `REPRO-FVM-SOD1978`、`XVAL-FEM-TPB1976` 等），**本锚点未登记**。后果：
- 该锚点无法被机器按容差裁定，只能由审计人按任务档案自声明容差人工核对（本报告 §4）；
- 生态层面这是一条**判据缺口**（应为每个 REPRO 锚点登记 ref/metric/tolerance 行），建议补登记而不是降级交付物。

## 4. 数值核对（ref vs got，逐项）

| # | 判据（来源=oss_solutions 登记） | ref | got（工件直读） | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|---|
| 1 | `L_r/S` @ `Re_Dh=100` | 3.0 | 2.644902 | **+11.84 %** | 15 % | **PASS** |
| 2 | `L_r/S` @ `Re_Dh=389` | 7.9 | 7.085588 | **−10.31 %** | 15 % | **PASS** |
| 3 | 出口剖面发展度 @ `Re=389`（x/S=11.97） | 0（平面 Poiseuille） | max_rel 10.908 % | 10.91 pp | 5 % | **FAIL** |
| 3′ | 出口剖面发展度 @ `Re=100` | 0 | max_rel 0.0985 % | 0.098 pp | 5 % | PASS |
| 4 | 网格扩展比 | 1.9423 | 1.94118 | 0.058 % | —（几何先验证） | PASS |

补充（非判据，供判级用）：`Re=389` 中间站位 max_rel = 47.5 % (x/S=5.97) / 22.5 % (x/S=8.97) —— 仍在回流区
（`L_r/S=7.09` 之后才开始恢复），说明剖面门 FAIL 的根因是**域长 12 S < 下游入口段 L_e≈19 S**，与报告归因一致，非求解器造假。

**是否 < 3 %？——明确回答：否。** 两个 reattachment 判据误差 11.84 % / 10.31 %，均只过 15 % 门；无 3 % 级声明。
被审报告本身也明确写了「There is no 3 % claim in this report」，无夸大。

## 5. 报告六段核查（`docs/repro-armaly1983-bfs/REPORT.md`，round-1 主报告）

①案例介绍（几何/参数/网格表）✔ ②方法流程（SIMPLE 设置、`L_r` 与剖面度量定义、CLI）✔ ③结果数字表（程序化取自 JSON）✔
④图 5 张 png（场量/剖面/摩擦/发展/reattachment）✔ ⑤注意点（§5 六条：压力 CG 预算、`reference.ARMALY_1983_REATTACHMENT` 归一化错位、
2D/3D 边界、域长属判据、锚点二手）✔ ⑥误差判定表含「是否 <3 %」逐项显式回答 ✔。
**但 round-2 `REPORT_ROUND2.md` 自述「待集群结果」，§5 待办 5 项一条未勾**，且**未发 `tensor_reports`**（m15386 在平台报告表 0 行；
全表 9611 行，同锚点 m15213 有报告 id=14339）→ 报告纪律未落实。

## 6. 缺陷清单（按严重度）

| ID | 严重度 | 事实 | 证据 |
|---|---|---|---|
| D1 | **blocking（判级）** | round-2 声明的网格加密（fine S/49）/长域 40 S 从未跑出：`convergence/convergence_analysis.json` 两级均为 `"status": "insufficient_data"`（单网格级） | 工件文件；`cj1103`(7 shards)、`cj1105`(11 shards) 在 `cluster_jobs` 中至今 `state=queued`，`finished_at` 为空 |
| D2 | high | 第三预声明判据 FAIL（10.91 % > 5 %），round-2 未修（域长未延长） | §4 行 3 |
| D3 | high | round-2 交付物未提交（untracked），main 分支不存在 | `git status`（m15386 工作树）；m15417/ECO 检缺失 |
| D4 | medium | `tensor_benchmark_cases` 本锚点 0 行 → 无机器裁判 | §3 SQL |
| D5 | medium | m15386 平台状态 succeeded 但无 `tensor_reports` 行，且自述「本轮未完成」→ 结单语义与内容不符 | `tensor_reports` 查询；REPORT_ROUND2 §6 |
| D6 | medium（继承，非本轮引入） | `inc_bench/reference.py:ARMALY_1983_REATTACHMENT` 单位错位（文档称 step heights，数值实为 `L_r/H_out`，Re=100 处 3.9 vs 实验 3.0） | 被审报告 §5.3 已自曝；本次未复算修正 |
| D7 | low | 存档工件 `cells=8019` 与报告表述 `7203 fluid cells` 并存，易被误读为矛盾（实为总胞/流体胞） | 工件字段 `cells`/`fluid_cells` |

## 7. 判级推导（L0–L5）

| 级 | 判据（maturity_gate JUDGE_SCHEME） | 本锚点实证 | 结论 |
|---|---|---|---|
| L1 | tests 非空 | 5 例，本次重跑 5 passed / 20.83 s | 过 |
| L2 | 非占位实现 + 入口 + 运行工件 | `armaly.py` 真算（摩擦零点插值/同通量 Poiseuille），CLI 出 JSON 工件 | 过 |
| L3 | ＋主求解器/平台侧调用方 | `inc_bench` 主求解器被直接驱动；`internal_flow/*`、`run_incompressible_benchmarks.py` 真引用 | **过（本次判定级别）** |
| L4 | ＋benchmark ref-vs-got 误差 + 可复现命令 | 数值对照齐全（§4），但**可复现性本条只证到 smoke 级**（全算例重跑未出数，判 WITHHELD）；**且网格收敛证据缺失**（D1）使 11.8 %/10.3 % 的离散误差不可分解，第三判据 FAIL（D2） | **不过** |
| L5 | ＋external_anchor 文献锚 | 有（JFM 127 引文、±0.2 S 二手数字化） | 停在 L4 门 |

**为什么判 L3 而非 L4**：L4 的功能实质是「完整算例 + 误差分析 + 可复现」。本例三点中「完整算例」只在**单一 coarse 网格（S/16）**
成立，「误差分析」缺离散收敛腿（唯一能说明 11 % 是耗散还是模型误差的证据，恰好是本轮该产出而未产出的），
「可复现」本次只验到入口级。round-2 的实质增量=调研文档 + 收敛脚本（且未入库、其输出为 `insufficient_data`），
按「附百页报告不改变级别、级别=功能实质」应记 **L1 增量挂在既有 L3 能力上**。
反面若判 L4，则等于用 m15213 的 round-1 工件替 m15386 的未完成 round-2 消账（D1/D3/D5 同时被抹平）——故不判。

## 8. 未跑 / 未验证（不得当 PASS 读）

- `unknown`：全算例 CLI 重跑（`cj1113` queued；本机 detached 重跑收尾时未出数）→ 存档数字的逐位可复现性**未验证**。
- `not_run`：fine 网格（S/49）与 40 S 长域算例（资源：单点 577 s@8 k cells，40 S 域为小时级；且队列 318 单）。
- `not_run`：`C_f(x)` 对实验曲线的对比（无离线数字化，被审报告亦判 WITHHELD）。
- `not_run`：`reference.ARMALY_1983_REATTACHMENT` 单位错位（D6）的修正与回归（另单）。
- `not_run`：`Re ≥ 400` 行为（实验三维，二维求解器不适用）。

## 9. 处置建议（disposition）

- **判级 L3 / 确认该锚点已具备「可运行 + 有据」的复现能力**，但 **L4 记账不得放行**；下一步单点很小：
  ① 跑 `medium` + `40 S`（`bash scripts/_cluster_armaly.sh 3..7`）→ 填 `convergence_analysis.json`；
  ② 把 round-2 未入库文件提交进 main；③ 用 `tensor_benchmark_cases` 登记该锚点三行容差（D4）。
- 修 D6（`reference.py` 归一化错位）宜并入上述同单，避免二次踩坑。

## 10. 本次审计自身的验证动作（可复核）

```
python -m pytest tests/test_armaly1983_reproduction.py -q      # 5 passed in 20.83s (m15417 工作树)
python docs/board-audit-m15417-armaly1983/make_figures.py      # 图与 figure_data.json 全部由存档工件程序化导出
psql -c "SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-FVM-ARMALY1983'"   # 0 rows
psql -c "SELECT id,state,finished_at FROM cluster_jobs WHERE id IN (1103,1105,1017,1113)"   # 1103/1105 queued
psql -c "SELECT count(*) FROM tensor_reports WHERE mission_id=15386"                        # 0
```
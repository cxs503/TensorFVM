# CASE-FVM-CAVITY-RE1000 —— 机器门与登记裁判行对齐（GATE）

本文件记录对 `CASE-FVM-CAVITY-RE1000`（＝`CASE-CAVITY-RE1000-FVM`，二维顶盖驱动方腔 Re=1000，
Ghia et al. 1982）**验收门**的收紧与取证，对应 board audit `m20230` 的返修项
F1（锚点错配）/ F2（机检门比登记裁判行松）/ F3（机检门未从原始场重算）/ F5（插值方案敏感性）。

## 1. 问题（返修前）

| 项 | 返修前状态 | 后果 |
|---|---|---|
| 容差 | `CRITERIA` 全部 5 %（`CASE_POOL_TOLERANCE`），而 `tensor_benchmark_cases` 登记行为 3 % + 涡心 `0.01 L @ n>=128` | 解退化到 3 %~5 % 之间时机检门仍绿（放水） |
| 锚点 | 登记行只挂在 `CASE-CAVITY-RE1000-FVM`；总账锚点 `CASE-FVM-CAVITY-RE1000` / 算例池码 `FVM-CAVITY-RE1000` 无行 | 按锚点解析裁判行得空 → 机械 gate WITHHELD |
| 涡心判据 | 直接采信求解器自身的极值搜索 | 无第二估计量交叉核对 |

## 2. 改了什么（最小改动，全部可机检）

1. `src/tensorfvm/case_cavity_re1000.py`
   - 新增 `REGISTERED_TOLERANCE = 0.03`、`VORTEX_TOLERANCE_L = 0.01`、`JUDGE_GRID = 128`；
     `C1..C5` 的 `threshold` 改为登记值，并带 `graded_from_n = 128`。
   - `evaluate()` 改为**分辨率感知**：`n >= 128` 用登记带（`band="registered"`），
     `n < 128` 仍用 5 % 开发带（`band="diagnostic"`）。粗网格不再继承验收声明，但也不被藏起来。
   - 新增 `ANCHOR_ALIASES` / `resolve_anchor()`：三种拼写归一到 `CASE-FVM-CAVITY-RE1000`。
2. `docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py`
   —— **不重跑求解器**，只从 `runs/fields.npz` 的收敛 MAC 场重算全部计分量（Hermite 与 PCHIP 两种
   插值各算一遍）、用独立积分的流函数极值重定位主涡心、把结果写回 `summary.json` / `benchmark.json`
   并另存 `runs/gate_registered.json`。存档剖面与场重算差 ≤ 1e-12（脚本内断言）。
3. `docs/CASE-FVM-CAVITY-RE1000/register_benchmark_case.sql`
   —— 在总账锚点下登记裁判行（幂等，`WHERE NOT EXISTS`）。已在平台库执行：`id = 146`，
   `anchor_slug = CASE-FVM-CAVITY-RE1000`，`status = passed`（与 `id = 18` 同度量同容差）。
4. `tests/test_case_cavity_re1000_gate_registered.py`
   —— 11 条断言型测试（含负对照，见 §4）。
5. `docs/CASE-FVM-CAVITY-RE1000/make_report.py`
   —— 离散不确定度小节改为从 `runs/uncertainty.json` 程序化注入（原先该节为人工段落，
   重生成报告时会丢失）。

## 3. 结果（n = 128 判级网格，全部由 `runs/gate_registered.json` 注入）

| 判据 | 实测 | 登记容差 | 结论 |
|---|---|---|---|
| C1 中心线 `max|Δu|/U` | 4.307e-03 | 0.03 | ✓ |
| C2 中心线 `max|Δv|/U` | 1.118e-02 | 0.03 | ✓ |
| C3 `u_min` 相对误差 | 0.647 % | 3 % | ✓ |
| C4 `v_max` 相对误差 | 0.779 % | 3 % | ✓ |
| C5 主涡心位移 | 0.00396 L | 0.01 L | ✓ |
| C6 `max|div u|` | 1.208e-13 | 1e-10 | ✓ |
| C7 定常残差 | 9.888e-08 | 1e-06 | ✓ |

**是否 < 3%：是**（最严的量是 C2 的 1.118 %，余量 2.7 倍；C1 仅 0.43 %）。三套网格的分带：
n = 32 → 诊断带 FAIL（17.36 % / 14.96 %），n = 64 → 诊断带 PASS（2.80 % / 2.34 %），n = 128 → 登记带 PASS。

插值方案敏感性（audit F5，本单量化）：n = 128 时 Hermite 与 PCHIP 的中心线指标差
`max|Δu|` 1.4e-07 U、`max|Δv|` 3.6e-05 U，极值差 ≤ 1.7e-04 —— 判定不随方案翻转；
n = 32 时同一差为 0.124 U，故粗网格只作诊断、不承担验收声明。

## 4. 负对照（门确实会拒）

| 注入的缺陷 | 判级网格 n=128 | 粗网格 n=64（诊断带） |
|---|---|---|
| 中心线误差 0.04 U（落在 3 %~5 % 窗口） | **FAIL** | PASS（带差所致） |
| 涡心位移 0.02 L（落在 0.01~0.05 L 窗口） | **FAIL** | PASS |

即：返修前会被判 PASS 的这一窗口，现在在判级网格上必然 FAIL；测试不是"跑通即绿"，而是断言这条不变量。

## 5. 复现

```bash
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py   # 从存档场重算并写回
PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000_gate_registered.py tests/test_case_cavity_re1000.py -q
runuser -u postgres -- psql -d tensorplatform -f docs/CASE-FVM-CAVITY-RE1000/register_benchmark_case.sql
```

## 6. 边界与未做项（not_run，不记 pass）

* 未重跑求解器（n = 128 约 1300 s）；本单只重算**度量**，不重算**解**。存档场与 `summary.json`
  的逐点一致性由脚本断言，但它不能替代一次冷启动复现（audit `cj2580` 已做过逐比特复现）。
* 参考地板未在本单核算：本单只对齐「门口径」；把 `max|Δu|/U` 拆成离散误差 + 参考偏移（并给出无参考口径下的判词）见 `REFERENCE_FLOOR.md`（后续单补）。
* 时间步（Courant）方向：`dt_study.py` + `runs/dt/` 已给出同网格（n = 128）、同初值、同收敛判据下 3 个 Courant 数的对照，结论与数字见 `UNCERTAINTY.md` §4（图 UQ-2）；区间外（更大 Courant 数）仍无证据，不记 pass。
* 平台 `tensorbench_results` 未写：本单只在 `tensor_benchmark_cases` 登记判据行，
  不主张"平台级集成"。
* `tests/test_cavity_anchor.py::test_superseded_digits_survive_only_in_documented_history`
  在本工作树 HEAD 上已失败（命中 `docs/case-fvm-channel-kmm180-sublayer/REPORT.md`，
  与本单无关，未修以免越域改他人证据）。
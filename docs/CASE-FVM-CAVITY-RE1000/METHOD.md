# CASE-FVM-CAVITY-RE1000 — 方法选型与复用记录（第 2 轮调研→第 3 轮落地）

本单（m20283）是锚点 `CASE-FVM-CAVITY-RE1000` 的返修轮：前轮 m19978 honest-FAIL（只跑粗网格/未收敛），
m20000 已产出 n = 32/64/128 的正确数字但**运行工件未入库**（evidence 只有报告文本），机器复核判 L1。
本文件记录本轮调研结论、选型与「站在已有实现上」的复用关系；数字证据见 `REPORT.md` 与 `runs/`。

## 1. 锚点要求

| 项 | 值 |
|---|---|
| 算例 | 二维顶盖驱动方腔，单位方腔，顶盖切向速度 U = 1 |
| Re | 1000（ν = 1e-3） |
| 基准 | Ghia, Ghia & Shin (1982), Table I/II + 主涡中心 (0.5313, 0.5625) L |
| 容差 | 算例池 `CASE-FVM-CAVITY-RE1000`：< 5% |
| 额外要求 | ≥ 2 个分辨率 + 收敛证据 + 可复现命令 |

## 2. 调研（① 开源横向 / ② 方法谱系）

* **① 开源横向**：`oss_inventory` / `oss_solutions` 对本 agent 角色是 `permission denied`；
  节点 `~/oss-cache` 里可用的现成实现是 LBM(`Maarten-vd-Sande_lbm`, `pylbm`) 与 SPH(`GraSPH`)，
  与交错网格 FVM 投影法不同族，**没有可直接迁移的 Re = 1000 方腔 FVM 实现**（沿用 m15458 的调研结论，
  `docs/CASE-CAVITY-RE1000-FVM/method-selection.md` §2）。
* **② 方法谱系**（教科书/文献标准做法）：
  1. SIMPLE/SIMPLEC 同位网格（Patankar 1980）——需 Rhie–Chow 动量插值避免棋盘压力，收敛依赖松弛因子；
  2. 涡量–流函数 + 多重网格（Ghia et al. 1982 原文即此法）——无压力场，但边界涡量需单侧差分、
     三维/自由面不可扩展；
  3. 交错 MAC + 投影法（Chorin 1968; Temam 1969），压力 Poisson 用快速变换直接解
     （闭腔纯 Neumann → DCT-II 对角化，Schumann/Ozawa 式）；对流用守恒通量 + TVD 限制器。
* **③ 选型**：第 3 条。理由：闭腔是最难的纯 Neumann 压力问题，直接解 Poisson 使离散相容条件
  `Σ div u* = 0` 可精确满足（不需压力参考点/迭代）；交错网格无棋盘模态；TVD van Leer 在
  Re_h = O(10) 的顶盖剪切层有界。代价是非定常推进到定常（显式格式步数多），因此保留**网格续算**
  只降成本（`runs/control/` 的冷启动对照证明续算不改变不动点）。
* **④ 复用（站在已有实现上）**：本轮不新写求解器，直接复用仓内生产模块
  `src/tensorfvm/lid_cavity.py`（m15458 起入库，ladder C2 已在用）+ 参考表
  `src/tensorfvm/lid_cavity_reference.py`（Table I/II 转写 + `self_check()` + 争议点登记）。
  本轮新增的只是**算例层与证据链**：`src/tensorfvm/case_cavity_re1000.py`（7 条命名判据 + 工件 schema）、
  `docs/CASE-FVM-CAVITY-RE1000/run_case.py`（入口，写 `runs/`）、`scripts/case_cavity_re1000.py`（单分辨率 CLI）、
  `tests/test_case_cavity_re1000.py`（机检门：从 `runs/fields.npz` 重算剖面与判据）。

## 3. 与主流水线的集成

* 工件 schema 与 `tensorfvm.verification.ladder.lid_cavity_verdict` 消费的键
  （`reynolds` / `n` / `steps` / `converged` / `residual` / `max_divergence` / `wall_seconds` /
  `u.max_abs_error` / `v.max_abs_error` / `rms_error`）一致，Ladder 的 C2 判据直接读本包 `runs/`；
  `tests/test_case_cavity_re1000.py::test_ladder_consumes_the_bundle` 就是这条集成调用的机检门。
* `scripts/case_cavity_re1000.py` 是本算例池条目的 CLI 入口（与 `scripts/case_cavity_re100.py` 同构）。

## 3b. 续算对照（counter-check）的有效性前提

「延拓只改成本不改解」这条结论的证据是 `runs/summary.json -> continuation_check`：
n = 64 由 n = 32 双线性延拓（59750 步 / 331 s）与同一网格冷启动（58750 步 / 384 s）收敛到
同一不动点（`max|Δu| = 3.34e-08`、`max|Δv| = 2.88e-08`）。

这条对照**只在对照网格本轮确实被延拓过时**才构成检验。若把 `--cold-control` 指向本轮由静止
启动的网格（例如最粗的 n = 32），两次运行就是同一次计算，比较结果恒为 `max|Δu| = 0`、
`speedup` = 墙钟噪声，却照样会被打印成 `same_steady_state: true` —— 一个恒真的「检验」。
本轮修复：

* `run_case.py` 的 `--cold-control` 默认值改为「本轮实际被延拓过的最粗网格」
  （`select_control_grid` / `warm_started_grids` 由工件里的 `initial_condition` 判定，
  不再硬编码 32）；
* 指向由静止启动的网格时，`build_continuation_check` 写 `kind = "repeat_run"`、
  `same_steady_state = null` 并注明「只是确定性重复、不足以判定延拓是否保持不动点」，
  同时 `make_report.py` 按这一事实措辞，不再把退化对照渲染成证据；
* `repro.sh` 的全量复现命令此前写的是 `--cold-control 32`，与 `runs/summary.json` 记录的
  出处（`--cold-control 64`）不一致，即唯一一条文档化的全量复现命令**无法**复现已公布的
  出处，且会把 `runs/summary.json` 的对照覆盖成退化版本；现改为 64；
* `tests/test_case_cavity_re1000_continuation_control.py` 把上述不变量机检化：抽取
  `repro.sh` 的战役参数与 `summary.json` 的 `command` 逐项比对、从存档工件重算
  `continuation_check` 的全部数字、并断言被引用的对照网格本轮是延拓启动的。

同一轮还发现报告本身不可复现：`REPORT.md` ⑤ 里那条「「吻合」不等于「精度」」（3 % 带宽有 61 %
被参考解吃掉）是 m21931 手工折进报告的，`make_report.py` 里并没有产出它的代码 —— 也就是说
跑一次文档化的 `make_report.py` 会**无声删掉**这条最关键的诚实声明。现在该条由
`runs/decomposition.json` 程序化注入（`_decomposition()`；数字零手抄，与 `REFERENCE_FLOOR.md`
同源），并且 `make_report.py` 覆盖前会做一次 `dropped_lines()` 对账：若新渲染会丢掉旧文件里
它不产出的整句，则拒绝写盘并报出被丢的句子。因此当前 `REPORT.md` 与
`render(summary, bench, fields)` 的输出在除时间戳/复现块之外逐行一致
（机检门：`test_rendering_reproduces_the_committed_report`）。

## 4. 检索 tk / 负知识台账的说明（诚实记录）

任务要求先查 `knowledge_base`（ktype='capability', domain='TensorFVM'）与
`negative_knowledge_ledger`（keywords='TensorFVM'）。本 agent 的数据库角色对这两张表返回
`permission denied for table ...`，`oss_inventory`/`oss_solutions` 同样不可读，因此**未能**引用 KB 条目
（`cited_kb` 为空）；替代做法是直接检索仓内既有实现与文档（`grep`/`docs/`），
即 `src/tensorfvm/lid_cavity.py`、`docs/CASE-CAVITY-RE1000-FVM/method-selection.md`、
`docs/verification-lid-cavity-ladder/`。前轮的失败模式（预算耗尽于写 runner、未留工件）已作为负知识在
本单开头被规避：先复用既有求解器，再把预算花在**跑出并入库工件**上。

## 5. 局限

* 二维层流定常；不覆盖三维、非定常脉动、Re ≫ 1000 的湍流方腔。
* 显式推进到定常，单核 ~30 min；更大 n（256）成本按 n² 增长，未在本单尝试。
* 判据只针对中心线剖面与主涡中心（Ghia 公布量），壁面剪应力/角涡尺寸未对标（原文未列表）。
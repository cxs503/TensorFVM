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
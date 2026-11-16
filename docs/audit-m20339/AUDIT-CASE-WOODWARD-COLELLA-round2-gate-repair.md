# AUDIT round-2 — CASE-WOODWARD-COLELLA 证据包门禁修复（m20339 重开第 1 轮）

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用，feature_ledger `open`）
**验收（总账）**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`
**上一轮**：PR #831 已由 owner 于 2026-11-05 裁定关闭；其内容经 #664 落到 main
（`docs/audit-m20339/AUDIT-CASE-WOODWARD-COLELLA-adversarial-review.md` + `coverage/` 8 文件 + 13 门）
**本轮性质**：对**已入 main 的审计结论与其自带的防回归门禁**做第二轮对抗性复核——不重开数值结论，
只查「声称 vs 现状」是否一致。**不重跑求解器，不改写任何历史数值证据。**
**范围纪律**：本轮只动 `docs/woodward-colella-double-blast/coverage/`（审计工具 + 见证 JSON）与
`tests/test_compressible_wc_coverage.py`，**未触碰 `src/tensorfvm/**` 任何核心模块**，与其它在开 PR 无热点重叠。

---

## 0. 裁决（先给结论）

**request changes → 已在本 PR 内完成修复；修复后：approve。**

- 第一轮的科学结论（`<10%` 在 δ∈[−0.022,+0.044] 全带成立、浅槽排除有据、盲区 6.44% 质量）
  **本轮复核维持不变**：所有数值证据用原工具重算一致，一个数都没改。
- 但第一轮的**门禁在 main 上是红的**（`11 passed / 2 failed`，不是文档 §6 声称的
  `60 passed, 1 skipped`）。root cause 不是数值污染，而是**审计工具把自己的一类记录误判为「复现失败」**，
  外加**见证文件（manifest）在后续任务落地后失效**。
- 因此第一轮的 §3.5「16/16 复现」与 §6「60 passed / 1 skipped」在**当前 main 上不可复现**，
  属**声称漂移（claim drift）**，已在原文件加注，不由本轮覆盖历史。
- 修复后：本锚点 provenance/coverage 门禁 `18 passed`（修前 `11 passed / 2 failed`）。
- **诚实记账**：锚点全套仍有 **2 例红**，但**与本轮改动无关、且在 main 上已存在**（§5.1 给出证据），
  本轮按「单 PR 单功能点」纪律**不修**，登记为后续单元。

---

## 1. 复核范围与独立性限制

| 项 | 说明 |
|---|---|
| 复核对象 | main `746adaf6` 上已合入的 m20339 全部交付物（经 #664） |
| 方法 | 独立重跑证据脚本 + 逐条核对文档数字 vs 重算值 + 检查门禁是否真断言 |
| 独立性限制 1 | 与上一轮同属一个智能体家族；缓解手段是**换证据种类**（不是换人）：本轮证据全部是「git 历史 + 记录分类 + 反例注入」，不用任何第一轮用过的量 |
| 独立性限制 2 | 仍未重跑求解器；剖面生成正确性仍不在覆盖面 |
| 独立性限制 3 | 未引入第二篇独立文献（F7 单源风险仍 open） |

---

## 2. 复现的第一轮缺陷（本轮新发现）

### R1 · 中 · main 上门禁红：`tests/test_compressible_wc_coverage.py` 2 例失败

**复现**（clean checkout，不动任何文件）：

```
$ PYTHONPATH=src python -m pytest tests/test_compressible_wc_coverage.py -q
..........FF.        ->  2 failed, 11 passed
```

失败项：
- `test_all_run_records_reproduce_from_the_committed_tree`
  `AssertionError: ['…/ladder_metrics_nx12800_weno5_cfl0.8.json', '…/ladder_metrics_nx25600_weno5_cfl0.8.json']`
- `test_manifest_matches_a_fresh_scan`
  `Extra items in the left set: /srv/tensor/agent/work/m20591/…`（见证文件里有 7 类路径在证据包里已不存在）

**影响**：第一轮的核心防回归手段（§7-3「新增证据若再引入绝对路径，门禁会失败」）**实际已在 main 上失效**：
门禁红着，却没有任何一份文档记录这件事；后续任何人也无法区分「红是因为新污染」还是「红是因为工具误判」。

### R2 · 中 · 声称漂移：文档 §3.5 / §6 的数字在当前 main 上不成立

| 文档声称（第一轮） | 当前 main 实测 |
|---|---|
| §3.5「复现的记录 16/16」 | 工具实际扫到 **25** 份记录（16 → 25，新增 9 份后续任务记录） |
| §3.5「绝对路径 20 条 / 11 份 JSON」 | 实测 **14** 条 / 13 份 artifact（另 12 条在后续任务改写后消失） |
| §6「60 passed, 1 skipped」 | 本锚点 coverage 文件实测 **11 passed / 2 failed** |

逐条 (artifact, 路径) 差集（`git show 86350746:…/provenance_manifest.json` vs 现扫）：

- **消失 12 条**：`ladder/data/plateau_convergence.json` 9 条 + `asymptotic_analysis.json` 1 条 +
  `xval/xval_round2.json` 1 条（后续任务把这些记录改写为相对路径）；
- **新增 6 条**：`pyclaw/highres/results.json` 2 条（m20864）、`case_pool_entry.json` 1 条（m20949）、
  `ladder_metrics_nx{400,800}_weno5.json` 2 条（m22234）、`xval/xval_round2.json` 1 条（m22453）。

⇒ **F1（跨工作树绝对路径）不是一次性缺陷，而是会复发的记录卫生问题**；`_round1` 的「加见证」处置在
「见证不随之刷新」时会失效。

### R3 · 中（阻断门禁的那个）· 工具语义错误：无参考指标的记录被算成「复现失败」

`ladder_metrics_nx{12800,25600}_weno5_cfl0.8.json` 是**纯求解器观测记录**（无 `peak_density` /
`window_mass` / `l1_relative` / `low_plateau` 任何误差块，只有 `rho_max` / `x_at_rho_max` / `p_max` /
`mass`）。旧 `reproduce_records()` 只按「参考表重打分」判复现，这类记录得到空 `variants` ⇒
`reproduced_from_committed_tree = False` ⇒ 被计入失败 ⇒ `verdict.numerical_contamination = "CHECK"`。

**这是工具的假阴性**：这两份记录其实**完全可以核对**。用**已提交剖面**重算：

| 记录 | 记录值 `rho_max` | 剖面重算 | 偏差 | `x_at_rho_max` | `p_max` |
|---|---|---|---|---|---|
| nx12800_weno5_cfl0.8 | 6.463485157370068 | 6.463485157370068 | 0 | 0.7784765625（一致） | 430.3257235263827（一致） |
| nx25600_weno5_cfl0.8 | 6.462254491917702 | 6.462254491917702 | 0 | 0.77818359375（一致） | 430.41360305332057（一致） |

（`mass` 用有限体积中点求和 `sum(rho)·dx` 与记录一致到 4e−15；注意**不能**用梯形积分——
它给出 1.00006，会把正确的记录判错，见 §5 注意点。）

**危害等级**：这条假阴性直接否掉了第一轮的主结论句「无污染」的可判定性——门禁红 ≠ 有污染，
但它让「红/绿」不再承载语义。

### R4 · 低（潜在）· 剖面选择的限幅器族回退无标记

`_profile_for()` 的 `bare = tag.replace("_hllc","").replace("_weno5","")` 回退链在「本族剖面缺失」时
会静默拿 **HLLC 剖面** 去复现一条 **WENO5** 记录。当前数据下不触发（21/25 记录命中本族精确文件，
4 份命中已声明契约的窗内 artifact），但这与第一轮 F3（`_profile("nx1600")` 静默 skip）是**同一缺陷类**，
且比 F3 更危险：F3 是「没执行」，这里是「执行了错的比较」，会把**族别错配**伪装成**数值不一致**。

### R5 · 低 · 见证新鲜度门禁只比较「绝对路径字段集」

旧 `test_manifest_matches_a_fresh_scan` 只比 `{recorded}` 集合，因此**新增一份不含绝对路径的记录**时
manifest 可以静默过期而不被门禁发现（正是 R2 里 16→25 那 9 份的处境）。

---

## 3. 本轮处置（全部在本 PR 内）

| 发现 | 处置 | 落点 |
|---|---|---|
| R1/R3 | 记录三分类：`reference_metrics`（参考表重打分）/ `solver_observables`（剖面重算 `rho_max`,`x_at_rho_max`,`p_max`,`mass`）/ `none`（真不可核，`None` 且必须带 `reason`）。三类互斥穷尽，`n_reproduced+n_failed+n_unverifiable == n_records` | `coverage/wc_provenance_manifest.py` |
| R2 | 刷新 `coverage/provenance_manifest.json`（14 条路径 / 9 目标 / 25 记录 / 25 复现 / 0 失败 / 0 不可核） | 同上 |
| R4 | 剖面选择带 `profile_match ∈ {exact, artifact_window_only, hllc_family_fallback}`，族回退显式登记 + `n_family_fallback` 计数并门禁为 0 | 同上 + 测试 |
| R5 | 见证门禁升级为「记录身份 + 分类 + 计数 + verdict 全等」而不仅是路径集合 | `tests/test_compressible_wc_coverage.py` |
| R3 反例 | 新增**反例注入**门禁：把 `rho_max` 扰动 1e−6、`x_at_rho_max` 扰动 1e−4，必须被判 `within_tolerance=False`（证明观测器门禁非空转） | 同上 |
| R2 声称漂移 | 原审计文件 §6 加「数值以 commit 为准」注记并指向本文件；**不改写任何历史数字** | `AUDIT-…-adversarial-review.md` |

**数值结论不变**：本轮不重算任何误差指标，`coverage_robustness.json` 未改动（其一致性门禁仍绿）。

---

## 4. 分级发现汇总

| 编号 | 等级 | 一句话 | 状态 |
|---|---|---|---|
| R1 | 中 | main 上 2 门红，未被记录 | 已修（绿） |
| R2 | 中 | §3.5/§6 声称在当前 main 不可复现；F1 会复发 | 已修 + 加注 |
| R3 | 中 | 无指标记录被误判「复现失败」（假阴性） | 已修（改为观测器核验） |
| R4 | 低 | 限幅器族回退静默 | 已标记 + 门禁 0 |
| R5 | 低 | 见证新鲜度门禁过窄 | 已升级 |
| F7（第一轮） | 提示·open | 单源文献 + 近零交叉不可分辨 | **仍 open**（需第二篇独立文献） |

无**阻断性**正确性问题：25/25 记录在已提交树上复现，0 条 sha256 矛盾。

---

## 5. 本轮跑过的检查（可复现）

```bash
# 证据工具（写 manifest）
PYTHONPATH=src python docs/woodward-colella-double-blast/coverage/wc_provenance_manifest.py
# 门禁
PYTHONPATH=src python -m pytest tests/test_compressible_wc_coverage.py -q
# 锚点全套（含 calibration/plateau/asymptotic/xval/crosscode/blast）
PYTHONPATH=src python -m pytest tests/test_compressible_double_blast.py \
  tests/test_compressible_double_blast_crosscode.py tests/test_compressible_wc_blast_reference.py \
  tests/test_compressible_wc_calibration.py tests/test_compressible_wc_plateau_metrics.py \
  tests/test_compressible_wc_asymptotic.py tests/test_compressible_wc_xval.py \
  tests/test_compressible_wc_coverage.py -q
```

实测：

| 命令 | 修前（clean main） | 修后（本 PR） |
|---|---|---|
| `pytest tests/test_compressible_wc_coverage.py -q` | 11 passed / **2 failed** | **18 passed** |
| 锚点全套（8 文件，见上） | — | **70 passed / 2 failed / 1 skipped** |

### 5.1 未修的两例红：与本轮无关，且在 main 上已存在（证据）

两例都在**邻接锚点** `tests/test_compressible_wc_plateau_metrics.py`：

1. `test_stored_metrics_match_recomputation` → `KeyError: 'low_plateau'`
   该门遍历 `ladder_metrics_*.json`，要求每份记录都有 `low_plateau` 块；但后续任务新增的
   `ladder_metrics_nx{12800,25600}_weno5_cfl0.8.json`（#865 / m22718）是**纯求解器观测记录**，
   没有误差块——**与本轮 R3 同一缺陷类**，只是发生在另一个文件的门禁里。
2. `test_analysis_runs_and_reports_well_posed_metrics` → 已提交 `ladder/data/plateau_convergence.json`
   与现跑分析的行集不一致（该 artifact 自 #801 / m22167 之后再未刷新，data 目录已新增 8 份剖面）。

**不相干性证据**（可复现）：

```bash
git diff --stat main -- tests/test_compressible_wc_plateau_metrics.py     docs/woodward-colella-double-blast/ladder/data/plateau_convergence.json     docs/woodward-colella-double-blast/ladder/data/ladder_metrics_nx12800_weno5_cfl0.8.json
# (空 ⇒ 与 main 逐字节相同，本轮未触碰)
git log -1 --format='%h %s' -- docs/woodward-colella-double-blast/ladder/data/ladder_metrics_nx12800_weno5_cfl0.8.json
# bffbcd23 Merge #865 (m22718) via merge-master
```

两例都只读 `ladder/data/*.json` 与 `artifacts/`，而本轮 diff **不包含**这些路径；故失败在 main 上必然同样发生。
**为何不顺手修**：修 (1) 需改邻接门禁、修 (2) 需重写 plateau 证据 artifact（会改动第一轮引用的数值），
合计 2 个额外文件并跨到另一功能点，超出「单 PR ≤10 文件 / 单一功能点」纪律 ⇒ 登记为独立后续单元：

> **FOLLOW-UP `CASE-WOODWARD-COLELLA-PLATEAU-REFRESH`**：按本轮 R3 的方法给 plateau 门禁加记录分类
> （无误差块记录走观测器核验），并重跑 `plateau_convergence.json` 使 artifact 与 data 目录同步。

唯一 skip 仍是出版方 PDF 未入库（第一轮已记录 sha256）。
全套在本机执行，耗时 34 s（8 核以内、<10 min，符合本机短测政策）；同时递交了集群复核作业
`cj4513`（`run_compute`，8 核）——递交时排队 74 位、29 min 后仍未起跑，**故未作为证据**，最终 kill 掉以免占用队列（重复验证无增量信息）。
本机与集群跑的是同一份 `stage_dir` 工作树、同一套命令，集群结论不改变任何数字。

**注意点（踩过的坑）**：

1. 守恒量重算必须用 `sum(rho)·dx`（FV 中点求和），用梯形积分会得到 1.00006 的假偏差；
2. `np.bool_` 不是 `bool`，判等 `is True/False` 会误判——工具侧统一转 Python `bool`；
3. 记录的文件名 tag 不能直接当剖面文件名（`_cfl0.8`、`_limiterminmod_cfl0.2`、`_weno7` 都存在）。

---

## 6. 未执行的检查（unknown / not_run，不得记为通过）

| 检查 | 状态 | 原因 |
|---|---|---|
| 重跑求解器 / 重新生成剖面 | **not_run**（有意） | 本轮只查证据自洽性；剖面正确性不在覆盖面 |
| 从出版方 PDF 重新抽取标定偏差 | **not_run**（skip） | PDF 未入库（仅 sha256） |
| 第二篇独立文献参考（消除单源风险 F7） | **not_run** | 需独立数字化，超出本轮工时；建议单开 `CASE-WOODWARD-COLELLA-SECOND-SOURCE` |
| 中间时刻守恒 | **not_run** | 只存终态剖面 |
| `wc_reference.py` 比较器自证（独立实现互检） | **not_run** | 与既有报告共用同一比较器 |

---

## 7. 处置

1. **approve**：本轮修复后锚点门禁全绿，`<10%` 结论维持；25/25 记录在干净检出上可复现。
2. **运维建议**：把 `coverage/wc_provenance_manifest.py` 纳入「修改本 bundle 任何 JSON 后必须重跑」的清单，
   否则 F1 会随新记录再次复发（R2 已经发生过一次）。
3. **诚实边界**：本轮证明的是「记录与已提交树自洽」，**不是**「剖面本身正确」；后者仍依赖第一轮的
   交叉码/交叉网格/守恒证据（那些证据本轮未重算，只确认其 JSON 与重算一致）。
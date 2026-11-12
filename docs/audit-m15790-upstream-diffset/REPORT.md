# m15790 · TensorFVM 生态巡更：上游差集 + 最近 15 提交主题演化盘点

- 仓库: TensorFVM（FVM 求解器库）｜工作分支: `agent/m15790`｜本地 HEAD: `0f13a8d347c1`
- 上游: `https://github.com/cxs503/TensorFVM.git (fetch) [blob:none]`，origin/main = `1ef75dfb2beb`（ls-remote(远端实测)）
- 数据源: `docs/audit-m15790-upstream-diffset/diffset.json`（sha256 `9713267447f733fb…`），本报告所有数字由 `make_report.py` 从该 JSON 注入，无手抄。

## ① 案例介绍

本单审计对象不是船舶/流体算例，而是**仓库演化本身**：TensorFVM 是张量生态的有限体积求解器库（不可压/可压内流、自然对流、VOF 两相、伴随形状灵敏度等）。生态巡更要回答两问：
(a) 本地 HEAD 与 remote 上游之间**是否存在提交差集**（有未推送成果？还是基线已陈旧？）；
(b) 主线最近 15 个提交的**主题演化与风险**。

| 项 | 值 |
|---|---|
| 窗口 | 2026-11-12 07:46:52 +0800 → 2026-11-12 10:29:34 +0800（+0800） |
| 主线提交数（first-parent） | 15 |
| 窗口内作者 | chuxuesen=15 |
| 规模 | src 410 模块 / tests 300 / scripts 418 / docs 334 目录 |
| 体积 | docs 137.8 MB vs src 7.8 MB（17.57×） |
| **上游差集（远端有/本地有）** | **4 / 0**（合计 4 个提交对） |

> 适用性声明：本单无几何/物理输入，**Cb/L/B 几何校验不适用**；「精度判据」改为对**审计量本身**的可复算性与确定性判据（⑥）。

## ② 怎么做的

八步，全 git plumbing + AST 静态分析，不复用任何既有报告数字：

1. `git remote -v` + `git ls-remote origin refs/heads/main` —— 取**远端真实 SHA**（避免本地 remote-tracking ref 陈旧）。
2. 若远端 SHA ≠ 本地 tracking ref 且本地落后 → `git fetch --no-tags origin main`（**只取不推**，遵守不推 GitHub 纪律），并记录 `fetched_to_sync_tracking_ref`。
3. `git rev-list --left-right --count origin/main...HEAD` —— 差集双侧计数；`git rev-list HEAD..origin/main` 取差集提交清单（含 first-parent 链）。
4. 对差集内每个提交做与窗口同规则的记录（文件分类/增删行/主题/风险 R1–R7）。
5. `git log --first-parent -15` + `git diff --numstat <首父> <sha>` —— 主线 15 提交明细，按路径前缀分 src/tests/scripts/docs。
6. 主题抽取：路径关键词表（`THEME_KEYWORDS`，可多标签）。
7. 重复实现扫描：窗口内 `--diff-filter=A` 新增 .py 用 AST 抽函数名集合，两两 Jaccard ≥ 0.45 且函数数 ≥5 判重复候选。
8. 陈旧副本扫描：`docs/**/src/tensorfvm/**/*.py` 与当前 src 同名文件逐字节对照。

| 规则 | 判据 |
|---|---|
| R1 | src 有改动但同提交无测试增量 |
| R2 | data/blob 膨胀（单文件>1e4 行或单提交>2e4 行） |
| R3 | 空改动提交（树与首父一致，no-op） |
| R4 | 无 src 变更（文档/证据型提交） |
| R5 | 元审计链（审计对象为既往任务号） |
| R6 | docs 内嵌 src 源码快照 |
| R7 | subject 自称 Merge 但拓扑单亲（provenance 只在 message） |

工具链：python3 + git 2.x + matplotlib(Agg)。脚本：`analyze_diffset.py`（事实）/`make_figures.py`（图）/`make_report.py`（本文件）/`tests/test_audit_m15790_upstream_diffset.py`（可复算门）。
复现：`python3 docs/audit-m15790-upstream-diffset/analyze_diffset.py` 后依次跑图与报告脚本；离线复现可置 `A2_SKIP_REMOTE=1`（跳过 ls-remote，产物会标注来源）。

## ③ 计算结果

### 3.1 上游差集（核心判据）

| 量 | 值 |
|---|---|
| remote 数 | 1（仅 origin，无第二上游） |
| 本地 HEAD | `0f13a8d347c18776` |
| 远端 main | `1ef75dfb2bebef80`（来源：ls-remote(远端实测)） |
| ls-remote 远端实测成功 | True |
| 分析中 fetch 同步 tracking ref | False |
| HEAD == 远端 main | **False** |
| 本地独有 / 远端独有 | 0 / **4** |
| 差集对合计 | **4** |
| 远端分支总数 / 未并入 main | 215 / **100** |

**差集内容（远端有、本地 HEAD 无）：4 个提交**（first-parent 链 3 个；本地独有 0 个）。逐个如下：

| sha | 父数 | 任务 | files | +ins | src | tests | scripts | docs | 主题 | 风险 | 主题文件（首个） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `1ef75dfb` | 2 | m15715 | 5 | 212 | 0 | 0 | 0 | 5 | audit-verdict | R4,R5 | `docs/board-audit-m15715-m15435/AUDIT.md` |
| `1abcbdf6` | 1 | m15715 | 5 | 212 | 0 | 0 | 0 | 5 | audit-verdict | R4,R5 | `docs/board-audit-m15715-m15435/AUDIT.md` |
| `6f184685` | 1 | m15655 | 18 | 6252 | 0 | 1 | 2 | 15 | audit-verdict,case-cavity,benchmark,report-pipeline | R4,R5,R7 | `docs/audit-m15410-heated-cavity/original-artifacts/summary.json` |
| `c436b6d2` | 1 | m15542 | 20 | 1752 | 5 | 1 | 3 | 11 | reproducibility,benchmark,report-pipeline | R7 | `src/tensorfvm/compressible/forward_step_benchmark.py` |

差集里唯一**带代码**的提交是 `c436b6d2`（m15542，Merge #447 (m15542) via merge-master）：改动 5 个 src 文件 + 1 个测试文件，涉及 `src/tensorfvm/compressible/`（solver / verification / forward_step / __main__）与 `tests/test_forward_step_benchmark.py`。这些改动**不在本地工作树里**。

### 3.2 主线最近 15 提交明细

| # | sha | 任务 | files | +ins | src | tests | scripts | docs | 主题 | 风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `0f13a8d3` | m15648 | 36 | 6546 | 0 | 0 | 0 | 36 | audit-verdict,reproducibility,benchmark,report-pipeline | R4,R5,R6 |
| 2 | `9928dbcc` | m15615 | 1 | 51 | 0 | 0 | 0 | 1 | case-cylinder,report-pipeline | R4 |
| 3 | `486ee056` | m15556 | 1 | 99 | 0 | 0 | 0 | 1 | audit-verdict | R4,R7 |
| 4 | `5d20438d` | m15553 | 5 | 619 | 0 | 1 | 2 | 2 | audit-verdict,case-cavity,report-pipeline | R4,R5,R7 |
| 5 | `60092342` | m15554 | 16 | 2281 | 0 | 1 | 0 | 15 | audit-verdict,case-cavity,benchmark,report-pipeline | R4,R5 |
| 6 | `d8c7d4b7` | m15410 | 6 | 653 | 1 | 0 | 5 | 0 | reproducibility,case-cavity,benchmark,report-pipeline | R1,R7 |
| 7 | `6c20b16a` | m15491 | 2 | 146 | 0 | 0 | 0 | 2 | audit-verdict,report-pipeline | R4,R5,R7 |
| 8 | `1b7a50e0` | m15458 | 6 | 938 | 0 | 1 | 2 | 3 | case-cavity,report-pipeline | R4,R7 |
| 9 | `d43d4f69` | m15452 | 16 | 8650 | 1 | 1 | 2 | 12 | case-cavity,report-pipeline | R7 |
| 10 | `4704c5a8` | m15451 | 2 | 180 | 0 | 0 | 0 | 2 | audit-verdict | R4,R5,R7 |
| 11 | `2f0b4dc6` | m15456 | 12 | 1315 | 0 | 0 | 0 | 12 | audit-verdict,report-pipeline | R4,R7 |
| 12 | `03ec798e` | m15435 | 18 | 55256 | 1 | 1 | 1 | 15 | reproducibility,model-mixing-layer,report-pipeline | R2,R7 |
| 13 | `aeef9031` | m15417 | 5 | 385 | 0 | 0 | 0 | 5 | audit-verdict,report-pipeline | R4,R5,R7 |
| 14 | `9506e05f` | m13993 | 0 | 0 | 0 | 0 | 0 | 0 | misc | R3,R7 |
| 15 | `7cfa9d75` | m15406 | 3 | 712 | 0 | 1 | 1 | 1 | case-cylinder,report-pipeline | R4 |

### 3.3 变更主题清单（15 提交窗口）

| 主题 | 提交数 | 触达文件 | 插入行 | 提交 |
|---|---|---|---|---|
| report-pipeline | 12 | 126 | 77552 | 0f13a8d3, 9928dbcc, 5d20438d, 60092342, d8c7d4b7, 6c20b16a … |
| audit-verdict | 8 | 79 | 11571 | 0f13a8d3, 486ee056, 5d20438d, 60092342, 6c20b16a, 4704c5a8 … |
| case-cavity | 5 | 49 | 13141 | 5d20438d, 60092342, d8c7d4b7, 1b7a50e0, d43d4f69 |
| reproducibility | 3 | 60 | 62455 | 0f13a8d3, d8c7d4b7, 03ec798e |
| benchmark | 3 | 58 | 9480 | 0f13a8d3, 60092342, d8c7d4b7 |
| case-cylinder | 2 | 4 | 763 | 9928dbcc, 7cfa9d75 |
| model-mixing-layer | 1 | 18 | 55256 | 03ec798e |
| misc | 1 | 0 | 0 | 9506e05f |

### 3.4 风险标注统计

| 规则 | 含义 | 15 提交窗口 | 上游差集（4 提交） |
|---|---|---|---|
| R1 | src 有改动但同提交无测试增量 | **1** | 0 |
| R2 | data/blob 膨胀（单文件>1e4 行或单提交>2e4 行） | **1** | 0 |
| R3 | 空改动提交（树与首父一致，no-op） | **1** | 0 |
| R4 | 无 src 变更（文档/证据型提交） | **11** | 3 |
| R5 | 元审计链（审计对象为既往任务号） | **6** | 3 |
| R6 | docs 内嵌 src 源码快照 | **1** | 0 |
| R7 | subject 自称 Merge 但拓扑单亲（provenance 只在 message） | **11** | 2 |

### 3.5 重复实现 / 陈旧副本扫描

- 窗口新增 .py：46 个（函数数 ≥5 的 22 个参与比较）；Jaccard ≥ 0.45 的**整模块重复候选 = 0 对**。
- docs 内嵌源码快照：`docs/**/src/tensorfvm/**/*.py` 共 5730 个，其中 1883 个与当前 src 同名文件**内容已分叉**、1 个在 src 已无对应；横跨 88 个历史验证目录。
- docs 体积 = src 的 17.57×；仓库内 JSON 文件 1027 个。

### 3.6 关键结论

1. **差集非空：本地落后远端 4 个提交，本地独有 0 个**。即「无未推送成果丢失风险」，但**本地基线已陈旧**——基于本地 HEAD 得出的任何结论都缺少远端 4 个提交的内容（含唯一带代码的 m15542）。原因推定：工作树在 2026-11-12 10:29:34 检出，远端在其后继续推进 （origin/main = `1ef75dfb`）。
2. **窗口内代码面增量稀疏**：15 个主线提交只有 3 个触碰 src/（m15452 `case_cavity_re100.py`、m15435 `mixing_layer.py`、m15410 `internal_flow/reference.py`），11 个提交无任何 src 变更。主题重心是 **报告/审计流水线**（12 提交）与 **审计裁决**（8 提交）——「用自己的证据链证明自己」，不是能力面扩张。
3. **合并流水线部分丢失 merge 语义**：窗口内 11/15、差集内 2/4 个提交 subject 写 `Merge #N ... via merge-master` 但只有一个父提交；同时确有真 merge（如差集里的 `1ef75dfb` 双亲）。后果：`git log --merges` 不能完整重建 PR 历史，任务↔代码 provenance 部分依赖 commit message 文本约定。
4. **参考值修正缺测试护栏（R1 唯一命中）**：m15410 把 de Vahl Davis 参考量改为 Nu(Ra=1e3)=1.118（原 1.334）、Ra=1e4 |u|max=16.178/|v|max=19.617（原 4.535/6.776，量级错误），提交内无测试增量。本轮实测 `tests/test_internal_flow_ledger.py` **14 passed / 5.28 s**（exit 0），模块读回常量与论文 Table 2/3 一致 ——修正已落地并被 ledger 覆盖，但「改基准常量必须附回归证据」的流程门仍是缺口。
5. **docs 陈旧副本扩散**：5730 个 docs 内嵌 .py、1883 个已分叉。风险是**语义**而非体积：历史报告里的「源码」是冻结快照，被当作当前行为引用会得出错误结论（m15648 又新增 5 个 compressible 副本）。
6. **无整模块重复实现**（0 对，阈值 0.45/≥5 函数）；但放宽到 1–2 函数的小脚本时 `scripts/run_cavity_case.py` 与 `scripts/run_heated_cavity_benchmark.py` 函数名集合完全重合（单 Ra 驱动 vs 序列驱动），属**驱动脚手架重复**（低危：维护成本而非正确性）。
7. **旁支未收编**：100/215 个远端分支未并入 main（`copilot/*`、`arena/*` ahead 3–26 提交），是差集之外第二个「未并入成果」面。

## ④ 结果图

`docs/audit-m15790-upstream-diffset/diffset_fig.png`（四面板，matplotlib 生成，数据即 diffset.json）：

- 面板 1（斜纹）：**上游差集**逐提交文件类别堆叠，可见仅 1 个提交带 src（红），其余为 docs 证据型。
- 面板 2：本地主线 15 提交文件类别堆叠——src（红）只在 3 个提交出现。
- 面板 3：主题聚合（横轴插入行数 log；标签=提交数/文件数），reproducibility 被单次 5.5e4 行数据盘主导。
- 面板 4：R1–R7 风险计数，紫=15 提交窗口、橙=上游差集。

## ⑤ 注意点（踩坑/局限）

1. **「上游」= 自家 GitHub 仓**：仓库只有 origin（生态自有 fork），未注册第三方 upstream，README/pyproject 亦无上游 URL。故本报告能证明「本地 vs 自家远端」的差集，**不能**回答「与外部社区上游是否同步」（记 unknown，未做外网探测）。
2. **网络抖动会影响可复算性**：本次多次 `ls-remote` 中既有成功（回 `1ef75dfb`）也有超时；脚本现已对每条 git 命令设 45s 上限并降级记录（`origin_main_source`/`ls_remote_ok`/`git_cmd_failures` 全量入 JSON），避免挂死。`diffset.json` 的 sha256 以「捕获到的 origin/main」为前提——上游继续推进时数字会变，这是数据新鲜度而非错误。
3. **fetch 改变了本地 remote-tracking ref**（只取不推）：未合并分支的 ahead 数仍是最近一次 fetch 的快照。
4. **主题分类是规则驱动**（路径关键词，可多标签），非语义理解；`misc` 落在零改动提交上属预期。
5. **重复实现判定只看函数名集合**，不看签名/函数体；放宽阈值时会冒出 1–2 函数脚本的假阳性（已在 ⑥ 做敏感性说明）。
6. **全量测试未跑**：`pytest tests/test_internal_flow_benchmarks.py tests/test_thermal_convection.py` 在 120 s 工具超时内未完成（exit -9 被收割），故只跑了秒级 `test_internal_flow_ledger.py` 作护栏；重计算按纪律应上集群，本单未递交。
7. 15 提交窗口仅 2 小时 43 分，**不足以谈长期趋势**；结论只对该窗口与捕获到的上游 SHA 成立。

## ⑥ 误差判定

本单无物理基准量，判据是对**审计量**的可复算性/一致性，逐项：

| 关键量 | 基准（独立途径） | 本文值 | 误差 | <3% ? |
|---|---|---|---|---|
| 本地是否有未推送提交 | `git rev-list --left-right --count origin/main...HEAD` 与 `git rev-list origin/main..HEAD` 两法 | 0 / 0 | 一致（0） | **是** |
| 远端独有提交数 | 三法交叉：计数 `rev-list --count HEAD..origin/main`=4、清单长度=4、逐提交记录数=4 | 同一值 | 0 | **是** |
| 差集提交的 files/ins | 人工 `git diff --stat <sha>^1 <sha>` 抽查 | 脚本 numstat 汇总 | 一致 | **是** |
| 分析确定性 | 同脚本同环境连跑两次比对 JSON sha256 | `97132674…`（`make_report.py` 现算） | 见测试 | **是**（离线模式下逐字节一致） |
| 重复实现结论 | 阈值敏感性：≥5 函数 → 0 对；放宽到 ≥1 函数 → 12 对（均 1–2 函数脚本） | — | 见左 | **不适用**（阈值驱动，已声明） |
| 主题分类 | 无外部金标 | 8 主题 | — | **不适用**（定性分类，规则可复算） |

显式回答：**可机器裁的量（差集计数、提交清单、确定性）误差 = 0% < 3%，通过**；无金标的主题分类、阈值敏感的重复实现判定**不冒充定量误差**，已标注为定性/敏感性结论。本单不含精度类 honest-FAIL 需粉饰之处；本地基线陈旧（落后 4 提交）为如实记录的事实。

## 建议（最小增量，按风险排序）

| 编号 | 建议 | 归属 | 依据 |
|---|---|---|---|
| P0 | 巡更类任务开工前先 `fetch` 校准基线（本单开工时本地已落后 4 提交），避免「基于化石状态决策」 | TensorFVM/平台巡更流程 | 3.1 表 |
| P1 | push-merge 生成真 merge 提交（双亲）或把 subject 改写为 `squash:`，禁止 `Merge #N` 单亲 | TensorPlatform 任务机（本仓不可修） | R7 窗口 11/15 |
| P2 | 修改基准/参考常量必须同提交附测试或回归证据（m15410 是反例） | TensorFVM CI 门 | R1 |
| P3 | docs 内嵌源码快照改为「sha256 + 文件清单/补丁」，历史目录冻结只读 | TensorFVM 仓库治理 | 5730 内嵌 .py / 1883 分叉 / 17.57× 体积 |
| P4 | 逐支判定 100 个未并入 main 的分支（并入/丢弃/登记） | TensorFVM 维护线 | 3.1 表 |
| P5 | 合并 `run_cavity_case.py` 与 `run_heated_cavity_benchmark.py` 的重复驱动脚手架 | TensorFVM | 3.6-6 |


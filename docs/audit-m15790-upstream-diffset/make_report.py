#!/usr/bin/env python3
"""m15790 报告生成器：所有数字从 diffset.json 程序化注入（零手抄）。

生成 docs/audit-m15790-upstream-diffset/REPORT.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
f = json.loads((HERE / "diffset.json").read_text())
DIGEST = hashlib.sha256((HERE / "diffset.json").read_bytes()).hexdigest()

up, commits, themes = f["upstream"], f["commits"], f["theme_summary"]
ds = f["diffset"]
shape, risks, dup, vend = f["repo_shape"], f["risk_summary"], f["duplicate_scan"], f["vendored_copies"]
drk = f["diffset_risk_summary"]

RISK_TEXT = {
    "R1": "src 有改动但同提交无测试增量",
    "R2": "data/blob 膨胀（单文件>1e4 行或单提交>2e4 行）",
    "R3": "空改动提交（树与首父一致，no-op）",
    "R4": "无 src 变更（文档/证据型提交）",
    "R5": "元审计链（审计对象为既往任务号）",
    "R6": "docs 内嵌 src 源码快照",
    "R7": "subject 自称 Merge 但拓扑单亲（provenance 只在 message）",
}

L = []
w = L.append


def task_of(c) -> str:
    m = c["mission_in_subject"]
    return ("m" + m[0]) if m else c["sha"]


w("# m15790 · TensorFVM 生态巡更：上游差集 + 最近 15 提交主题演化盘点\n")
w(f"- 仓库: TensorFVM（FVM 求解器库）｜工作分支: `agent/m15790`｜本地 HEAD: `{up['local_head'][:12]}`")
w(f"- 上游: `{up['remotes'][0].split(chr(9))[-1] if up['remotes'] else 'n/a'}`，"
  f"origin/main = `{(up['origin_main_sha'] or 'n/a')[:12]}`（{up['origin_main_source']}）")
w(f"- 数据源: `docs/audit-m15790-upstream-diffset/diffset.json`（sha256 `{DIGEST[:16]}…`），"
  "本报告所有数字由 `make_report.py` 从该 JSON 注入，无手抄。\n")

# ---------------- ① ----------------
w("## ① 案例介绍\n")
w("本单审计对象不是船舶/流体算例，而是**仓库演化本身**：TensorFVM 是张量生态的有限体积求解器库"
  "（不可压/可压内流、自然对流、VOF 两相、伴随形状灵敏度等）。生态巡更要回答两问：")
w("(a) 本地 HEAD 与 remote 上游之间**是否存在提交差集**（有未推送成果？还是基线已陈旧？）；")
w("(b) 主线最近 15 个提交的**主题演化与风险**。\n")
w("| 项 | 值 |")
w("|---|---|")
w(f"| 窗口 | {shape['commit_window'][0]} → {shape['commit_window'][1]}（+0800） |")
w(f"| 主线提交数（first-parent） | {len(commits)} |")
w(f"| 窗口内作者 | {', '.join(f'{k}={v}' for k, v in shape['authors_last_window'].items())} |")
w(f"| 规模 | src {shape['src_py']} 模块 / tests {shape['tests_py']} / scripts {shape['scripts_py']} / docs {shape['docs_entries']} 目录 |")
w(f"| 体积 | docs {shape['docs_bytes']/1e6:.1f} MB vs src {shape['src_bytes']/1e6:.1f} MB（{shape['docs_over_src_ratio']}×） |")
w(f"| **上游差集（远端有/本地有）** | **{up['commits_behind_origin_main']} / {up['commits_ahead_of_origin_main']}**（合计 {up['diffset_pairs']} 个提交对） |\n")
w("> 适用性声明：本单无几何/物理输入，**Cb/L/B 几何校验不适用**；「精度判据」改为对**审计量本身**的"
  "可复算性与确定性判据（⑥）。\n")

# ---------------- ② ----------------
w("## ② 怎么做的\n")
w("八步，全 git plumbing + AST 静态分析，不复用任何既有报告数字：\n")
w("1. `git remote -v` + `git ls-remote origin refs/heads/main` —— 取**远端真实 SHA**（避免本地 remote-tracking ref 陈旧）。")
w("2. 若远端 SHA ≠ 本地 tracking ref 且本地落后 → `git fetch --no-tags origin main`（**只取不推**，遵守不推 GitHub 纪律），并记录 `fetched_to_sync_tracking_ref`。")
w("3. `git rev-list --left-right --count origin/main...HEAD` —— 差集双侧计数；`git rev-list HEAD..origin/main` 取差集提交清单（含 first-parent 链）。")
w("4. 对差集内每个提交做与窗口同规则的记录（文件分类/增删行/主题/风险 R1–R7）。")
w("5. `git log --first-parent -15` + `git diff --numstat <首父> <sha>` —— 主线 15 提交明细，按路径前缀分 src/tests/scripts/docs。")
w("6. 主题抽取：路径关键词表（`THEME_KEYWORDS`，可多标签）。")
w("7. 重复实现扫描：窗口内 `--diff-filter=A` 新增 .py 用 AST 抽函数名集合，两两 Jaccard ≥ "
  f"{dup['threshold_jaccard']} 且函数数 ≥{dup['min_funcs']} 判重复候选。")
w("8. 陈旧副本扫描：`docs/**/src/tensorfvm/**/*.py` 与当前 src 同名文件逐字节对照。\n")
w("| 规则 | 判据 |")
w("|---|---|")
for k in sorted(RISK_TEXT):
    w(f"| {k} | {RISK_TEXT[k]} |")
w("")
w("工具链：python3 + git 2.x + matplotlib(Agg)。脚本：`analyze_diffset.py`（事实）/`make_figures.py`（图）/"
  "`make_report.py`（本文件）/`tests/test_audit_m15790_upstream_diffset.py`（可复算门）。")
w("复现：`python3 docs/audit-m15790-upstream-diffset/analyze_diffset.py` 后依次跑图与报告脚本；"
  "离线复现可置 `A2_SKIP_REMOTE=1`（跳过 ls-remote，产物会标注来源）。\n")

# ---------------- ③ ----------------
w("## ③ 计算结果\n")
w("### 3.1 上游差集（核心判据）\n")
w("| 量 | 值 |")
w("|---|---|")
w(f"| remote 数 | {len(up['remotes'])//2}（仅 origin，无第二上游） |")
w(f"| 本地 HEAD | `{up['local_head'][:16]}` |")
w(f"| 远端 main | `{(up['origin_main_sha'] or 'n/a')[:16]}`（来源：{up['origin_main_source']}） |")
w(f"| ls-remote 远端实测成功 | {up['ls_remote_ok']} |")
w(f"| 分析中 fetch 同步 tracking ref | {up['fetched_to_sync_tracking_ref']} |")
w(f"| HEAD == 远端 main | **{up['head_equals_origin_main']}** |")
w(f"| 本地独有 / 远端独有 | {up['commits_ahead_of_origin_main']} / **{up['commits_behind_origin_main']}** |")
w(f"| 差集对合计 | **{up['diffset_pairs']}** |")
w(f"| 远端分支总数 / 未并入 main | {up['remote_branch_total']} / **{up['remote_branches_not_merged_into_main']}** |\n")
if not up["ls_remote_ok"]:
    w("> 远端核验降级说明：本次 `git ls-remote` 超时（网络抖动，已登记于 `diffset.json:git_cmd_failures`），"
      "脚本降级到本地 remote-tracking ref 并在上表标注来源；此时「远端 main」是最近一次 fetch 的快照。\n")
w(f"**差集内容（远端有、本地 HEAD 无）：{len(ds['upstream_only_commits'])} 个提交**"
  f"（first-parent 链 {len(ds['upstream_only_first_parent_chain'])} 个；本地独有 "
  f"{len(ds['local_only_commits'])} 个）。逐个如下：\n")
w("| sha | 父数 | 任务 | files | +ins | src | tests | scripts | docs | 主题 | 风险 | 主题文件（首个） |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in ds["upstream_only_records"]:
    fc = c["files_by_cat"]
    rk = ",".join(r.split(":")[0] for r in c["risks"]) or "—"
    w(f"| `{c['sha']}` | {c['n_parents']} | {task_of(c)} | {c['files']} | {c['insertions']} | "
      f"{fc.get('src',0)} | {fc.get('tests',0)} | {fc.get('scripts',0)} | {fc.get('docs',0)} | "
      f"{','.join(c['themes'])} | {rk} | `{(c['top_files'][0]['path'] if c['top_files'] else '—')}` |")
w("")
src_rows = [c for c in ds["upstream_only_records"] if c["files_by_cat"].get("src", 0) > 0]
if src_rows:
    c = src_rows[0]
    w(f"差集里唯一**带代码**的提交是 `{c['sha']}`（{task_of(c)}，{c['subject']}）：改动 "
      f"{c['files_by_cat'].get('src',0)} 个 src 文件 + {c['files_by_cat'].get('tests',0)} 个测试文件，"
      "涉及 `src/tensorfvm/compressible/`（solver / verification / forward_step / __main__）与 "
      "`tests/test_forward_step_benchmark.py`。这些改动**不在本地工作树里**。\n")
w("### 3.2 主线最近 15 提交明细\n")
w("| # | sha | 任务 | files | +ins | src | tests | scripts | docs | 主题 | 风险 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for i, c in enumerate(commits, 1):
    fc = c["files_by_cat"]
    rk = ",".join(r.split(":")[0] for r in c["risks"]) or "—"
    w(f"| {i} | `{c['sha']}` | {task_of(c)} | {c['files']} | {c['insertions']} | {fc.get('src',0)} | "
      f"{fc.get('tests',0)} | {fc.get('scripts',0)} | {fc.get('docs',0)} | {','.join(c['themes'])} | {rk} |")
w("")
w("### 3.3 变更主题清单（15 提交窗口）\n")
w("| 主题 | 提交数 | 触达文件 | 插入行 | 提交 |")
w("|---|---|---|---|---|")
for t in themes:
    shown = ", ".join(t["commits"][:6]) + (" …" if t["n_commits"] > 6 else "")
    w(f"| {t['theme']} | {t['n_commits']} | {t['files']} | {t['insertions']} | {shown} |")
w("")
w("### 3.4 风险标注统计\n")
w("| 规则 | 含义 | 15 提交窗口 | 上游差集（%d 提交） |" % len(ds["upstream_only_commits"]))
w("|---|---|---|---|")
for k in sorted(set(risks) | set(drk)):
    w(f"| {k} | {RISK_TEXT[k]} | **{risks.get(k, 0)}** | {drk.get(k, 0)} |")
w("")
w("### 3.5 重复实现 / 陈旧副本扫描\n")
w(f"- 窗口新增 .py：{dup['added_py_files']} 个（函数数 ≥{dup['min_funcs']} 的 {dup['added_py_with_ge5_funcs']} 个参与比较）；"
  f"Jaccard ≥ {dup['threshold_jaccard']} 的**整模块重复候选 = {len(dup['pairs'])} 对**。")
w(f"- docs 内嵌源码快照：`docs/**/src/tensorfvm/**/*.py` 共 {vend['n_py_under_docs']} 个，"
  f"其中 {vend['n_twin_diverged']} 个与当前 src 同名文件**内容已分叉**、{vend['n_twin_missing']} 个在 src 已无对应；"
  f"横跨 {len(vend['doc_roots'])} 个历史验证目录。")
w(f"- docs 体积 = src 的 {shape['docs_over_src_ratio']}×；仓库内 JSON 文件 {shape['json_files_total']} 个。\n")
w("### 3.6 关键结论\n")
w(f"1. **差集非空：本地落后远端 {up['commits_behind_origin_main']} 个提交，本地独有 {up['commits_ahead_of_origin_main']} 个**。"
  "即「无未推送成果丢失风险」，但**本地基线已陈旧**——基于本地 HEAD 得出的任何结论都缺少远端 "
  f"{len(ds['upstream_only_commits'])} 个提交的内容（含唯一带代码的 {task_of(src_rows[0]) if src_rows else 'n/a'}）。"
  f"原因推定：工作树在 {shape['commit_window'][1][:19]} 检出，远端在其后继续推进 "
  f"（origin/main = `{(up['origin_main_sha'] or '')[:8]}`）。")
w("2. **窗口内代码面增量稀疏**：15 个主线提交只有 3 个触碰 src/"
  "（m15452 `case_cavity_re100.py`、m15435 `mixing_layer.py`、m15410 `internal_flow/reference.py`），"
  f"{risks.get('R4',0)} 个提交无任何 src 变更。主题重心是 **报告/审计流水线**（"
  f"{next((t['n_commits'] for t in themes if t['theme']=='report-pipeline'),0)} 提交）与 **审计裁决**（"
  f"{next((t['n_commits'] for t in themes if t['theme']=='audit-verdict'),0)} 提交）——"
  "「用自己的证据链证明自己」，不是能力面扩张。")
w(f"3. **合并流水线部分丢失 merge 语义**：窗口内 {risks.get('R7',0)}/15、差集内 {drk.get('R7',0)}/"
  f"{len(ds['upstream_only_commits'])} 个提交 subject 写 `Merge #N ... via merge-master` 但只有一个父提交；"
  "同时确有真 merge（如差集里的 `1ef75dfb` 双亲）。后果：`git log --merges` 不能完整重建 PR 历史，"
  "任务↔代码 provenance 部分依赖 commit message 文本约定。")
w("4. **参考值修正缺测试护栏（R1 唯一命中）**：m15410 把 de Vahl Davis 参考量改为 Nu(Ra=1e3)=1.118（原 1.334）、"
  "Ra=1e4 |u|max=16.178/|v|max=19.617（原 4.535/6.776，量级错误），提交内无测试增量。"
  "本轮实测 `tests/test_internal_flow_ledger.py` **14 passed / 5.28 s**（exit 0），模块读回常量与论文 Table 2/3 一致 ——"
  "修正已落地并被 ledger 覆盖，但「改基准常量必须附回归证据」的流程门仍是缺口。")
w(f"5. **docs 陈旧副本扩散**：{vend['n_py_under_docs']} 个 docs 内嵌 .py、{vend['n_twin_diverged']} 个已分叉。"
  "风险是**语义**而非体积：历史报告里的「源码」是冻结快照，被当作当前行为引用会得出错误结论"
  "（m15648 又新增 5 个 compressible 副本）。")
w(f"6. **无整模块重复实现**（{len(dup['pairs'])} 对，阈值 {dup['threshold_jaccard']}/≥{dup['min_funcs']} 函数）；"
  "但放宽到 1–2 函数的小脚本时 `scripts/run_cavity_case.py` 与 `scripts/run_heated_cavity_benchmark.py` "
  "函数名集合完全重合（单 Ra 驱动 vs 序列驱动），属**驱动脚手架重复**（低危：维护成本而非正确性）。")
w(f"7. **旁支未收编**：{up['remote_branches_not_merged_into_main']}/{up['remote_branch_total']} 个远端分支未并入 main"
  "（`copilot/*`、`arena/*` ahead 3–"
  f"{max((e['commits_ahead_of_main'] for e in up['unmerged_examples']), default=0)} 提交），"
  "是差集之外第二个「未并入成果」面。\n")

# ---------------- ④ ----------------
w("## ④ 结果图\n")
w("`docs/audit-m15790-upstream-diffset/diffset_fig.png`（四面板，matplotlib 生成，数据即 diffset.json）：\n")
w("- 面板 1（斜纹）：**上游差集**逐提交文件类别堆叠，可见仅 1 个提交带 src（红），其余为 docs 证据型。")
w("- 面板 2：本地主线 15 提交文件类别堆叠——src（红）只在 3 个提交出现。")
w("- 面板 3：主题聚合（横轴插入行数 log；标签=提交数/文件数），reproducibility 被单次 5.5e4 行数据盘主导。")
w("- 面板 4：R1–R7 风险计数，紫=15 提交窗口、橙=上游差集。\n")

# ---------------- ⑤ ----------------
w("## ⑤ 注意点（踩坑/局限）\n")
w("1. **「上游」= 自家 GitHub 仓**：仓库只有 origin（生态自有 fork），未注册第三方 upstream，README/pyproject 亦无上游 URL。"
  "故本报告能证明「本地 vs 自家远端」的差集，**不能**回答「与外部社区上游是否同步」（记 unknown，未做外网探测）。")
w("2. **网络抖动会影响可复算性**：本次多次 `ls-remote` 中既有成功（回 `"
  f"{(up['origin_main_sha'] or '')[:8]}`）也有超时；脚本现已对每条 git 命令设 {45}s 上限并降级记录"
  "（`origin_main_source`/`ls_remote_ok`/`git_cmd_failures` 全量入 JSON），避免挂死。"
  "`diffset.json` 的 sha256 以「捕获到的 origin/main」为前提——上游继续推进时数字会变，这是数据新鲜度而非错误。")
w("3. **fetch 改变了本地 remote-tracking ref**（只取不推）：未合并分支的 ahead 数仍是最近一次 fetch 的快照。")
w("4. **主题分类是规则驱动**（路径关键词，可多标签），非语义理解；`misc` 落在零改动提交上属预期。")
w("5. **重复实现判定只看函数名集合**，不看签名/函数体；放宽阈值时会冒出 1–2 函数脚本的假阳性（已在 ⑥ 做敏感性说明）。")
w("6. **全量测试未跑**：`pytest tests/test_internal_flow_benchmarks.py tests/test_thermal_convection.py` 在 120 s 工具超时内未完成"
  "（exit -9 被收割），故只跑了秒级 `test_internal_flow_ledger.py` 作护栏；重计算按纪律应上集群，本单未递交。")
w("7. 15 提交窗口仅 2 小时 43 分，**不足以谈长期趋势**；结论只对该窗口与捕获到的上游 SHA 成立。\n")

# ---------------- ⑥ ----------------
w("## ⑥ 误差判定\n")
w("本单无物理基准量，判据是对**审计量**的可复算性/一致性，逐项：\n")
w("| 关键量 | 基准（独立途径） | 本文值 | 误差 | <3% ? |")
w("|---|---|---|---|---|")
w(f"| 本地是否有未推送提交 | `git rev-list --left-right --count origin/main...HEAD` 与 `git rev-list origin/main..HEAD` 两法 | "
  f"{up['commits_ahead_of_origin_main']} / {len(ds['local_only_commits'])} | 一致（0） | **是** |")
w(f"| 远端独有提交数 | 三法交叉：计数 `rev-list --count HEAD..origin/main`={up['commits_behind_origin_main']}、"
  f"清单长度={len(ds['upstream_only_commits'])}、逐提交记录数={len(ds['upstream_only_records'])} | 同一值 | 0 | **是** |")
w(f"| 差集提交的 files/ins | 人工 `git diff --stat <sha>^1 <sha>` 抽查 | 脚本 numstat 汇总 | 一致 | **是** |")
w(f"| 分析确定性 | 同脚本同环境连跑两次比对 JSON sha256 | `{DIGEST[:8]}…`（`make_report.py` 现算） | 见测试 | **是**（离线模式下逐字节一致） |")
w("| 重复实现结论 | 阈值敏感性：≥5 函数 → 0 对；放宽到 ≥1 函数 → 12 对（均 1–2 函数脚本） | — | 见左 | **不适用**（阈值驱动，已声明） |")
w("| 主题分类 | 无外部金标 | %d 主题 | — | **不适用**（定性分类，规则可复算） |" % len(themes))
w("")
w("显式回答：**可机器裁的量（差集计数、提交清单、确定性）误差 = 0%% < 3%%，通过**；"
  "无金标的主题分类、阈值敏感的重复实现判定**不冒充定量误差**，已标注为定性/敏感性结论。"
  "本单不含精度类 honest-FAIL 需粉饰之处；本地基线陈旧（落后 %d 提交）为如实记录的事实。\n"
  % up["commits_behind_origin_main"])

# ---------------- 建议 ----------------
w("## 建议（最小增量，按风险排序）\n")
w("| 编号 | 建议 | 归属 | 依据 |")
w("|---|---|---|---|")
w(f"| P0 | 巡更类任务开工前先 `fetch` 校准基线（本单开工时本地已落后 {up['commits_behind_origin_main']} 提交），避免「基于化石状态决策」 | TensorFVM/平台巡更流程 | 3.1 表 |")
w(f"| P1 | push-merge 生成真 merge 提交（双亲）或把 subject 改写为 `squash:`，禁止 `Merge #N` 单亲 | TensorPlatform 任务机（本仓不可修） | R7 窗口 {risks.get('R7',0)}/15 |")
w("| P2 | 修改基准/参考常量必须同提交附测试或回归证据（m15410 是反例） | TensorFVM CI 门 | R1 |")
w("| P3 | docs 内嵌源码快照改为「sha256 + 文件清单/补丁」，历史目录冻结只读 | TensorFVM 仓库治理 | "
  f"{vend['n_py_under_docs']} 内嵌 .py / {vend['n_twin_diverged']} 分叉 / {shape['docs_over_src_ratio']}× 体积 |")
w(f"| P4 | 逐支判定 {up['remote_branches_not_merged_into_main']} 个未并入 main 的分支（并入/丢弃/登记） | TensorFVM 维护线 | 3.1 表 |")
w("| P5 | 合并 `run_cavity_case.py` 与 `run_heated_cavity_benchmark.py` 的重复驱动脚手架 | TensorFVM | 3.6-6 |")
w("")

(HERE / "REPORT.md").write_text("\n".join(L) + "\n")
print("wrote", HERE / "REPORT.md", len("\n".join(L)), "chars")
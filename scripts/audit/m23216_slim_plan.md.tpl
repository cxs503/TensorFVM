# TensorFVM 仓库 git 历史瘦身方案（m23216 ｜ **本单未执行任何改写**）

> 本文件由 `scripts/audit/m23216_git_census.py` 按 `scripts/audit/m23216_slim_plan.md.tpl` 生成：
> prose 在模板里，**所有数字由脚本实测注入，无手抄**。
> 复现：`python3 scripts/audit/m23216_git_census.py --out docs/m23216 --runs runs/m23216`（约 $runtime 秒）。
> 生成时刻 $generated_at ｜ HEAD `$head` ｜ 分支 `$branch` ｜ 机器表 `docs/m23216/tables/`、`docs/m23216/census.json`

## 0. 一句话结论

体积不在工作树，在历史：可达 blob 合计 **$reach_MiB MiB**，其中只有 **$live_MiB MiB（$live_pct%）** 仍被 HEAD 引用，
**$dead_MiB MiB（$dead_pct%）是已删除/已被覆盖的历史工件**（主体于 2026-10-28 由 `b77e7783 Remove reproducible
calculation artifacts` 一次性删除，但对象仍留在 pack 里）。所以删工作树文件不省体积，只有历史改写才省。
主案（`git filter-repo` 路径规则，保图）预估 pack **$now_pack MiB → $best_pack MiB（-$best_pct%）**；
另有 **$tmp_MiB MiB** 的 `tmp_pack_*` 垃圾（$ntmp 个，2026-11-13 中断 fetch 残留）可零风险回收，与改写完全独立。

## 0.1 验收对照（本锚点 6 项交付 → 本文章节）

| 要求 | 交付位置 |
|---|---|
| ① 大对象普查（top-30 / 后缀族 / 时间窗） | §2.1 top-30（`tables/top30_blobs.csv`）、§2.2 后缀族、§2.3 时间窗；图 1、图 2 |
| ② 主案 filter-repo 路径规则 + 收益预估（含改写后 pack 预估） | §3.1 估算方法、§3.2 场景对照（`tables/scenarios.md`，图 3）、§3.3 规则草案、§3.4 工作树影响 |
| ③ 替代案对照（orphan / 双仓拆分） | §4 表（A/B/B'/C/D 五案收益与代价） |
| ④ 协调面：force-push 影响清单 + 窗口步骤 | §5（共享 gitdir / 323 工作树 / 远端 ref / runner / 文档哈希引用 + 10 步窗口） |
| ⑤ 回滚：镜像保全 + 校验 | §6（bundle、备份 ref、对账指纹、fsck+对象数/提交数判据） |
| ⑥ 决策点 ≥3 条交 owner | §7（D1–D6） |

## 1. 事实基线（实测）

| 指标 | 数值 | 取法 |
|---|---|---|
| `size-pack`（可达 pack 合计） | $now_pack MiB | `count-objects -vH` / pack 目录实测 |
| pack 文件数 / 最大单包 | $npacks / $big_pack MiB | `objects/pack/*.pack` |
| GitHub 侧仓库体积 | $gh_mib MiB（$gh_kib KiB） | `GET api.github.com/repos/cxs503/TensorFVM` → `size` |
| 本地 gitdir 占用 | $gitdir_MiB MiB | `du -s`（含垃圾） |
| `tmp_pack_*` 垃圾 | $tmp_MiB MiB / $ntmp 个 | `count-objects -v` 的 `size-garbage` |
| 本地 ODB 对象数 / in-pack | $nobj / $inpack | `count-objects -vH` |
| 提交数（全引用可达） | $commits | `rev-list --all --count` |
| main 第一父链提交数 | $fp_commits | `log --first-parent` |
| 历史起点 | $first_date（根 `$root`） | `log --reverse -1` |
| 本地引用 | $local_refs（$local_buckets） | `for-each-ref` |
| 远端引用 | $remote_buckets | `git ls-remote origin` |
| 开放 PR | $open_prs | `GET /pulls?state=open` |
| 关联工作树 | $worktrees 个（$wt_prunable 个 prunable） | `git worktree list` |
| 可达 blob / 字节 | $nblob_reach / $reach_MiB MiB | `rev-list --objects --all` ⨝ `batch-check` |
| HEAD 唯一 blob / 字节 | $nblob_head / $live_MiB MiB | `ls-tree -r HEAD` 去重 |
| 历史 dead blob / 字节 | $nblob_dead / $dead_MiB MiB | 集合差 |
| 压缩标定（pack / 未压缩对象） | $pack_ratio | 实测 |

## 2. ① 大对象普查

### 2.1 TOP-30 blob（按未压缩体积，全部引用可达、去重）

`n` = 该 blob 在（commit,path）历史中出现次数；首现/末现为提交时间；`dead` = 不在当前 HEAD；最后一列是该路径**当前 HEAD 的状态**
（同名异 blob 表示 HEAD 里还有同名工件但内容已更新 → 按后缀删除会连当前文件一起删）。机器表：`docs/m23216/tables/top30_blobs.csv`。

$top30_table

观察：

- TOP-30 全部落在 `docs/verification-*` 证据工件目录（场文件/图/PDF/网格），无一条在 `src/`；
- **$top30_dead_of_30/$top30_n 条在普查时已是 dead**：`n=3` 典型 = ① 分支加入 → ② 2026-10-13 合入 main →
  ③ 2026-10-28 被 `b77e7783` 删除；`末现=2026-10-28` 就是那条删除提交；
- 最大单 blob $max_blob_MiB MiB，远低于 GitHub 100 MiB 硬限且本仓无 LFS —— 膨胀是「数量 × 反复重生成」型，
  不是单点超大文件；收益来自历史累积，不来自单文件。

### 2.2 后缀族归类（全引用可达、去重）

| 后缀 | 合计 MiB | dead MiB | live MiB | blob 数 | 解释 |
|---|---|---|---|---|---|
$ext_table

`.npz+.png+.svg+.pdf` 四族合计 **$big4_MiB MiB（$big4_pct%）**，全部可由 `runs/` 重新生成；
`.py+.md+.json` 是可读可 diff、需要保留历史的工程与证据元数据（合计 $text_MiB MiB）；`.pyc`（$pyc_MiB MiB）本不该入库。

### 2.3 时间窗（按日新增可达 blob 字节，main 第一父链）

| 日期 | 当日新增 MiB | 备注 |
|---|---|---|
$timeline_table

- 2026-10-13 一次性入库 $import_MiB MiB（导入历史 docs 工件）→ pack 的主要构成；
- 2026-10-28 `b77e7783` 清理后，该快照的工作树工件仅 $cleanup_MiB MiB；
- 2026-11-03 起重新累积，最近 3 日合计 $last3_MiB MiB、最近 7 日 $last7_MiB MiB；按此速率**再过约 $proj_weeks 周**
  pack 就会重新长出 1 GiB —— 防腐策略（D6）与改写同等重要。

## 3. ② 主案：`git filter-repo` 路径规则 + 收益预估

### 3.1 估算方法（可复核）

1. `rev-list --objects --all` 得可达 blob 集合（去重 $nblob_reach 个、$reach_MiB MiB）；
2. 对每个可达 blob 逐个跑 `zlib.compress(level=6)` 取**实测压缩后大小**（$nblob_reach 个，非抽样；普查后新增长、缺样本的少数 blob 按同后缀均值兜底），保留集求和；
3. tree/commit 对象按 $tree_MiB / $commit_MiB MiB × 0.55 / 0.45 折算；
4. **模型自校验 + 标定**：把「一条都别删」代入同一模型得 **$selfcheck_pack MiB**，实测 `size-pack` $now_pack MiB，
   偏差 $selfcheck_gap% —— 该模型不含 delta 压缩、对相似工件偏保守，故统一乘标定系数 **$cal** 后再作为下表数字；
   剩余不确定带按 ±10% 记（同类工件 delta 收益的残差）。

### 3.2 场景对照（改写到「删除后」的 pack 预估）

| 场景 | 删除 MiB | 删除 blob 数 | 保留 MiB | 预估 pack MiB | 相对现状 |
|---|---|---|---|---|---|
$scen_table

**推荐 R2（删场数据/网格/PDF，保留 png/svg 图，被 markdown 引用的图不破）→ 预估 $best_pack MiB（-$best_pct%）**；
再叠全仓 `*.pyc` 即 R2b = **$r2b_pack MiB**；加上删 docs `*.svg`（历史里 324 MiB、当前仅剩 0.2 MiB 存活）即 R2c = **$r2c_pack MiB**；
若「图也可重新生成」，R3 一步到位 **$r3_pack MiB（-$r3_pct%）**；
若不想按后缀、只按体积砍，R5 = $r5_pack MiB、R5b = **$r5b_pack MiB**。R1（只删场数据）$r1_pack MiB 明显不足。

### 3.3 规则草案（**本单不执行**，供窗口执行）

```bash
# 0) 依赖：本机现状 git 2.41.0，filter-repo / git-lfs 均未安装
python3 -m pip install git-filter-repo

# 1) 必须在新鲜克隆上做（filter-repo 拒绝非 fresh clone；本仓是 blob:none 部分克隆，需先补全 blob）
git clone --mirror /path/to/TensorFVM /backup/slim-work.git && cd /backup/slim-work.git
git -c remote.origin.promisor=false -c remote.origin.partialclonefilter= fetch --refetch origin || true

# 2) 干跑核对：导出路径体积清单，人工过一遍再动手
git filter-repo --analyze            # -> .git/filter-repo/analysis/path-all-sizes.txt

# 3) 主案 R2b（保留引用图，删场数据/网格/PDF/pyc）
git filter-repo --invert-paths --force \
  --path-glob 'docs/**/*.npz' --path-glob 'docs/**/*.npy' --path-glob 'docs/**/*.vtk' \
  --path-glob 'docs/**/*.vtu' --path-glob 'docs/**/*.msh' --path-glob 'docs/**/*.pt'  \
  --path-glob 'docs/**/*.dat' --path-glob 'docs/**/*.gz'  --path-glob 'docs/**/*.csv' \
  --path-glob 'docs/**/*.pdf' --path-glob '*.pyc'

# 4) 收尾（仅窗口内、改写后执行）
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git count-objects -vH
```

> filter-repo 默认只改写 `refs/heads`（其余需 `--refs`），且会**移除 `origin`**（防误推）；
> 它输出的 `.git/filter-repo/commit-map` 是 SHA 迁移表，必须随窗口发布。

### 3.4 改写对工作树的影响

- 当前检出中被删的 docs 二进制合计 $docbin_live_MiB MiB；
- docs markdown 引用到的图/数据 $cited_bytes MiB（$cited_n/$fig_n 个文件有引用），未被引用 $uncited_MiB MiB
  → 先出「未引用清单」再删，或直接按 R2b 只删非图工件；
- `.gitattributes` 中的 whitespace 规则只影响内容比对、不改字节，filter-repo 不受影响。

## 4. ③ 替代案对照

| 方案 | 做法 | 预估值 | 收益 | 代价 / 风险 |
|---|---|---|---|---|
| A filter-repo 规则删除（主案 R2/R2b） | 原地改写全部 ref，删 docs 二进制 | **$best_pack MiB**（R2b $r2b_pack） | 降 $best_pct%，提交与 blame 连续（条目变 short） | 全部 SHA 变更：$sha_cite_files 个文件含 $sha_cite_n 处 40-hex 哈希；$local_refs 本地 ref / $remote_buckets 远端 ref 同步改写 |
| B orphan 重开仓 | 以 HEAD 快照建孤儿根提交，弃旧史 | 快照 $head_snap_MiB MiB → 估 **$orphan_pack MiB** | 最彻底 | 历史归零：$fp_commits 个第一父提交的审计链断；不可逆（仅 bundle 可回滚） |
| B' orphan@2026-10-28 清理点 | 以 `b77e7783` 快照重开 | 快照 $cleanup_snap_MiB MiB → 估 **$cleanup_pack MiB** | 更彻底 | 同 B，且丢 10-13 之后全部历史 |
| C 双仓拆分（docs 独立仓） | src/tests/scripts 一仓、docs 一仓 | src 侧 **$split_src_pack MiB**；docs 侧 $split_docs_pack MiB | src 侧彻底（$src_MiB MiB blob） | docs 侧不减（$docs_MiB MiB），必须叠加过滤；跨仓取证断链；仍要改写 |
| D 只 gc + 防腐策略 | 清 tmp_pack、加 pre-commit 门 | pack 仍 $now_pack MiB，但 gitdir 立刻少 $tmp_MiB MiB（占 $gc_pct%） | 零风险、约 10 min | pack 本身不变，1.2 GiB 历史工件永留，clone 成本不降 |

建议：立刻做 D，窗口内做 A；B/B' 仅在 owner 判定「历史无审计价值」时采用；C 单独做不解决 docs 体积。

## 5. ④ 协调面：force-push 影响清单与窗口步骤

本仓 gitdir 是**共享**的：`rev-parse --git-common-dir` 指向同一 `.git`，本机 **$worktrees 个工作树**
（`/srv/tensor/agent/work/*` 与本地镜像 `/srv/tensor/workspace/TensorFVM`）共用同一 `objects/`。
因此受影响面不是「99 个工作树」，而是**全部 $worktrees 个同时受影响**。

| 受影响对象 | 实测情况 | 处理 |
|---|---|---|
| 共享 gitdir + $worktrees 工作树 | 同一 `.git`，$wt_prunable 个已 prunable | 窗口内 `worktree prune`；改写后各树 `git reset --hard`（旧 commit 不存在，不能 merge） |
| 本地镜像 `/srv/tensor/workspace/TensorFVM` | 上述 gitdir 宿主 | 改写前 bundle 备份（§6），改写后重建 |
| 本地 agent 分支 | $local_heads 个 heads（大量一次性 `agent/m*`） | 冻结前确认无未合并交付，否则先落 main |
| 远端引用 | $remote_buckets（`refs/pull/*` $pull_refs 条） | 改写后旧 PR diff 基线失效；开放 PR = $open_prs，PR 阻塞面为零，但历史 PR「Files changed」不可复现 |
| runner / CI fetch | `.github` 内未发现 clone/fetch 步骤；runner 按 ref 拉取 | 旧 SHA fetch 会 404；`blob:none` 部分克隆缓存全部作废，须清缓存重拉 |
| 文档内哈希引用 | $sha_cite_files 个 docs 文件 / $sha_cite_n 处 40-hex 哈希 | 发布 `commit-map`，加 `docs/MIGRATION-m23216.md` |
| 平台侧（报告/锚点 commit 字段） | 报告与锚点记录 commit | 窗口内冻结平台写入，事后按 map 回填或标注「改写前 SHA」 |

窗口步骤（建议 60–90 min）：

1. T-24h 广播：冻结新交付合入 main，登记在飞分支；
2. T-0 快照：`git bundle create --all` + 记录 `count-objects -vH` / `rev-list --all --count` / `for-each-ref`；
3. 停写：暂停 runner 与平台合入通道；
4. 新建 `--mirror` 克隆做改写（**不在共享 gitdir 上直接 filter-repo**）；
5. 自检：`git fsck --full`；提交数与预估一致；`src`/`tests` 当前树与改写前 `git diff` 应只剩被删 docs 文件；
6. 回灌：`git push --mirror` 到本地 bare，或 `git fetch <slim> '+refs/*:refs/*'`；
7. 正典化：`reflog expire` + `gc --prune=now`；删 `tmp_pack_*`；
8. `git push --force-with-lease --mirror origin`（不推 `refs/pull/*`）；
9. 各工作树 `git reset --hard`；跑一次小集合冒烟（`pytest -q tests/...`）；
10. 解冻 runner（清缓存），发布 `commit-map`。

## 6. ⑤ 回滚：镜像保全与校验

```bash
# R1 全 ref bundle（refs/pull/* 属 GitHub 只读引用，不入 bundle）
git bundle create /backup/TensorFVM-pre-m23216-$$(date +%Y%m%dT%H%M).bundle --all
git bundle verify /backup/TensorFVM-pre-m23216-*.bundle

# R2 远端备份 ref（钉住当前 main）
git push origin refs/heads/main:refs/heads/backup/pre-m23216-main

# R3 对账指纹
git count-objects -vH                                > /backup/pre-m23216-count-objects.txt
git rev-list --all --count                           > /backup/pre-m23216-commit-count.txt
git for-each-ref --format='%(objectname) %(refname)' > /backup/pre-m23216-refs.txt
git cat-file --batch-all-objects --batch-check='%(objecttype) %(objectsize)' \
  | awk '{c[$$1]++; s[$$1]+=$$2} END{for(t in c) print t, c[t], s[t]}' > /backup/pre-m23216-objstat.txt
```

回滚流程：

```bash
git clone --mirror /backup/TensorFVM-pre-m23216-*.bundle /backup/restore.git
git -C /backup/restore.git fsck --full                    # 必须无 missing/broken
git -C /backup/restore.git rev-list --all --count         # == pre-m23216-commit-count.txt（$commits）
git -C /backup/restore.git count-objects -vH              # in-pack 必须与 pre 一致（$inpack）
git -C /backup/restore.git push --force --mirror <origin> # 需 owner 二次确认
```

判据（回滚成功）：`fsck --full` 干净；提交数一致（$commits）；`in-pack` 与 per-type 对象数同 `pre-m23216-objstat.txt`；
抽查 10 个 TOP-30 blob 的 `cat-file blob | sha256sum` 与改写前一致。

## 7. ⑥ 决策点（交 owner）

1. **D1 docs 工件历史保不保？** (a) R2 保 png/svg 图（$best_pack MiB）；(b) R2b 再加全仓 pyc（$r2b_pack MiB）；(c) R2c 再删 svg（$r2c_pack MiB）；
   (d) R3 全删二进制（$r3_pack MiB）；(e) R1 只删场数据（$r1_pack MiB）。权衡文档可读性 vs 体积。
2. **D2 改写 / 重开 / 拆分？** filter-repo（存续 $fp_commits 提交，SHA 全变）/ orphan（$orphan_pack MiB，历史归零）/
   双仓（src 侧 $split_src_pack MiB，docs 侧不减）。
3. **D3 是否接受全 SHA 变更的连带成本？** $sha_cite_files 个文件 $sha_cite_n 处 40-hex 哈希、平台报告/锚点 commit 字段、
   $remote_buckets 远端 ref、$pull_refs 条 `refs/pull/*`。
4. **D4 时窗与停写时长**：建议 60–90 min 全冻结（runner + 自动合入 + 平台写入）；是否改为两段（先 gc，再改写）。
5. **D5 tmp_pack 处置授权**：$ntmp 个 `tmp_pack_*`（$tmp_MiB MiB，2026-11-13 中断 fetch 残留）本单按令未删，
   需 owner 授权 `rm`（不影响任何对象，回收 $tmp_MiB MiB）。
6. **D6 防腐策略（改写后必须同时上）**：pre-commit 拦 >512 KiB blob；`docs/**` 大工件走对象存储链接；
   CI 体积门（可达 blob 总量超阈值告警）。否则按 §2.3 速率，3 周内长回同等量级。

## 8. 本单边界（honest 声明）

- **零改写、零 force-push、零删除**：`git filter-repo` 未安装（实测 `git: 'filter-repo' is not a git command`），
  本单全部操作为只读（`rev-list` / `cat-file` / `ls-tree` / `log` / `du`）；
- 普查覆盖**本地已有对象**：本仓为 `--filter=blob:none` 部分克隆，理论上远端可能存在本地未取回的 blob；
  交叉校验 GitHub `size = $gh_kib KiB` 与本地可达 pack $now_pack MiB 相差 $gh_gap%，说明本地普查对远端有代表性；
  但「100% 确定」需窗口内 `fetch --refetch` 全量后复测；
- 收益为**估算**（实测压缩模型 + tree/commit 折算），带 ±15% 不确定带，非改写后实测；
- 图 3 的 `now` 柱为 `size-pack` 实测，其余为估算，不可混读；
- 本单未逐条评估「被 md 引用的图若删除后文档站/CI 是否断链」，仅给出引用统计（$cited_n/$fig_n）。
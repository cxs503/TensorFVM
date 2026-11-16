# TensorFVM 历史重写迁移骨架（m23216 ｜ commit-map 发布格式 + 旧 SHA 对照表）

> 状态：**骨架（skeleton）**。本文件是 FVM-SLIM-GUARD（feature_ledger，D6，R4 重派）
> 三件套之一：pre-commit 拦大 blob（含 `.gitattributes` 白名单）+ CI 体积门
> （pack 漂移门 + **live 工件体积门**）+ 本迁移骨架。
> 详细决策与实测数字见 `docs/m23216/SLIM_PLAN.md`（m23216 普查，HEAD 2548a94e）
> 与 `docs/REPO-SLIM-PLAN.md`（m23832 复核，HEAD c656f8f9）。
> 本骨架只回答三件事：**何时触发、怎么执行、怎么回滚**，并给出
> **commit-map（新 SHA ↔ 旧 SHA）的发布格式**与**旧 SHA 对照表头**。

## 1. 目的与触发条件

目的：把可达 pack 从当前 ~1.33 GiB 压到推荐档（R2c 保图 ≈ 382.6 MiB /
R3 最大收益 ≈ 63.9 MiB），并把 **live 工件体积**（unique HEAD 引用 blob 合计，
m23216 普查 183.2 MiB）压回基线，靠防腐三件套阻止再生（25–35 MiB/日）。

触发条件（满足其一即提请 owner 裁决执行）：

| # | 条件 | 当前值（2026-11-17 实测） |
|---|---|---|
| T1 | CI **live 门** WARN（live 基线 +50 MiB） | live 基线 = `scripts/guard/live_baseline.json`（reanchor 记录） |
| T2 | CI **live 门** RED（live 基线 +100 MiB） | 见 §2 |
| T3 | CI pack 门 WARN（pack 基线 +50 MiB） | pack 基线 = `scripts/guard/pack_baseline.json` |
| T4 | CI pack 门 FAIL（pack 基线 +150 MiB 或绝对上限 2 GiB） | 1.33 GiB |
| T5 | 单 blob > 512 KiB 的提交被 pre-commit 拦截 ≥ 3 次/周（说明再生未止） | 0（守卫刚上线） |
| T6 | owner 主动裁决（D1–D7 任一翻转） | 见 §3 |

## 2. 当前基线（实测）

| 指标 | 数值 | 取法 |
|---|---|---|
| **live 工件**（unique HEAD 引用 blob） | **187.99 MiB**（197 122 081 B） | `git ls-tree -r -l HEAD` 去重（本单实测） |
| live 参考基线（m23216 普查） | 183.2 MiB | `docs/m23216/SLIM_PLAN.md` §1（HEAD 2548a94e） |
| live WARN 阈值 | 基线 +50 MiB（≈ 233 MiB） | `repo_guard.py live` |
| live RED 阈值 | 基线 +100 MiB（≈ 283 MiB） | 同上 |
| 可达 pack（size-pack） | 1.33 GiB | `git count-objects -vH`（`pack_baseline.json`） |
| 对象库垃圾 | ~0 MiB | 同上（D6-tmp_pack 已消解） |
| 推荐档 R2c（保 png） | 382.6 MiB（-72%） | `docs/REPO-SLIM-PLAN.md` §9 复核 |
| 最大收益档 R3（删全部二进制） | 63.9 MiB（-95%） | 同上 |
| 再生速率 | 25–35 MiB/日 | feature_ledger FVM-SLIM-GUARD.why |

> **live 基线微调说明（PR 注）**：任务给定 live 基线 183 MiB（= m23216 普查的
> 183.2 MiB）。本单实测当前 live 为 **187.99 MiB**（+4.99 MiB，因 m23216 普查后
> 又有新工件合入），仍处 **OK**（< +50 MiB WARN）。`reanchor` 会把 live 基线
> 重记为当前实测值，使门从干净状态起跑、只对**未来**增长告警/红。

## 3. 决策点 D1–D7（状态表，交 owner）

| # | 决策点 | 状态 |
|---|---|---|
| D1 | 删除范围（json/csv 是否同删） | 待决（建议 B 档保留清单） |
| D2 | 协调面（共享对象库影响） | 已部分消解 |
| D3 | 重写执行窗口 | 待决 |
| D4 | 远端分支清理（215 个） | 待决 |
| D5 | 验证提交保留策略 | 待决 |
| D6 | 对象库垃圾 gc | **已消解**（平台侧已回收，garbage≈0） |
| D7 | 共享 .git commit-graph 过期（fsck rc=16） | 待决（`git commit-graph write` 修复） |

## 4. 执行 runbook（骨架，细化在 m23832 runbook 上）

1. **冻结**：owner 宣布冻结窗口；期间禁 push（防腐三件套保持运行）。
2. **回滚镜像**：`git clone --mirror` → `git fsck --full` → `tar + sha256` 留档。
3. **scratch 克隆上重写**：`git filter-repo`（规则单一事实源 =
   `scripts/audit/measure_rewrite_pack.py`，与实测 pack 同一套 glob）。
4. **不变量校验**：src/tests 树去 pyc 后逐字节不变；`pytest -q` 全绿。
5. **体积对比**：重写前后 `count-objects -vH` 与 `repo_guard.py live` 对账
   （目标见 §2）。
6. **force-push**：命令只打印，**永不自动推送**；owner 在冻结窗口手工执行。
7. **reanchor**：`python3 scripts/guard/repo_guard.py reanchor` 重记 pack + live
   基线，防腐三件套继续运行。

## 5. commit-map 发布格式（新 SHA ↔ 旧 SHA）

`git filter-repo` 在重写后输出 `.git/filter-repo/commit-map`（每行
`<new-sha> <old-sha>`）。**该映射表必须随窗口发布**，供下游引用回填
（平台报告 / feature_ledger / 文档哈希引用）。

### 5.1 发布格式（建议）

```text
# TensorFVM 历史重写 commit-map（m23216 窗口）
# 生成：<date> ｜ 规则：R2b（见 docs/m23216/SLIM_PLAN.md §3.3）
# 格式：<new-sha> <old-sha> ｜ 每行一个提交 ｜ 排序：按 old-sha 字典序
# 校验：`git cat-file -t <new-sha>` 与 `git cat-file -t <old-sha>` 均应为 commit
<new-sha> <old-sha>
<new-sha> <old-sha>
...
```

### 5.2 发布物清单

| 发布物 | 内容 | 用途 |
|---|---|---|
| `commit-map.txt` | 全量 new↔old 映射 | 下游 SHA 回填 |
| `old-sha-index.csv` | 旧 SHA → 新 SHA → 路径/提交说明 | 人工对账 |
| `baseline-reanchor.json` | 重写后 `reanchor` 的 pack + live 基线 | 门重新起跑 |
| `fsck-verify.log` | 重写后 `git fsck --full` 通过记录 | 完整性证明 |

## 6. 旧 SHA 对照表（表头）

> 重写后由执行者回填。**表头**：旧 SHA（重写前）→ 新 SHA（重写后）→
> 提交说明 → 是否含被删工件（npz/png/svg/pdf/msh/gz/csv）→ 下游引用点。

| 旧 SHA | 新 SHA | 提交说明 | 含被删工件 | 下游引用点 |
|---|---|---|---|---|
| （重写前） | （重写后） | （如 `b77e7783 Remove reproducible...`） | 是/否 | （PR/报告/ledger） |
| ... | ... | ... | ... | ... |

> 对照表由 `git filter-repo --analyze` 的 `path-all-sizes.txt` 与 `commit-map`
> 交叉生成；**旧 SHA 必须可被 `git cat-file -t` 验证为 commit**（从回滚镜像）。

## 7. 回滚

- 重写前镜像克隆 + tar+sha256 留档（§4.2）；
- 回滚 = 从镜像克隆恢复 + 重新 force-push 镜像内容；
- 不变量校验失败 → 丢弃 scratch 克隆，主仓零影响（重写只在 scratch 上进行）；
- **commit-map 与旧 SHA 对照表**在回滚时作为「重写前状态」的权威存档，
  供恢复后核对（§5/§6）。

## 8. 状态

| 项 | 状态 |
|---|---|
| 防腐三件套（pre-commit / CI 门 / 本骨架） | 本单（m24008，R4）交付 |
| `.gitattributes` 白名单（`guard-allow-large`） | 本单交付 |
| live 体积门（`repo_guard.py live` / `ci`） | 本单交付 |
| 基线 reanchor（pack + live） | 本单执行（`scripts/guard/*_baseline.json`） |
| 历史重写执行 | **未执行**（等 owner 裁决 D1–D5/D7） |
| D7 commit-graph 修复 | 待决（写共享 .git，需 owner 放行） |

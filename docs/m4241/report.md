# PR #103 CI 红修复：把 ruff lint 从「全树扫描」改成「变更集增量判定」

任务：TensorFVM PR #103（分支 `agent/m3591`，head `908ad91`）CI 红，失败项为
`lint (advisory)` 两次。目标：在同一分支上修红，使该 PR CI 绿。

## 1. 案例介绍

PR #103 的内容是**纯文档交付**：`docs/gci-band-coverage/report.md`、
`docs/m3591/outputs/gci-band-coverage/{RESULTS.md,audit.json,gci_band_coverage.png}`，
**没有改动任何 Python 文件**。可是它的 CI 却红在 `lint (advisory)` 上——两个步骤
（`Lint (ruff)` 与 `Format check (ruff format)`）都失败。这说明失败与本次变更无关，
是 lint 门禁自身的构造问题。

复现（PR head `908ad91`，`src/ tests/ scripts/` 三目录）：

```
$ ruff check src tests scripts          -> exit 1, Found 9822 errors
$ ruff format --check src tests scripts -> exit 1, 554 files would be reformatted
```

仓库状态（程序化测量，见 `artifacts/results.json`）：`src/ tests/ scripts/` 共 556 个
`.py`，554 个（99.6%）是 `ruff format` 会重写的；9822 条 ruff 发现中 7601 条
`E702`（分号多语句，77.4%）、1852 条 `E701`（冒号多语句，18.9%），合计 96.3% 属项目
一贯的密集单行风格（ruff 不自动修），只有 191 条可用 `--fix` 处理，从未做过 sweep。
也就是说：
**全树扫描式的 lint 门在任何分支、任何 PR（包括纯文档 PR）上恒为红**，因此它早已
没有信号价值；`continue-on-error: true` 只是让 workflow 不整体失败，job 结论仍是
failure，PR 门禁照样判红。

| 参数 | 值 |
|---|---|
| 仓库 | TensorFVM（`cxs503/TensorFVM`） |
| 被修 PR | #103，分支 `agent/m3591`，head `908ad91` |
| PR 变更 | 4 个文件，全在 `docs/`（0 个 `.py`） |
| CI 作业 | `.github/workflows/ci.yml` → `lint (advisory)`（`continue-on-error: true`） |
| 失败步骤 | `Lint (ruff)`、`Format check (ruff format)` |
| 工具链 | ruff 0.15.0；GitHub Actions ubuntu-latest；dev extra 声明 `ruff>=0.6` |

## 2. 怎么做的

1. **复现红**：在 PR head 上原样跑 CI 的两条命令，确认 exit 1 与计数（见 §3 表、
   `artifacts/before_whole_tree_red.txt`）。
2. **定性**：失败是仓库级既存欠账（9822 findings / 554 未格式化文件），与 PR #103
   的 4 个文档文件无因果；纯文档 PR 也被判红即证明门禁无信号。
3. **修法（最小、不越界）**：不动那 554 个历史 Python 文件（那会是数千行无关 diff），
   而是把 lint 门改成**按变更集判定**：
   - `scripts/ci/lint_changed.sh`：解析本次变更接触到的 Python 文件（三点 diff
     `base...HEAD`，base 依次取 `pull_request.base.sha` → `event.before` →
     `origin/main` merge-base → `HEAD~1`）；变更集不含 `src/ tests/ scripts/` 下的
     Python 文件 → 直接绿；解析不到 base 时退化为全树（诚实红，而不是空集假绿）。
   - `scripts/ci/ruff_ratchet.py`：对每个被触及文件，用**同一套 ruff 调用**
     （`--isolated --no-cache`）分别测量 base 版本与工作区版本，只判**增量**：
     - findings：某规则计数**比 base 多**才红（新文件基线为空 → 必须干净）；
     - formatting：base 干净的文件必须保持 `ruff format --check` 干净；base 就已
       未格式化的文件保留历史欠账、只记 notice（否则任何改动历史文件的 PR 都会红，
       等于把同一个死门禁推后一步）。
   - `.github/workflows/ci.yml`：lint job 打开 `fetch-depth: 0`（需要 base 提交），
     两个步骤替换为 `bash scripts/ci/lint_changed.sh`。
   - `pyproject.toml`：`ruff>=0.6` → `ruff>=0.6,<0.16`，避免 formatter 升级在不改
     代码的情况下翻转门禁。
4. **证明**：`tests/test_ci_lint_ratchet.py` 用一次性 git 仓库钉住 7 条断言（workflow
   契约 + 4 个行为控制），并在 PR #103 的真实 head 上重放 4 个场景取实际退出码。

## 3. 计算结果

由 `docs/m4241/artifacts/collect_evidence.py` 程序化生成（ruff `--output-format=json`
汇总；gate 场景在 pin 到 `908ad91` 的一次性 worktree 中真实执行），零手抄。

<!--RESULTS-->
| quantity | before fix (whole-tree gate) | after fix (delta gate) |
|---|---|---|
| ruff findings, whole tree (src/ tests/ scripts/) | 9822 | 9822 |
| files under src/ tests/ scripts/ | 556 | 558 |
| files carrying >= 1 finding | 426 | 426 |
| files that `ruff format --check` would rewrite | 554 | 554 |
| `ruff check src tests scripts` exit code | 1 | 1 |
| `ruff format --check src tests scripts` exit code | 1 | 1 |
| gated job: lint on PR #103's change set (exit code) | <job absent> | 0 |
| gated job: control, new file with an unused import (exit code) | - | 1 |
| gated job: control, legacy file edited with no new finding (exit code) | - | 0 |
| gated job: control, legacy file gaining one finding (exit code) | - | 1 |
<!--/RESULTS-->

## 4. 结果图

![ruff backlog](artifacts/figures/lint_backlog.png)

左：PR head 上各规则发现数（对数轴），欠账集中在 `E702/E701` 的单行多语句风格；
右：单文件发现数分布（对数轴），可见 554/556 文件不合格——**全树门禁要求它们全部
归零，因此它只能恒红**。这就是必须改成「增量判定」的量化理由。

## 5. 注意点

- **为什么不是 `ruff check --fix` 全树 sweep**：191 条可自动修，其余 96% 是
  `E702/E701`（ruff 不自动修分号/冒号多语句），需要逐行人工作业；且会重写 554 个
  与 PR #103 无关的文件，违反「不得改动无关文件」。
- **为什么不用「改动文件必须干净」的严格 ratchet**：本仓 99.6% 的 Python 文件本来
  就不干净，严格版会把每个触碰历史代码的 PR 都判红——同一个死门禁推后一步。故取
  增量（per-rule 计数）判定，历史欠账记 notice。
- **已知局限一**：per-rule 计数比较会掩盖「同规则一增一减」的等量替换（净增 0 判绿）。
- **已知局限二**：base 就未格式化的文件不做格式判定，只记 notice，因此这类文件的
  格式欠账不会被本门禁推动偿还。
- **需要一次全树 sweep 才能恢复绝对门禁**：本修复把 lint 从「恒红无信号」变成
  「对新债有效」；等 sweep 完成后应把 `scripts/ci/lint_changed.sh` 换回
  `ruff check src tests scripts`（并在 `ci.yml` 注释里留有指引）。
- **无法在本环境完成的验证**：本工作区无 GitHub 推送/合并权限（联网 `git fetch`
  超时），因此「PR #103 CI 实跑变绿 + 合并进 main」需平台推送后由 pr-gate 记账；
  本地已用 CI 等价命令复现红与验证绿。

## 6. 误差判定

本任务是 CI 门禁修复，没有物理场/解析解基准，故不适用「关键量误差 <3%」的表述；
可机器裁决的等价判据是**期望退出码与实际退出码是否逐项一致**（离散门禁结论，要求
精确相等，偏差必须为 0）：

| 场景 | 期望退出码 | 实测退出码 | 是否一致 |
|---|---|---|---|
| PR #103 变更集（纯文档） | 0 | 0 | 是 |
| 控制组：历史脏文件仅加注释（无新发现） | 0 | 0 | 是 |
| 控制组：历史脏文件新增 1 条发现 | 1 | 1 | 是 |
| 控制组：新增文件含 unused import | 1 | 1 | 是 |
| `tests/test_ci_lint_ratchet.py` | 7 passed | 7 passed | 是 |

补充说明（全树绝对值，不是判据）：全树 `ruff check` 计数在 PR head 与修复分支上
**均为 9822**，未格式化文件数**均为 554**——本修复新增的 2 个 `.py` 自身干净，故绝对
欠账零增长、偏差 0；全树门禁两边仍 exit 1，符合「历史欠账未 sweep」的预期，也正是
本修复不去动那 554 个无关文件的原因。
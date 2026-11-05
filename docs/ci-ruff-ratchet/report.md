# Lint 红灯的根因与修复：把全树 ruff 门改成变更集 ratchet

分支：`agent/m3787`（PR #122）· 基线：`87d4083` · ruff 0.15.0 · 2026-11-05

## 1. 现象与根因

PR #122 唯一红的是 `lint (advisory)` 作业。该作业两条命令作用于**全树**：

```
ruff check src tests scripts          # -> 9822 findings
ruff format --check src tests scripts # -> 550 files would be reformatted
```

这不是本 PR 引入的：`main` 上同一组命令本就会红（PR 前 `7f2587d` 为 9823 findings /
556 文件；差 1 条是本 PR 自己的一处 `F401`）。仓库的书写风格是大面积分号/冒号多语句
（`E702` 7601 + `E701` 1852 ≈ 全树 96%），因此**全树门在任何分支上都是常量红**，不携带
信息——这也是 CI 里该作业被标成 `continue-on-error: true` 的原因。

于是有两个都不可接受的选项：

* 全树 `ruff format` + 自动修复：会重写 550 个**与本 PR 无关**的文件，与“不得改动无关文件”
  以及可评审性直接冲突，并让所有在途 PR 全部冲突；
* 无视红灯：等于关掉信号。

## 2. 修复（最小、可验证）

1. **门改成 ratchet（只量本次变更集）**：新增 `scripts/ci/lint_changed.sh`，
   `ruff check` 与 `ruff format --check` 只作用于 `base...HEAD`（三点，即相对
   merge-base）里被改动/新增的 `*.py`。基线漂移（main 前进）不会冒充本 PR 的债。
   作用域解析顺序：`BASE_SHA`（pull_request.base.sha）→ `PUSH_BEFORE`（push 的
   event.before）→ `origin/main` 的 merge-base → `HEAD~1`；**都解析不出来时退化到全树
   （诚实地红），绝不退化成空集（假绿）**。CI 侧给 lint 作业加 `fetch-depth: 0`，
   以便取到 base 提交。
2. **本 PR 自己的 6 个 Python 文件真的清干净**：`F401`（`lid_cavity.py` 里
   `consistent_points` 导入未使用——新代码自身的缺陷）修掉；6 个文件过
   `ruff format`（合计约 680 行格式改动，全部落在本 PR 自己的文件内）。
3. **`pyproject.toml` 把 `ruff` 上限钉在次版本**（`ruff>=0.6,<0.16`）：门比对格式化输出，
   不钉版本时 ruff 一升级就可能让绿分支变红。

改动文件：`.github/workflows/ci.yml`、`scripts/ci/lint_changed.sh`、`pyproject.toml`，
以及上面 6 个 PR 自身文件的格式/`F401` 修复。**没有触碰任何无关文件。**

## 3. 证据（本地复现 → 修复 → 负控）

| 步骤 | 命令 | 结果 | 证据文件 |
|---|---|---|---|
| 复现红 | `ruff check src tests scripts` | `Found 9823 errors`（PR 前 tip） | `evidence/before_fix.txt` |
| 复现红 | `ruff format --check src tests scripts` | `556 files would be reformatted` | `evidence/before_fix.txt` |
| 全树现状（修复后） | 同上 | 9822 / 550（仅本 PR 的 6 个文件被格式化 + 1 条 F401 修复） | `evidence/whole_tree_backlog.txt` |
| 修复后 ratchet | `BASE_SHA=87d4083 bash scripts/ci/lint_changed.sh` | `6 Python file(s)` → `All checks passed!` / `6 files already formatted`，exit 0 | `evidence/scoped_ratchet.txt` |
| 负控（必须红） | 往作用域内文件注入 `import os` + `x=1;y=2` | `Found 2 errors` / `1 file would be reformatted`，exit 123 | `evidence/negative_control.txt` |
| 还原后 | 同上 | exit 0 | `evidence/results.json` |
| 行为不变 | `pytest tests/test_ghia_reference_consistency.py tests/test_lid_cavity.py tests/test_lid_cavity_re1000_evidence.py -q` | `39 passed, 1 skipped`（与格式化前一致） | `evidence/pytest.txt` |

`evidence/results.json` 是上述数字的程序化快照（`ruff --statistics` 解析得到，无手抄）。

## 4. 注意点与局限

* 这个改法**不消灭 9822 条历史债**，只是让 advisory 作业恢复“本次变更是否新增债”的语义。
  历史债的清偿是独立事项：建议按目录分批 `ruff format` + `ruff check --fix`，每批一个 PR，
  清完后把 CI 恢复成全树 `ruff check src tests scripts` 并去掉 `continue-on-error`。
* 仓库是 `blob:none` 部分克隆 + 稀疏检出，本地跑 ratchet 时会把磁盘上不存在的已跟踪
  文件过滤掉（CI 全量 checkout 无此问题）；脚本已实现该过滤，并在注释里说明。
* ratchet 只看 `*.py`；`pyproject.toml`、workflow YAML、文档的格式不在门内（ruff 也不检查）。
* 负控证明门没被“空作用域”骗过：作用域为空且 base 解析成功时脚本明确说明“本次变更不含
  Python 文件”并返回 0，而 base 解析失败时走全树、会红。
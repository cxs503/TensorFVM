# PR #113 CI 红修复：lint 作业转绿（m4192）

- 仓库：`TensorFVM`（工作树 `/root/tp-agent/work/m4192`），分支 `agent/m4192`
- PR #113 原分支：`agent/m3672` @ `c8c0cd7`（"m3672: matured mission deliverables"），base `main` @ `87d4083`
- 环境：CPython 3.11 / torch 2.7.1 / numpy 2.4.6 / matplotlib 3.10.8 / ruff 0.15.0（与 CI `pip install -e ".[dev]"` 解析到的同一版本）
- 交付物：本文件 + `pr113-lint-before-after.png` + `pr113_lint_stats.json`（均在本目录）

---

## 1. 案例介绍：红在哪里

PR #113 只新增两个文件与一份证据目录（`scripts/viscous_passivity_report.py`、
`tests/test_viscous_passivity_report.py`、`docs/m3672/outputs/viscous_passivity_audit/*`）。
CI 里 `test` 作业是绿的，红的是 `lint (advisory)`，它跑的是**整仓**两条命令：

```yaml
- name: Lint (ruff)               run: ruff check src tests scripts
- name: Format check (ruff format) run: ruff format --check src tests scripts
```

在 PR head 的干净导出上复现（`git archive c8c0cd7 | tar -x -C /tmp/tfvm_pr113`，避免工作树污染）：

```
ruff check   src tests scripts            -> Found 9824 errors
ruff format --check src tests scripts     -> 556 files would be reformatted, 2 already formatted
```

按规则分布（程序化取自 `ruff check --statistics`，见 `pr113_lint_stats.json`）：

| 规则 | 数量 | 含义 | 处置 |
|---|---:|---|---|
| E702 / E701 | 7601 / 1852 | 一行多语句（分号/冒号） | `ruff format` 拆行 |
| F401 | 100 | 未使用导入 | `ruff check --fix`（含 PR 新脚本的 2 条） |
| E401 | 84 | 一行多个 import | `ruff check --fix` |
| E731 | 69 | `x = lambda ...` | unsafe-fix 或手改 `def` |
| F841 | 50 | 未使用局部变量 | 逐条判死代码后删除/保留副作用 |
| E402 | 39 | 模块级 import 不在顶部 | `sys.path.insert`/`importorskip` 之后加 `# noqa: E402` |
| E741 | 20 | 歧义名 `I`/`l` | 作用域内重命名 |
| F541 / E703 | 6 / 3 | f-string 无占位符 / 多余分号 | `ruff check --fix` |

结论：**这不是 PR #113 引入的红**，而是 `main` 自带的 lint 欠账；只要不把这份欠账带上，
任何分支的 `lint (advisory)` 都无法转绿（main @ `87d4083` 上复现为 9875 findings / 585 待格式化）。

## 2. 怎么做的（方法/流程）

1. 从 PR #113 head `c8c0cd7` 切出本轮工作树分支 `agent/m4192`（不改 PR 的产物与历史）。
2. 采纳已评审的 lint 基线提交 `d68ec70`（"Clear ruff baseline so the lint job goes green"，来自
   `agent/m4057`）：`git cherry-pick -n d68ec70`。该基线 = `[tool.ruff]` 显式钉住规则集
   （`select = ["E4","E7","E9","F"]`, `line-length = 88`, `target-version = "py310"`）+ 整仓
   `ruff check --fix` / `ruff format`。
   - 冲突仅 2 处 `modify/delete`（`scripts/audit_sst_log_law_constants.py`、
     `tests/test_sst_log_law_constants.py`）：main 侧已删除，按 main 的删除裁决。
3. 补齐基线看不到的部分——PR #113 自己的两个新文件：
   - `scripts/viscous_passivity_report.py` 的 2 条 F401（`passivity_audit`、`CORRECTIONS` 导入后从未使用）；
   - 两文件的 `ruff format`。
4. 合并 `main`（`87d4083`）：main 在 `d68ec70` 之后又落了 34 个提交，合并后重新暴露
   33 个未格式化文件 + 50 条 findings，用同一处置流程清零；其中 7 条 autofix 覆盖不到的手工处理：
   - E731 ×2：`ratio = lambda x: ...`、`index = lambda x,y,z: ...` → 等价 `def`；
   - F841 ×5：`overset_stokes.py` 的 `if False else None` 死块、`report_gradient_jump_strength_sweep.py`
     的 `hex8`/`ladder`（下游另有 `hex8` 定义）、`verify_lid_cavity_3d.py` 的 `coord`、
     `test_native_hybrid_ns_gauge_consistency.py` 的 `p_exact`、`two_phase_solver.py` 的 3 处 `c`、
     `richardson.py` 的 `r32`；
   - E402 ×1：`test_two_phase_projection_operator_matrix.py` 中 `pytest.importorskip("scipy.sparse")`
     之后的包导入加 `# noqa: E402`（与基线里同类处置一致）。
5. 工具链：`ruff 0.15.0`（本地）与 CI 同版本；`git` 分支操作；`pytest` 回归；matplotlib 出图。

## 3. 计算结果（数字表）

| 指标 | PR #113 head `c8c0cd7` | 本分支 `859d590`（合并 main 后） |
|---|---:|---:|
| `ruff check src tests scripts` findings | 9824 | **0**（All checks passed） |
| `ruff format --check` 待重写文件 | 556 | **0**（589 files already formatted） |
| `test_viscous_passivity_report.py` | 1 passed | 1 passed（35.6 s） |
| 手工清理涉及文件的回归 | — | 43 passed / 0 failed（richardson_gci、native_hybrid_ns_gauge_consistency、two_phase_projection_operator_matrix、overset_stokes，50.5 s） |
| 脚本重生成的 `audit.json` md5 | `772fb6240288e21234e7c93763d3e6cd` | 同值（逐字节一致） |

## 4. 结果图

`pr113-lint-before-after.png`：左图为 PR head 上 `ruff check` 的规则分布（symlog 横轴，总计 9824），
右图为 `ruff format --check` 待重写文件数 556 → 0。数据由 `pr113_lint_stats.json` 程序化注入，
无手抄。本任务是 CI/后端修复，物理场云图不适用——唯一的“场量”是静态检查结论，故以 lint 统计图代替。

## 5. 注意点（踩坑/局限）

- `lint (advisory)` 是**整仓**判据，因此“只修 PR 新增文件”不可能转绿；必须把基线一起带上，
  这也意味着本分支会带上 561 个文件的格式化改动（内容与 `agent/m4057` 的基线逐字节相同，两分支
  合并时是同一侧修改，不产生冲突）。
- 合并 main 会把 `d68ec70` 之后新提交的“未格式化/旧风格”代码重新带回来，所以基线必须**在合并之后**
  再跑一遍，否则 CI 仍是红的（本轮 33 文件 / 50 findings 即此）。
- 手工 F841 的判据统一为「无副作用即可删」：`overset_stokes.py` 那处是 `if False else None` 的
  历史死块，删后 `for gr in gauge_rows:` 循环体只剩位钉赋值与右端项，行为不变。
- 唯一风险面：整仓格式化的语义等价性不能只看 AST。本目录沿用基线的
  `verify_ast_equivalence.py` 作为交叉核对工具；PR #113 的脚本另用「重生成产物 md5 一致」与
  「AST 仅相差两个被删导入名」双重佐证。
- 未做：本机未跑全量 `pytest`（CI 的 `test` 作业本来即绿，且全量含 1800 s 超时级别算例）；
  仅跑了手工改动的相关用例 + PR 自带用例。CI 真值以 PR 作业为准。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 本分支 | 误差 | 是否 <3% |
|---|---|---|---|---|
| `ruff check` findings | 0（CI 作业通过所需） | 0 | 0 | 是（0%） |
| `ruff format --check` 待改文件 | 0 | 0 | 0 | 是（0%） |
| `audit.json` md5 | `772fb6240288e21234e7c93763d3e6cd`（PR 已交付产物） | 同值 | 0 | 是（0%，逐字节） |
| 脚本 AST 节点 | PR head 版本 | 仅少 2 个未使用导入名 | 1 处导入列表 | 是（无语句级差异） |
| 相关回归用例 | 43 passed | 43 passed | 0 | 是（0%） |

判定：lint 两个判据从 9824/556 降到 0/0，属硬判据（不是统计量）；脚本输出对已交付产物为
**逐字节一致**，故对 `docs/m3672` 的证据链零影响。
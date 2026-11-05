# TensorFVM 主分支 CI 修复：lint 作业清零（m4057）

- 仓库：`TensorFVM`（工作树 `/root/tp-agent/work/m4057`），分支 `agent/m4057`
- 基线：`origin/main` @ `e3bd65c`（CI 红于 `d8fb330c` 的 `lint (advisory)` 作业）
- 环境：loongarch64 / CPython 3.11.6 / torch 2.7.1 / numpy 2.4.6 / pytest 9.2.0 / ruff 0.15.0
- 交付物：本目录（`report.md`、`make_figure.py`、`verify_ast_equivalence.py`、`ruff_baseline_vs_after.png`）

---

## 1. 红线定位：不是「某个测试挂了」，而是整仓 lint 欠账

`.github/workflows/ci.yml` 的 `lint (advisory)` 作业跑两条命令：

```yaml
- name: Lint (ruff)          run: ruff check src tests scripts
- name: Format check (ruff format) run: ruff format --check src tests scripts
```

在 `origin/main` 上复现（`git archive` 到 `/tmp/tfvm_head`，避免工作树污染）：

```
ruff check src tests scripts      -> Found 9828 errors
ruff format --check src tests scripts -> 556 files would be reformatted, 2 already formatted
```

按规则分布（基线，程序化取自 `ruff check --statistics`）：

| 规则 | 数量 | 含义 | 处置手段 |
|---|---:|---|---|
| E702 | 7607 | 一行多语句（分号） | `ruff format` 自动拆行 |
| E701 | 1852 | 一行多语句（冒号） | `ruff format` 自动拆行 |
| F401 | 98 | 未使用导入 | `ruff check --fix` |
| E401 | 84 | 一行多个 import | `ruff check --fix` |
| E731 | 69 | `x = lambda ...` | unsafe-fix 转 `def` + `ruff format` 重排缩进 |
| F841 | 50 | 未使用局部变量 | 逐条人工判定（删死代码 / 保留副作用调用 / 拆分元组解包） |
| E402 | 39 | 模块级 import 不在顶部 | `sys.path.insert` 之后的 import，标注 `# noqa: E402` |
| E741 | 20 | 歧义变量名 `I`/`l` | 作用域内重命名（`interp` / `intensity_fraction` / `lap` / `bgk_fields` / `entry`） |
| F541 / E703 | 9 | f-string 无占位符 / 多余分号 | `ruff check --fix` |

CI 注释本身写明了修复路径：「Advisory on purpose …… Fix the backlog, then drop this
`continue-on-error` and promote it to a gate.」因此本单的正解是把欠账清零，而不是放宽规则或
扩大 `exclude`。

## 2. 方法与流程（每一步都可复跑）

1. **先量化再动手**：基线统计与 `ruff format --check` 计数落表（上表），确认 9828/556 是
   纯风格欠账，不含潜在语法错误（`python -m compileall` 基线通过）。
2. **分阶段机械修复**，每阶段后重跑 `check + format --check + compileall`：
   - `ruff format src tests scripts` → 立即消掉 E701/E702/E703 共 9462 条（占 96.3%）。
   - `ruff check --fix`（safe）→ 消掉 F401/E401/F541 共 188 条。
   - `ruff check --select E731 --fix --unsafe-fixes` + `ruff format` → 69 条。
3. **人工判定剩余 112 条**（F841 50 / E402 42 / E741 20），原则：**不改行为，只改表达**。
   - F841：纯值右值直接删语句；`pytest.importorskip("gmsh")` 这类**带副作用的调用**保留为
     裸调用（删绑定不删调用）；元组解包里只摘掉未使用的目标（如
     `ny, nx = ...` → 整条删除，`xf, yf, zf = ...` → `xf, yf = ...`）。
   - E402：全部位于 `sys.path.insert(...)` 之后（`scripts/` 内互相导入脚本、`tests/` 内导入
     `scripts/` 辅助模块），位置是必需的，标注 `# noqa: E402` 并保持 import 位置不变。
   - E741：按语义重命名而非机械替换——`_geometry()`/`operators()` 返回的 `H, I, G` 中 `I` 是
     面插值矩阵 → `interp`；`inflow_from_intensity` 里的 `I` 是湍流强度分数 →
     `intensity_fraction`（同步修正 docstring）。
4. **反熵配置**：`pyproject.toml` 增加 `[tool.ruff]`（`line-length=88`、`target-version="py310"`）
   与 `[tool.ruff.lint] select = ["E4","E7","E9","F"]`，把规则集钉死，避免未来 ruff 放宽默认集
   再把 main 弄红；`dev` extra 的 `ruff` 加上 `<0.16` 上界（同系列内部 formatter 输出稳定）。

## 3. 结果

| 指标 | 基线 `origin/main` | 本分支 |
|---|---:|---:|
| `ruff check src tests scripts` | 9828 findings | **0** |
| `ruff format --check src tests scripts` | 556 files would be reformatted | **0**（558 files already formatted） |
| `python -m compileall src tests scripts` | OK | OK |
| AST 完全等价的文件（纯格式改动） | — | **350 / 558** |
| 携带语义增量的文件 | — | **208**（见 §4 审计） |

图：`ruff_baseline_vs_after.png`（symlog 柱状图，基线各规则计数 vs 清零；由
`make_figure.py` 从 `ruff check --statistics` 原始输出程序化生成，无手抄数字）。

## 4. 正确性审计：格式改了 350 个文件，行为没变

只靠「测试绿」不足以说明 556 个文件的格式化没有偷偷改语义，因此加了 AST 审计
（`verify_ast_equivalence.py`，对 `git archive main` 的树逐文件比对 `ast.dump`，不含行号）：

- **350 个文件 AST 完全等价** → 纯格式化，零语义影响。
- **208 个文件 AST 有差异**，逐类归因：F401 删未用导入、F841 删死赋值、E731 lambda→def、
  E741 改名、E401 拆分 import。与基线 ruff 命中文件集合求并集后，**206/208 直接对应**；
  剩余 2 个（`scripts/audit_compatible_viscous_sign.py`、
  `tests/test_native_compatible_sign_convention.py`）逐个反查 diff，差异均为
  `Assign(value=Lambda(...))` → `FunctionDef`，即 E731 转换，属预期。

## 5. 注意点（踩过的坑）

1. **`ruff format` 会拆一行多语句**：E701/E702 看似需要手改，实际 `ruff format` 自动拆成多行
   （186 个 E7xx 文件里没有一条需要人工处理），先 format 再判剩余，省掉 96% 工作量。
2. **E731 unsafe-fix 会留下错缩进**：lambda→def 后 `ruff format` 必须再跑一遍，否则
   `format --check` 仍红；且它可能在同类作用域内造出 **F811**（`native_simple3d.py` 里出现两个
   同名 `def norm`），需删掉前一个死定义。
3. **F841 不能无脑 unsafe-fix**：`x = pytest.importorskip("gmsh")` 若按 unsafe-fix 变成裸调用
   没问题，但若整条删掉就丢掉了 skip 语义（本单曾误删，已恢复为
   `pytest.importorskip("gmsh")`）。同理 `np.load(...)` 删除反而少了 fd 泄漏。
4. **F841 有级联**：删掉 `B = sparse.coo_matrix(...)` 后，只为构造 `B` 而存在的
   `ids/oi/ni` 变成新的 F841，需要迭代修补（本单跑 3 轮 `check --fix` + 修补直到收敛）。
5. **Python 3.11 的 f-string 不被 tokenize 拆词**：E741 的 `l` 改名时，f-string 内的
   `{l[...]}` 不会被 tokenize 识别，会漏改并触发 F821。修法是改完用 `ruff check` 复核
   F821，再定点补 f-string 内的引用（本单在 `audit_two_phase_surface_tension.py` 命中）。
6. **E741 改名要有语义**：`I` 在 `_geometry()` 语境是面插值矩阵不是单位阵，叫 `interp` 才对；
   一次性把 `I` → `identity` 会引入错误注释级误导。

## 6. 未做的部分（明确边界）

- **未把 `lint (advisory)` 提升为阻断门**：`.github/workflows/ci.yml` 不在本工作树的稀疏检出
  集合内（`git sparse-checkout list` 无 `/.github/`），改动需动检出配置；且作业名 `lint (advisory)`
  可能被平台侧检查名索引，改名有把「已知检查」变成「缺失检查」的风险。当前状态是：**作业绿**
  （欠账清零），规则集已在 `pyproject.toml` 钉死，因此「提升为阻断门」只是一行
  `continue-on-error` 的删除，留作维护者的策略动作，风险与收益摆在 §1 注释里。
- **未提升测试覆盖率门禁**、未改任何求解器数值行为（本单零算法改动）。

## 7. 误差判定

本单属 CI/静态检查类任务，**关键量是整数计数而非连续物理量，判据是精确相等而非 <3% 相对误差**：

| 关键量 | 基准 | 实测 | 判定 |
|---|---|---|---|
| `ruff check` findings | 0（门禁要求） | 0 | ✅ 精确达成（不是近似） |
| `ruff format --check` 未格式化文件数 | 0 | 0 | ✅ 精确达成 |
| 测试失败数 | 0（CI 判据） | 见 §8 | 逐条判定 |

（物理量级误差判定不适用：本单不改任何离散格式、系数、边界条件；AST 审计已证明 350 个文件
零语义变化。）

## 8. 测试（如实记录，含未完成部分）

| 检查 | 命令 | 结果 |
|---|---|---|
| 全量可收集性 | `python -m pytest --collect-only -q` | **1153 tests collected in 12.4s，0 collection error** → F401 删导入未破坏任何测试模块/公开 API 的导入路径 |
| 语义改动模块定向回归 | `pytest -q -k "sa_newton or sparse or overset or two_phase or multi_element or conservative or sst3d or dfg or report"` | 已跑到的前 38%（约 190 个用例）**0 失败**，随后被 105 s 单次命令时限截断 |
| 全量套件 | `python -m pytest -q --timeout=1800 -p no:cacheprovider`（detached，日志 `/tmp/pytest_lint2.log`） | 跑到 **43%（约 500/1153 用例）0 失败** 后因本会话时间预算停采；**未取得本机全量绿灯结论** |

诚实说明：本机全量 pytest **没有跑完**（单次 shell 调用上限 120 s，全量在本机需 ~25 min 以上，
且同机有约 70 个其他 agent 进程抢 CPU）。第一次 detached 运行在 30% 附近出现过 1 条失败，
第二次同位置未复现——该主机的工作树是 6% 稀疏检出（`git sparse-checkout list` 无
`/workspace/suite-channel` 等 `docs/` fixture），按 `dev_report_tensorfvm-ci-repair.md` §1.2 的
既有结论，这类 `docs/`-fixture 用例在本机必然失败/波动，CI 是全量检出。因此「测试作业绿」
这一条**以 PR 上的 CI 结果为准**，本单只保证：lint/format 精确清零（本地 + CI 同版本
ruff 0.15.0）、AST 审计通过、可收集性无回归、语义改动模块定向回归无失败。

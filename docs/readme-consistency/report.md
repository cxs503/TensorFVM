# README / 示例 与代码 API 一致性巡检与修复（TensorFVM，m8859）

日期：2026-11-10 · 分支：`agent/m8859` · 巡检对象：`README.rst`（539 行）及其 15 条示例命令
产物：`scripts/audit_readme_consistency.py`（可复跑巡检器）、`docs/readme-consistency/audit.json`（机器可读结果）、
`docs/readme-consistency/audit-summary.md`（自动生成表）、`docs/readme-consistency/audit-figure.png`、
`tests/test_readme_consistency.py`（回归门）

## 1. 案例介绍（背景与对象）

`README.rst` 是 TensorFVM 唯一的对外说明面。它同时承担三种可被证伪的断言：

1. **可执行断言**：``python -m tensorfvm --mesh-type c-grid …`` 里的每个选项都必须被目标程序接受；
2. **API 断言**：导出的符号、`SolverConfig` 默认值、网格合法性规则、输出文件清单；
3. **证据断言**：benchmark 参考系数（`Cd=5.579535`）、误差门（3%）、文档指针（`docs/…`）、
   测试运行方式、可选依赖安装方式。

三类断言都会在代码演进中**静默漂移**：读者照着 README 执行，得到的是"命令跑不通"或更糟的
"看起来跑通了但只跑了一小部分回归"。本单把 20 条 README 声明做成可执行检查，逐条对照代码，
修掉最关键的 3 处不一致，并留下回归门防止再次漂移。

仓库规模参考：`tests/` 233 个测试文件（176 个 `import pytest`、26 个 `unittest.TestCase`、1,399 个模块级 `test_*` 函数），
README 引用 37 个 `docs/` 指针与 6 个 `scripts/` 脚本。

## 2. 怎么做的（方法与工具链）

### 2.1 巡检器设计

`scripts/audit_readme_consistency.py` 把 README 解析成结构（RST 缩进代码块 → 拼接 `\` 续行 → 15 条完整命令），
对每条声明做一次机器判定：

- **A 类（示例命令/选项）**：把命令解析成目标（`python -m` 模块 / 控制台脚本 / `scripts/*.py`），
  取其真实选项集（`--static` 用源码 `add_argument(...)` 字面量；默认模式实跑 `--help` 解析），
  逐 token 校验 36 个选项；同时校验 8 个文档化模块的 `__main__` 入口与 `pyproject [project.scripts]`
  9 个入口是否指向真实函数（含 `module:func` 解析）。
- **B 类（API/配置/输出）**：`import tensorfvm` 校验 6 个文档化导出名；比对 `SolverConfig` 8 个默认字段
  （4×2 域、中心 (1,1)、R=0.2、U=1、ρ=1、Re=20）与派生黏度 `mu=rho·U·2R/Re=0.02`；
  用 6 个边界用例实测网格合法性规则；核对 10 个运行输出文件名是否真被导出代码写出、`fields.csv` 表头。
- **C 类（基准常数）**：DFG `Cd=5.579535` + 最细 256×96 + 3% 门；NACA0012 `Cl=0.2009636 / Cd=0.1250065`
  + 读图 ±0.0001；平板 `Cf=0.074Re^-1/5`、Re_L=1e5、20% 门、首层 `y+<1`；URANS `Cd=1.12/St=0.20`；
  30P30N `Cl=2.167089/Cd=0.033243`。
- **D 类（文档与前置）**：37 个 `docs/` 指针存在性；README 41 个三位以上有效数字能否在 `docs/` 证据中
  找到同值（数值等价，容忍 2e-3 相对差，避免 `6.11e-9` vs `6.110e-09` 误判）；测试运行方式与语料一致性；
  文档化命令的可选依赖是否有安装说明。
- **E 类（行为/证据安全）**：文档化命令默认输出路径是否落在只读证据树 `docs/`；
  并按 README 参数**实跑**一次贴体圆柱示例，核对退出码与产出文件、CSV 表头。

### 2.2 判定与运行

```bash
python scripts/audit_readme_consistency.py                 # 全量（含 --help 探测 + 端到端实跑）
python scripts/audit_readme_consistency.py --static        # 静态（测试门用）
python -m pytest -q tests/test_readme_consistency.py       # 回归门（4 项）
python -m ruff check scripts/audit_readme_consistency.py tests/test_readme_consistency.py
```

环境：本机 loong64 / Python 3.11 / torch 2.7.1 / CPU；全量巡检 1 min 43 s，静态巡检 7 s。

## 3. 计算结果

### 3.1 巡检总账

| 阶段 | 抽查项 | PASS | FAIL |
|---|---|---|---|
| 修复前（静态+全量均确认） | 20 | 17 | **3** |
| 修复后 | 20 | **20** | 0 |

分面计分见 `audit-figure.png` 左图；逐项证据（含实测数字）见 `audit-summary.md` 与 `audit.json`。

### 3.2 发现的 3 处关键不一致与修复

| # | 面 | README 声明 | 代码实测 | 危害 | 修复 |
|---|---|---|---|---|---|
| 1 | 测试入口 | "测试使用标准库 unittest"；``python -m unittest discover -s tests -v`` | `tests/` 233 个文件中 **176 个 `import pytest`**、26 个定义 `unittest.TestCase`、1,399 个模块级 `test_*` 函数；`unittest discover` 只能收集橙色那 26 个文件（11.2%），且本例实测 >120 s 超时 | 读者据文档得到的"通过"只覆盖约 1/9 回归面，**假绿**风险最高 | README 改为 `pip install -e ".[dev]"` + `python -m pytest -q`，并显式说明 `unittest discover` 只覆盖少数 `TestCase` 文件 |
| 2 | 证据安全 | "新增 ``python -m tensorfvm.benchmark_suite_channel``"，未给 `--output` | 该模块默认 `--output=docs/suite-channel`，**唯一**把默认输出指进已发布证据树 `docs/` 的 benchmark（其余 7 个默认 `results/…`）；实测照 README 运行会改写 `docs/suite-channel/{36x12,72x24}/si-fields.json` 并新增 6 个文件，违反 `docs/` 证据只读策略（本单已用 `git checkout` 完整回滚） | 一次普通运行即污染已提交证据，破坏可追溯性 | 默认改 `results/suite-channel`；README 说明"重建证据须显式 `--output docs/suite-channel`" |
| 3 | 前置依赖 | 安装节"仅依赖 PyTorch"，其余处直接给出 `tensorfvm-benchmark` / `python -m tensorfvm.verification` 命令 | `verification/report.py` 模块级 `import numpy / matplotlib`；实测在屏蔽 numpy 的进程里 `import tensorfvm.verification.__main__` 抛 `ModuleNotFoundError`，且错误信息不指向 extras | 照文档安装后跑文档化命令直接崩，且看不出缺什么 | 安装节补 `.[dev]`/`.[benchmark]`/`.[mesh]` 三个 extras 的用途与命令，verification 段落标注需 `.[benchmark]` |

修复均为最小增量：`git diff --stat` = README 3 段文字 +12/−5 行、`benchmark_suite_channel.py` +3/−1 行（默认值 1 行 + 2 行注释）。

### 3.3 示例可运行性验证（实跑，非纸面）

| 示例（README 原文命令，按成本缩小网格/迭代数） | 退出码 | 产出 | 与 README 声明一致 |
|---|---|---|---|
| `python -m tensorfvm --mesh-type body-fitted --nx 16 --ny 8 --reynolds 20 --max-iterations 20 --output …` | 2（达迭代上限，符合"2=未收敛但已导出"） | `fields.csv, history.json, summary.json, nodes.csv, cells.csv, velocity.svg, pressure.svg`；表头 `x,y,u,v,p,fluid` | ✅ 7/7 声明文件 |
| `python -m tensorfvm --mesh-type c-grid --airfoil-code 0012 --nx 16 --ny 8 --domain-length 10 --domain-height 8 --airfoil-x 2 --airfoil-y 4 --max-iterations 10` | 2 | 上述 7 个 + `airfoil.csv` | ✅ C 型网格额外产出与 README 一致 |
| `tensorfvm-mesh-cylinder-3d --output … --nx 24 --ny 12 --nz 4 --stretching 2.5` | 0 | `nodes.csv, cells.csv, cylinder-faces.csv, summary.json` | ✅ 三维 O-grid 网格导出可执行 |
| `python -m tensorfvm.benchmark_suite_channel --max-iterations 40`（三网格 36×12/72×24/144×48） | 2（40 步未达 3% 门，如实返回 2） | 三档 `si-fields.json` + `benchmark.json` | ✅ 命令可跑；修复后不再写入 `docs/` |
| `python -m tensorfvm.verification`（屏蔽 numpy 环境） | ImportError | — | ❌→ 已在 README 补 extras 说明（修复 3） |

### 3.4 结果图

`docs/readme-consistency/audit-figure.png`（matplotlib，200 dpi）：

- 左：13 个抽查面按 PASS/FAIL 堆叠，修复前 3 个红色段（D 测试入口、D 前置依赖、E 证据安全）修复后清零；
- 右：测试语料构成（176 pytest / 26 `unittest.TestCase` / 31 两者皆无，共 233），
  直观解释"旧命令只覆盖 11% 回归面"。

> 说明：渲染主机无 CJK 字体（`fc-list` 仅有 Noto Sans 拉丁族），图内标签用英文以免豆腐块；正文与表格中文不受影响。

## 4. 误差判定

本单是文档/后端一致性任务，关键量不是物理场误差，而是**声明复现率**与**常数偏差**，逐项判定：

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| README 声明复现率 | 20/20 条声明可复现 | 修复后 20/20 | 0% | ✅ 是 |
| 5 组 benchmark 参考常数（DFG Cd、NACA Cl/Cd、平板 Cf 系数、URANS Cd/St、30P30N Cl/Cd） | README 印刷值 | 代码常量逐位一致（如 `5.579535`、`0.2009636`、`0.1250065`、`2.167089`、`0.033243`） | 0.000% | ✅ 是 |
| 37 个 `docs/` 指针 | 检出中可打开 | 37/37 | 0% | ✅ 是 |
| 41 个 README 有效数字 | `docs/` 证据同值 | 41/41（数值等价） | 0% | ✅ 是 |
| 36 个示例命令行选项 | 目标程序接受 | 36/36 | 0% | ✅ 是 |
| 修复前 3 处不一致 | 一致的声明 | 3 条不可复现 | 类别性失配（非数值） | ❌ 修复前不满足；修复后为 0 |

**未做的部分**：本单不重跑任何 3% 物理验收 benchmark（DFG 256×96、NACA C 网格、URANS 300 步、30P30N Re=9e6），
因此不对任何流动物理精度给出新结论；`Cd/Cl/St` 等参考常数只做"README 数值 ↔ 代码常量"的**一致性**核对，
不代表这些案例重新通过验收。

## 5. 注意点（踩坑与局限）

1. **证据树默认输出陷阱（真实踩到）**：直接执行文档化的 `benchmark_suite_channel` 命令改写了 `docs/suite-channel/`
   两个 `si-fields.json`，用 `git checkout -- docs/suite-channel && git clean -fdq` 完整回滚后 git 状态归零；
   这正是修复 2 的来源，也是本单唯一的"污染-回滚"事件。
2. **`--help` 探测成本**：9 个模块冷启动 `--help` 各约 8 s（torch 导入），全量巡检 ≈1 min 43 s；
   故巡检器分 `--static`（源码 `add_argument` 字面量，7 s，供测试门）与全量两种模式。
3. **包 `__main__` 不需要 `if __name__ == "__main__"` 守卫**：`src/tensorfvm/__main__.py` 是裸执行模块，
   首版检查误报 FAIL，已改为"包看 `__main__.py` 是否导出 main、单文件才要求守卫"。
4. **数字溯源要按数值而非字面**：README 的 `6.11e-9` 与证据 `6.110e-09` 不同字面，需数值等价比较，否则误报。
5. **README 的"数字"只到三位有效数字**：三位以下（如 `3%`、`20%` 门限）不参与溯源统计，避免把门限值当实测值。
6. **局限**：巡检器只覆盖 README 显式声明；叙述性更强的段落（"有限体积面通量守恒""Rhie–Chow 耦合"）
   只做了间接核对（代码路径存在），未做数值级验证——那属于各 benchmark 报告的责任面。
7. **未修的小瑕疵（如实记录，避免越界改动）**：README 仍写 "返回码 … 1 表示计算或输出失败"，而 CLI `main()`
   只显式返回 0/2，1 由未捕获异常给出；属于表述宽松而非缺陷，未动。README 亦未逐一列出
   `tensorfvm-benchmark-dfg/airfoil/flat-plate/urans/30p30n/cylinder-3d` 六个控制台入口（均可用 `python -m` 等价调用）。

## 6. 复现

```bash
git checkout agent/m8859
python scripts/audit_readme_consistency.py --json docs/readme-consistency/audit.json \
    --report docs/readme-consistency/audit-summary.md     # 期望 exit 0，20/20 PASS
python -m pytest -q tests/test_readme_consistency.py      # 期望 4 passed
```

回滚点：本单改动仅 `README.rst`、`src/tensorfvm/benchmark_suite_channel.py` 一段默认值，
新增 `scripts/audit_readme_consistency.py`、`tests/test_readme_consistency.py`、`docs/readme-consistency/`。
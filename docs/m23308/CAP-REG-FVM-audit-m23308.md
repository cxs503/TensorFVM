# 交差核验：m23215 · CAP-REG-FVM（TensorFVM 能力注册 + 内核架构审计）

审计单：m23308（board-audit，非重做）。被审对象：mission 23215，报告 `tensor_reports.id=20873`，
交付提交 `agent/m23215` = `300daa9bdd7699989e2493b2e8ec9e438bd226ab`（工作树 `/root/tp-agent/work/m23215`）。
本单**不改被审代码**：全部复核都在 m23215 工作树内只读执行，复核前后 `git status --short` 均为 0 行
（工件可重新生成且逐字节一致）。复核日志：`artifacts/m23308/repro_log.txt`。

## 0. 基准裁判（唯一裁判）

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CAP-REG-FVM';
-- (0 rows)
```

**本锚点没有登记任何基准条目** → 不存在可机核的 ref/got/tolerance 三元组，L4 在本锚点上**按定义不可达**。
TensorFVM 仓内共 13 条基准（CASE-FVM-CAVITY-RE1000 / CHANNEL-KMM180 / KOVASZNAY-ORDER / HEATED-CAVITY /
WOODWARD-COLELLA / SOD1978 等），它们判的是**求解器物理算例**，不是「能力注册表」这个交付物本身，
不能借来给 CAP-REG-FVM 加分。以下判级只依据功能实质。

## 1. 三件套取证（逐项实测）

| 项 | 声明 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts 文件真存在 | 5 件 | `docs/m23215/registry_dump.json`(902 L)、`fvm_audit.json`(761 KB)、`error_bands.json`、`SOLVER_INVENTORY.md`(140 L)、`KERNEL_ARCHITECTURE.md`(208 L)、`README.md`(43 L) 全在 | ✅ 存在 |
| benchmark ref vs got | 误差带 4 条 | 无登记基准；4 条均为**自算对解析/守恒量**（见 §3） | ⚠️ 无裁判，只有 A 级自证 |
| repro_cmd 可执行 | 5 条 | 全部实跑，见下表 | ✅ 4/5 绿，1 条红（本单核心发现） |

| # | repro_cmd | exit | 关键输出 |
|---|---|---|---|
| 1 | `python src/tensorfvm/registry.py --static --strict-exit` | 0 | `count=47`，`problems=3`（全 warning；error 级 0） |
| 2 | `python src/tensorfvm/registry.py --static --parity` | 0 | `ok=true, static_count=47, import_count=47, mismatched=[], import_failures=[]` |
| 3 | `PYTHONPATH=src python scripts/m23215_error_bands.py` | 0 | annular `2.813e-01 %`（与提交件一致，重跑后 git 干净） |
| 4 | `python scripts/m23215_fvm_audit.py --check` | 0 | 装饰器↔注册表对账通过，全仓审计输出 |
| 5 | `python -m pytest tests/test_capability_registry_fvm.py -q` | **1** | **1 failed, 16 passed, 71.78 s** |

五条 repro 命令本身可执行、无环境依赖陷阱（`--static` 模式还带 torch 投毒测试），这一点是真实成立的。

## 2. 严重度排序的发现

### F1（阻断“problems=0/error 0”这条自证链）被审件自带契约测试为红
`tests/test_capability_registry_fvm.py::test_registry_error_bands_equal_the_measured_bands`（`tests/test_capability_registry_fvm.py:206-215`）失败：

```
E  AssertionError: fvm.verification.annular_band claims verified without a measured band
```

原因：`src/tensorfvm/verification/annular.py:101-111` 新增 `@capability('fvm.verification.annular_band', maturity='mature', verified=True, evidence={'max_error_pct': 0.2813, 'coarse_max_error_pct': 3.0336, ...})`，
而 `docs/m23215/error_bands.json` 的键只有 `fvm.core.annular_axial` 等 4 个，没有 `fvm.verification.annular_band`。
即：**数值不是编的**（0.2813 % 与 `fvm.core.annular_axial` 的实测 Linf 完全一致，`error_bands.json` 可复算），
但「已量测误差带」与「verified 声明」的机器绑定关系断了 → 该 `mature` 在机器口径下是**未被支撑的成熟声明**。
这正是注册表反虚标设计要挡的东西，被它自己的测试挡住了。复现：
`cd /root/tp-agent/work/m23215 && python -m pytest tests/test_capability_registry_fvm.py -q -k error_bands`。

### F2（中）仓内 README 数字与代码/自报报告不一致（工件陈旧）
| 口径 | count | mature | tested | verified | wired |
|---|---|---|---|---|---|
| `docs/m23215/README.md` 正文 | 46 | 3 | 41 | 4 | 35 |
| `docs/m23215/registry_dump.json`（已提交） | 46 | 3 | 41 | 4 | 35 |
| **实测 `--static`** | **47** | **4** | **42** | **5** | **36** |
| 已发布报告 `tensor_reports.id=20873` | 47 | 4 | — | — | — |

差值恰为新增的 `fvm.verification.annular_band`：`registry_dump.json` 与 README 是在最后一次装饰器落地**之前**生成的，
发布报告用的是**之后**的口径。结论：报告比仓内 README/工件**更新**，仓内 README 是陈旧件。
方向上是「自报偏低」而非夸大，但它破坏了「文档=代码」这一本单的全部价值前提。

### F3（低）README 里一条误差数字陈旧 2 个量级
README 写 `fvm.linear.diagnostics` 后向误差 `1.5e-14`，实测 `error_bands.json` 为
`componentwise_backward_error=1.54e-16`、`normwise=5.4e-17`。两者都 ≪3%，不影响结论，但说明表格非程序化注入。

### F4（中，范围性）验证覆盖 = 4/47，且无锚点基准
`verified=True` 仅 5 项、`mature` 4 项；其余 42 项 `verified=False`，本单如实标注（honest-FAIL 边界，值得肯定），
但这意味着注册表的**成熟度结论覆盖面约 9 %**，而 `problems=3`、`with_endpoint=3` 这些数字都不构成外部精度证据。
`CAP-REG-FVM` 无登记基准（§0），故「基准验证」维度为空。

### F5（信息）集成为「被动元数据」，无生产消费者
46 个 `src/` 模块 `from .registry import capability`（如 `src/tensorfvm/solver.py:14`、`solver3d.py:15`）——
装饰器随主求解器 import 生效；但 `registry.dump()` 的调用方只有 registry 自身 CLI、
`scripts/m23215_fvm_audit.py` 与测试，**平台/后端无任何端点或运行期逻辑消费该注册表**。
同时 `src/tensorfvm/registry.py` 在审计时只存在于 `agent/m23215`（`300daa9b`），**尚未并入 main**（m23308 工作树中不存在该文件）。
故「接入主求解器」只到 import 边，未到数据流。

### F6（低）`problems=3` 与 README「error 级 0」自洽
`--strict-exit` 返回 0 且 `problems` 3 条均为 warning 级，README 的措辞成立。

## 3. 误差异议逐项（唯一可核部分）

| 能力 | 参考 | got | 容差 | 是否 <3% |
|---|---|---|---|---|
| `fvm.core.annular_axial`（nt128×nr32） | 环形轴向 Poiseuille 闭式解 | 相对 Linf 0.2813 %，流量 0.1056 % | <3 %（自定） | ✅ |
| 同上（nt32×nr16 粗网格） | 同上 | Linf 3.0336 % | <3 % | ❌（3.03 %；且径向加密无效，阶 −0.083；角向阶 1.985/1.929 → 误差由分面墙主导）—— 如实 FAIL |
| `fvm.freesurface.plic_geometry` | `A(n,c(α))=α` 代数恒等式 | 1.36e-11 % | — | ✅ |
| `fvm.flux.overset_transfer` | 非匹配界面通量守恒 | 1.17e-13 %（20 trials） | — | ✅ |
| `fvm.linear.diagnostics` | 直接解分量型后向误差 | 1.54e-16（相对量） | — | ✅ |
| 其余 43 项 | 无可机核参考 | 未测 | — | 不判（`verified=False`，已如实标注） |

`tensor_benchmark_cases` 无 CAP-REG-FVM 行，故上表容差是**被审单自定**、非平台登记裁判；判级不因此给 L4。

## 4. 判级：L2

- **L0** 文档：成立（两份主文档 + 140 行能力清单 + 761 KB 审计 JSON），但文档数字与代码有漂移（F2/F3），L0 单独不足以定级。
- **L1** 数学/单元验证：成立（4 条解析/守恒误差带 + 网格收敛阶实测；17 项契约测试中 16 绿）。
- **L2 独立功能实现：成立** —— `src/tensorfvm/registry.py`（805 行，真代码：`Capability@170` / `validate_capability@203`
  反虚标纪律 / `Registry@292` / `capability()@377` / `scan_static@548` AST 五字段解析 / `import_registry@647` /
  `dump@706` / `_parity@757` 双模式逐字段比对 / `main@776` JSON stdout 契约）。输入=源码树，计算=AST+import 注册与校验，
  输出=可机读 JSON（`--parity` 实测 47≡47、零失配、零 import 失败）。这是「真实现」，不是脚本堆壳。
- **L3 不成立**：无生产消费者（F5），registry 未并入 main；装饰器只是 import 期元数据，无数据流进求解器或平台。
- **L4 不成立**：锚点基准 0 条（§0），验证只覆盖 4/47 能力，且其中 1 条成熟声明未被误差带绑定（F1）。

## 5. 处置

**approve with notes → level 2（confirmed 预期）**。返修项（不影响 L2，影响升 L3/L4）：
1. F1：把 `fvm.verification.annular_band` 绑到实测带（`error_bands.json` 增键或复用 `fvm.core.annular_axial` 测量），使 `pytest -k error_bands` 转绿——一行改动，但它决定「mature 可机核」这条底线。
2. F2/F3：在最终代码提交上**重新生成** `registry_dump.json` 与 README 表格（程序化注入，禁止手抄）。
3. 欲升 L3：给 `registry.dump()` 加一个真实消费者（平台能力端点 / CI 闸门），并合入 main。
4. 欲升 L4：在 `tensor_benchmark_cases` 为 CAP-REG-FVM 登记可机核判据（如「mature 集合的误差带必须逐条落表且 <3%」这类可判命题），而不是继续靠自算带。

## 6. 独立性声明与局限

- 本席非 m23215 作者；未修改被审工作树任何文件（复核前后 `git status` 均干净），未 push。
- 未运行：torch 相关 import 模式的**数值**回归（只跑了 parity 的一致性比对，未跑求解器算例）、全仓 587 模块测试矩阵、
  `fvm_audit.json` 全字段源码级抽检（只跑了 `--check` 对账与其输出）。这些标 "not_run"，不计入通过。
- 判级只依赖可复算数字：本单全部 got 值由 `artifacts/m23308/repro_log.txt` 记录的命令现场产出。
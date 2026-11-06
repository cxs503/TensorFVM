# 归档证据的源码陈旧性审计（verification archival staleness）

## 1. 为什么做这件事

仓库的 179 份 `docs/**/manifest.json` 用 SHA-256 把每份验证证据绑定到“产出它的源码”。
既有的 `scripts/audit_external_evidence.py` 只检查一个方向：**归档的那份拷贝是否与记录的摘要一致**。
反方向从未被检查：**当前工作树是否还是那个源码**。

于是出现一种静默失效：某模块在验证归档之后被修改，`manifest.json` 里的摘要仍然与冻结快照一致，
审计通过、报告照旧发布，但报告里的数字已经不是当前代码算出来的了。这类“证据绑定失效”
既不是数值错误，也不是测试失败——除非有人逐条手查，否则不可见。

本审计把它变成可执行、可复算、可棘轮化的机器判据。

## 2. 方法

对每份 `docs/<case>/manifest.json` 中所有 `{路径: sha256}` 型生产者绑定
（键名：`source_sha256` / `acceptance_source_sha256` / `report_source_sha256` /
`integration_source_sha256` / `plot_sources`），按如下顺序解析文件：

1. `docs/<case>/<source_snapshot_root>/<path>`（冻结快照，优先——冻结源码的案例按设计就是要与快照比）；
2. `<repo>/<path>`（仓库根，用于归档任务写下的非源码输入：`docs/*.json`、网格、数据 README）；
3. `docs/<case>/<path>`（案例目录）。

解析不到任何文件 → `missing`（比陈旧更严重：绑定既不在快照也不在树里，无法核对）。
解析到且摘要一致 → `fresh`；摘要不一致 → `changed`。

每个入口都记录它实际解析到哪个来源（`provenance`）与解析结果路径，读者可以逐条手工复核。
另外用 `docs/benchmarks-index.md`（仓库对外主张总账）判定哪些陈旧案例是**对外主张**，
哪些只是内部记录。

## 3. 结果（当前工作树，53 个含源码绑定的案例）

| 指标 | 数值 |
|---|---:|
| 案例数（含源码绑定） | 53 |
| 记录的源码绑定 | 1378 |
| `fresh` | 1275 |
| `changed` | 102 |
| `missing` | 1 |
| 陈旧案例（有 changed 或 missing） | 19 |
| 陈旧且被 `benchmarks-index.md` 引用 | 16 |

逐条明细见本目录 [report.md](report.md)（由脚本从 `staleness.json` 程序化生成，零手抄）与
[staleness.json](staleness.json)。图：[staleness-bars.png](staleness-bars.png)。

**集中规律**（这条比总数更有信息量）：102 条陈旧绑定几乎全部落在 2026-10/11 之前的
**遗留验证套件**（`verification-benchmarks` / `-dfg` / `-engineering` / `-gpu` / `-simple-gpu` /
`-extended` / `-sa-*` / `-cylinder3900-*`）以及少数共享模块
（`src/tensorfvm/solver.py`、`src/tensorfvm/verification/*.py`、`pyproject.toml`）。
这些案例当初归档时**没有冻结源码快照**，所以后续任何一次对 `solver.py` 的修改都会让它们整体陈旧。
相反，近期采用 `source_snapshot_root` 的案例（`verification-gmsh-*`、`verification-overset*`、
`verification-30p30n-*`、`native-*` 等）全部 `fresh`：冻结快照这条纪律确实把问题挡住了。

**唯一的 `missing`**：`verification-gmsh-controls` 把
`docs/verification-gmsh-cylinder/meshes/scale-1.msh` 记为输入绑定，但该网格并未入库
（也不在快照里）。绑定无法核对，属于“输入不可追溯”的欠账，与数值精度无关。

## 4. 这意味着什么、不意味着什么

- **意味着**：上表 16 个被总账引用的案例，其报告数字不能声称已被当前 HEAD 源码复现；
  它们只能声称“在归档时刻的源码上成立”。这是证据新鲜度问题，不是数值错误。
- **不意味着**：这些结果错了，也不意味着精度退化。本审计不重算任何流场，
  因此对“当前 HEAD 上这些数字是否仍成立”**没有**结论（见第 6 节）。
- 需要强调的是：这些遗留套件里的**结论本身大多是失败/诊断记录**（例如 30P30N、SUBOFF 的未收敛诊断），
  陈旧化不会把“失败”变“通过”，但会把“当时的失败证据”与“今天的代码”割裂。

## 5. 棘轮与门禁

- `tests/test_source_staleness.py`（11 例）用夹具复现 `fresh/changed/missing` 三态、
  快照优先于工作树的解析顺序、总账引用判定，以及棘轮本身；
  两条负控（改写文件 / 删除文件）证明判据会红，不会空转通过。
- 活树棘轮：`staleness.json` 记录的 `known_stale_cases`（当前 19 个）是已知欠账；
  **新增**的陈旧案例（或某案例新增陈旧绑定）会直接红。接受方式唯一且显式：
  `PYTHONPATH=src python3 scripts/audit_archived_source_staleness.py --update-ledger`
  ——一次可评审的编辑，不是沉默。
- 实测：`pytest tests/test_source_staleness.py tests/test_report_traceability.py tests/test_evidence_checkout_policy.py`
  → 27 passed；`ruff check` 三文件全绿。

## 6. 还没做（诚实声明）

1. **没有重算**任何遗留案例。要回答“当前 HEAD 上这些数字是否仍成立”，需要在当前源码上
   重跑对应生产脚本（`verification-dfg` / `-engineering` / `-gpu` 等需要 GPU 或较长 CPU 时间），
   本单在 load15≈124（nproc=128）的容量纪律下只做静态审计，不启动重算。
2. 快照内的 `changed` 未做**语义**判定（改了注释与改了离散格式一样计为 `changed`）。
   保守方向正确（不会漏报），但会把少量无害改动也标成陈旧。
3. 只有 1 例 `missing` 属输入未入库；未系统扫描 `artifact_sha256` 一侧的缺失
   （那部分由既有 `audit_external_evidence.py` 覆盖）。

## 7. 建议的下一步（按性价比排序）

1. 对 16 个被总账引用的陈旧案例，逐个决定“补冻结快照”还是“在当前 HEAD 上重跑并重新归档”。
   前者是纯机械劳动，后者需要算力，可派给集群单。
2. 把“归档时必须冻结生产者源码”从惯例升级为写入 `manifest.json` 的硬约束
   （缺 `source_snapshot_root` 且记录了 `source_sha256` → 门禁红）。
3. 修 `verification-gmsh-controls` 的网格输入绑定：要么入库网格，要么把它降级为
   “生成式输入”（记录生成脚本与摘要），不要留不可核对的绑定。
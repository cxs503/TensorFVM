# 交差核验 m22209 `JOUR-ICE-STRUCT-GUARD`（冰域参数护栏，TensorFVM）

**核验对象**：mission m22209（`status=succeeded`，`task_key=jour-ice-struct-guard-r2`），
分支 `agent/m22209`，**已推送** commit `3db7b1f5`（"m22209: matured mission deliverables（11 个文件被门禁滤除）"），
本地工作树 `/srv/tensor/agent/work/m22209` 的树内 commit 为 `ff1e129a`（同内容，未推送版本）。
**核验者**：m22252（`agent/m22252`，TensorFVM），性质=**结榜交差核验，非重做**；本单不改被审交付物。
**基准判据（唯一裁判）**：`tensor_benchmark_cases WHERE anchor_slug='JOUR-ICE-STRUCT-GUARD'`。

## ① 案例介绍（物理背景/几何/参数表）

**物理背景**：碎冰–船耦合冰载荷（`JOUR-ICE-STRUCT` 主线）里，冰况参数（冰厚 h、抗压强度 σ_c、
摩擦系数 μ、航速 v、密度、弹模、泊松比、温度）一旦越出经验公式标定域，求解器会**静默**给出
看似正常但物理荒谬的冰阻力/冰载荷。本锚点要交的能力就是"**越域护栏**"：在参数进入模型之前
做三级边界判定（硬边界=物理不可能→拒绝；软边界=标定域外→警告/可显式放行；典型区→通过）。

**几何**：无（纯参数域校验，不涉及网格/场量）。**工具链**：python 3.11.6 / loongarch64，pytest 8，
matplotlib Agg。

**交付参数表**（本审计**机器解析** `docs/FVM_ICE_GUARDRAILS/REPORT.md` §2.2 表格得到，
非手抄，见 `docs/audit_m22252/artifacts/spec_vs_impl.json`）：

| 参数 | 硬边界 | 软边界（标定域） | 典型区 |
|---|---|---|---|
| ice_thickness h [m] | [0.01, 10.0] | [0.05, 5.0] | 0.3–2.5 |
| ice_strength σ_c [MPa] | [0.1, 30.0] | [0.3, 15.0] | 1.0–5.0 |
| ice_friction μ [-] | [0.0, 1.0] | [0.01, 0.4] | 0.05–0.20 |
| ice_speed v [m/s] | [0.0, 15.0] | [0.1, 8.0] | 0.5–5.0 |
| ice_density ρ [kg/m³] | [800, 920] | [850, 918] | 880–910 |
| ice_elastic_modulus E [GPa] | [0.1, 20.0] | [1.0, 12.0] | 3.0–8.0 |
| ice_poisson_ratio ν [-] | [0.0, 0.5] | [0.25, 0.40] | 0.30–0.35 |
| ice_temperature T [°C] | [−60, 0] | [−45, −1] | −30…−5 |

交付物三件套（推送 commit `3db7b1f5` 实有）：
`src/tensorfvm/ice/{__init__,parameters,guardrails}.py`（71/270/296 行）、
`tests/test_ice_guardrails.py`（405 行 / 40 个 test 函数）、
`docs/FVM_ICE_GUARDRAILS/{REPORT.md,demo_guardrails.py}`、`docs/oss_solutions/JOUR-ICE-STRUCT-GUARD.json`。

## ② 怎么做的（方法/流程/关键设置）

1. **裁判先查**：`su postgres -c psql -d tensorplatform` 现查
   `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='JOUR-ICE-STRUCT-GUARD'`
   → **0 行**（全表 144 行；`anchor_slug LIKE 'JOUR%'` 亦 **0 行**；最近的冰锚点是 id 9/11/48/56，
   全部与本锚点无关）。原始输出存 `docs/audit_m22252/artifacts/judge_query.txt`。
   → 结论：**该锚点没有注册基准判据**，因此"got 是否在容差内"这一问**无 ref 可对**（不是通过，是缺题）。
2. **交付物在不在**：`git cat-file -e 3db7b1f5:<path>` 逐文件核 18 个 commit 文件 → 全部在分支上；
   `git ls-remote origin` 确认 `refs/heads/agent/m22209 = 3db7b1f5`（**已推送**）；
   `git ls-tree origin/main -- src/tensorfvm/ice` → **空**（**未并入 main**）。
3. **单测实跑（不复用被审单结论）**：把 `3db7b1f5:src/tensorfvm/ice` + `3db7b1f5:tests/test_ice_guardrails.py`
   取到 `/tmp/audit_m22252` 独立包路径下跑 pytest → **40 passed, rc=0**（0.10–0.29 s）。
4. **run_artifacts 复算**：`git archive 3db7b1f5` 取交付树到 `/tmp/audit_delivery`，跑交付自带的
   `docs/FVM_ICE_GUARDRAILS/demo_guardrails.py` → 3 个工件与工作树里交付的那份 **md5 逐位相同**
   （json / txt / png 三者全同）。**但**：这 3 个工件从未被 commit（`git show --stat ff1e129a` 无
   `artifacts/`，推送分支 `git ls-tree 3db7b1f5 -- artifacts` 亦无）→ 工件只活在被审单工作树里。
   记录：`docs/audit_m22252/artifacts/delivery_repro.json`。
5. **独立重导判定规则（本审计的主体工作）**：不调用交付判定代码，仅从**文档 §2.2 表格**机器解析出边界，
   按文档 §2.1 语义自行实现 `hard→fail / soft→extrapolation / else→pass`，再与交付 API
   在 **8 参数 × 38 点**（硬边界、硬边界±ε、软边界±ε、典型区端点、硬域内 25 点等距扫描）上逐点比对
   → **304/304 一致（100.0%，偏差 0）**；同时核 **48 个边界字段**的 spec↔impl 漂移 → **0 处**，
   以及嵌套不变式 `typical ⊆ soft ⊆ hard` → **8/8 成立**。
   脚本 `docs/audit_m22252/audit_ice_guardrails.py`，数据 `docs/audit_m22252/artifacts/{decision_grid,spec_vs_impl,summary}.json`。
6. **非有限值/异型探针（护栏类件的真正失效面）**：`nan / inf / -inf / bool / str / None / -0.0` 七类输入
   逐类灌进 `validate_ice_parameters` 与 `clamp_ice_parameters`。
7. **集成面取证**：`git grep` 推送 commit 全树
   `tensorfvm.ice|validate_ice_parameters|clamp_ice_parameters` → 除模块自身与
   `tests/test_ice_guardrails.py` 外，唯一调用方是**文档 demo**（`docs/FVM_ICE_GUARDRAILS/demo_guardrails.py:120`）；
   `src/tensorfvm/__init__.py` **未**导出 `ice`（grep 无命中）。

## ③ 计算结果（数字表，程序化注入）

**A. 判据/一致性**（`docs/audit_m22252/artifacts/summary.json`）

| 检查项 | 规模 | 结果 |
|---|---|---|
| 注册基准行（`anchor_slug='JOUR-ICE-STRUCT-GUARD'`） | 144 行表 | **0 行**（缺题） |
| spec↔impl 边界字段漂移 | 8 参数 × 6 字段 = 48 | **0 处** |
| 嵌套不变式 typical⊆soft⊆hard | 8 | **8/8 成立** |
| 独立规则 vs 交付判定（数值网格） | **304** 例 | **304 一致 = 100.0%** |
| 交付单测 | 40 个 test | **40 passed / 0 failed**（本审计实跑） |
| 交付工件可复现 | 3 文件 | **3/3 md5 逐位相同** |
| 工件入库 | — | **0/3 在 git 内**（仅工作树） |

**B. md5 复算表**（`delivery_repro.json`）

| 工件 | 交付工作树 md5 | 本审计重跑 md5 | 同 |
|---|---|---|---|
| validation_demo.json | `ac50e191…3bf378` | `ac50e191…3bf378` | ✅ |
| failure_case_report.txt | `b39ddc27…a221393` | `b39ddc27…a221393` | ✅ |
| fig_ice_guardrail_bounds.png | `992565c7…4744a07` | `992565c7…4744a07` | ✅ |

**C. 非有限值/异型输入判据表**（`summary.json::nan_battery`，逐条实跑）

| 输入 | 期望（护栏目的） | 交付实际 `status/is_valid` | 判定 |
|---|---|---|---|
| `h = nan` | 拒绝 | **pass / True**，无 error 无 warning | **F1 漏洞** |
| `σ_c = nan` | 拒绝 | **pass / True** | **F1 漏洞** |
| 8 参数全 `nan` | 拒绝 | **pass / True**，8 个 NaN 全进 `validated_params` | **F1 漏洞** |
| `h = inf` | 拒绝 | fail / False | ✅ |
| `h = -inf` | 拒绝 | fail / False | ✅ |
| `h = True`（bool） | 拒绝或至少告警 | pass / True（`isinstance(True,int)` → 1.0 m 合法冰厚） | F5（低危） |
| `h = "1.5"` / `None` | 拒绝 | fail / False | ✅ |
| `clamp(h = nan)` | 钳到硬边界 | **返回 NaN**（`min/max` 对 NaN 不生效，无 warning） | **F1 漏洞** |
| `get_parameter_report(h = nan)` | 标记异常 | 打印 **"[WITHIN CALIBRATION]"** | **F1 漏洞**（假安全标签） |

## ④ 结果图

`docs/audit_m22252/artifacts/fig_audit_ice_guardrails.png`（matplotlib，150 dpi）：
- **A 面板**：8 个参数的三级边界带（从交付文档机器解析后归一化叠加），图上直接标注硬/软区间数值 ——
  用于肉眼核 spec↔impl 零漂移；
- **B 面板**：304 例判定网格的逐参数一致率（全部 100%）——独立规则对交付实现的判别力对照曲线；
- **C 面板**：失效模式探针条形图（typical / extrapolation / impossible / **NaN→PASS** / inf→fail），
  把本审计唯一发现的阻塞缺陷钉在同一张图上。

**无场量云图/无物理比较曲线的原因**：被审件是**参数域校验器**，不产生场解，也不存在注册基准
（判据表 0 行），因此"场量云图 + 解析解/文献曲线"这一项在本锚点上**物理不适用**；替代物是上面的
规则一致性曲线（B）与失效模式图（C），并如实声明该替代不等价于 L4 意义上的验证。

## ⑤ 注意点（坑与局限）

1. **F1（阻塞级，影响本能力的目的本身）**：`guardrails.py:110`（validate）/ `:199`（clamp）只做
   `isinstance(value,(int,float))`，**未做 `math.isfinite`**，而 `IceParameterBounds.is_hard_violation`（`parameters.py:38-40`）
   用 `<`/`>` 比较，NaN 与任何值比较恒 False → NaN 被当"典型值"放行。一个以"拦静默垃圾"为唯一
   使命的护栏，对上游最常见的静默垃圾（0/0、未初始化数组、`np.nan` 传播）**开口**。修法极小：
   两处 isinstance 后加 `if not math.isfinite(value): errors.append(...)`，并补 nan/inf 用例。
2. **F2（中危，证据不可持续）**：交付 JSON 的 `artifacts` 列表点名的 3 个工件**从未入库**，只在
   m22209 工作树里；本审计用推送 commit 重跑得到**逐位相同**结果，才把这一肢补成可用证据
   （`delivery_repro.json` + 本单 artifacts）。若工作树被回收，原证据链即断。
3. **F3（决定级别）**：`tensorfvm.ice` **零集成调用方**——求解器/API/Runner 全无引用，package 根也不导出；
   唯一调用方是文档 demo。故 L3（系统集成）**不成立**。
4. **F4（决定级别）**：`tensor_benchmark_cases` 该锚点 **0 行**、`docs/oss_solutions/…json` 的
   `tolerance: null` → 无注册判据即无 L4（"完整算例 + 误差分析 + 可复现"缺"误差分析 vs 外部 ref"）。
5. **F5（低危）**：11 个 `__pycache__/*.pyc` 进了 commit（噪声扩散到 `src/tensorfvm/`、`tests/`）。
6. **F6（账目）**：`tensor_reports` 无 `mission_id=22209` 行（`report_owed=false`、`report_id` 空）——
   六段报告只在仓内 `docs/FVM_ICE_GUARDRAILS/REPORT.md`（六段齐全，但没有发布到报告库）。
7. **F7（证据等级）**：8 个参数的边界值只在 docstring 里挂文献名（ISO 19906:2019 / Timco & Weeks 2010 /
   Riska 1997 / ITTC 2017），**没有值级溯源文件**（无逐条摘录、无页码/表号）。本审计只能核"文档↔代码一致"，
   不能核"边界值↔文献原值" → **external_anchor 记 WITHHELD**（不是 REAL）。
8. **F8（低危，表述过强）**：`parameters.py:8`（及 `:39` 的 `is_hard_violation` docstring）与 REPORT.md:141 §5.1 声称硬边界外"physically impossible"/
   "encompass all physically possible values"——对 `h>10 m` 不成立（冰脊龙骨、冰岛、冰川舌远超 10 m），
   该边界实际是"**船行平整冰模型的适用上界**"，宜改写为 apply-domain 语义。不影响本单判级。
9. **独立性限制**：单机单核、无外网，文献原值核查止步于"常识区间 + 文档自洽"；未对 8 参数做第三方
   数据源复算。**未做**的事：未跑全仓测试套件（与本件无关的回归不在本单范围）、未改被审代码
   （交差核验非重做）、未验证 m22191 前轮 FAIL 的具体失败面。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

**基准缺失声明**：`tensor_benchmark_cases` 该锚点 0 行，**没有 ref/got 可对**，故"物理量误差%"无从谈起；
下面用本审计能构造的、与被审件目的等价的量化判据给出误差，并逐项回答。

| 关键量 | 基准（ref） | got | 误差 | <3%? |
|---|---|---|---|---|
| 8 参数 × 6 边界字段 | 交付文档 §2.2 表格 | 代码常量 | **0/48 处不一致 = 0.000%** | **是** |
| 三级判定（304 例网格） | 文档规则独立重导 | 交付 `validate_ice_parameters` | **304/304 一致 = 0.000%** | **是** |
| 交付工件复现 | 交付工作树 md5 | 重跑 md5 | **3/3 逐位相同 = 0.000%** | **是** |
| 任务指定失败用例 h=50 m / σ_c=100 MPa | 必须 rejected | rejected（带原值+边界文本） | 定性一致 | **是** |
| **NaN 输入拦截率** | 目的：100% 拒绝 | **0%（pass）** | **偏差 100 个百分点** | **否（honest-FAIL 这一肢）** |
| clamp(NaN) 安全回退率 | 目的：100% 产有限值 | **0%（回吐 NaN）** | **偏差 100 个百分点** | **否（honest-FAIL 这一肢）** |

**结论**：数值/规则一致性三项均 0.000% 偏差（远严于 3%）；但**"拦静默垃圾"这一目的性判据在 NaN 上
100% 失效**，故本单不做"关键量全部 <3%"的粉饰——护栏对"物理荒谬但有限"的输入成立，对"非有限垃圾"不成立。

## 附：判级与交差结论

**判级 = L2（独立功能实现：真代码、输入→判定→输出、40 单测 + 工件可复现）**，理由逐级：
L1 不取（不止数学/单测，是带 API 的可用模块且工件可复现）；**L3 不取**（零集成调用方、根未导出、未并入 main）；
**L4 不取**（注册判据 0 行、无外部 ref 的误差分析）；L5 不取（无实验/外场锚）。

**处置：request changes（返修）**。F1 需在 `guardrails.py:110/:199` 加 `math.isfinite` 检查并补 nan/inf 用例；
F2 工件入库；F3 接入一个真实调用方（如 `ice_channel`/冰阻力路径）方可达 L3；F4 需在判据表注册基准行方可达 L4。
各级别证据指针与上述缺陷编号已一并写入 `tensor_maturity_verdicts`（`verdict_by='board-audit'`）。
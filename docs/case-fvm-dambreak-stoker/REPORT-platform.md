# wp-9632-01 交付复核：报告 9632 资产登记包（重开 PR#173 · 第 2 轮）

报告 9632（冰脊固结热力学）资产登记包重开交付：源 sha256 逐字节与契约一致（10749 chars/239 行），登记 25 条资产（1 resolved/6 metadata_only/18 unresolved），44/44 定位器回环通过，验收 6/6，repo 交付物与线上报告 17389 逐字节相同（9/9），离线测试 19/19；本轮修复「源副本未入库导致定位器测试静默 skip」并补 publish --check 事后门。

# wp-9632-01 交付复核：报告 9632 资产登记包（重开 PR#173 · 第 2 轮）

**一句话**：把报告 9632 声明的 **25** 条资产逐条钉死在源 markdown 的字符/行坐标上（44/44 定位器回环通过），源身份以**实测**而非抄写方式与契约基线逐字节比对一致；六项验收自检 6/6 通过，repo 内交付物与平台已发布报告 17389 **逐字节同一**（9/9）。

## ① 案例介绍（物理背景 / 对象 / 参数表）

**物理背景**：报告 9632《IN101-N07 冰脊固结热力学：龙骨大孔隙再冻结与固结度-阻力耦合》研究冰脊龙骨**大孔隙盐水再冻结**——龙骨内碎冰块间的孔隙水在低温下二次冻结，形成固结层 h_c(t)，其厚度按 Stefan 解随冻结指数 FDD 增长，固结层有效导热率用 Maxwell-Eucken 混合律处理（盐水隙/气泡/冰三相），再把固结度映射为龙骨强度与**冰阻力**（`ice_ridge_consolidation`）。

**本工序对象不是物理量**：任务类型 `01-asset-registration`，只做「资产登记 + 源身份钉死」，不做知识抽取、不做证据分级、不写 `verified`、不重述报告里的物理判据数值。

| 项 | 值 |
|---|---|
| 报告 ID / 状态 | `9632` / `published`（平台 mission 8348） |
| 场景键 | `ice_ridge_consolidation`（契约 `ice_ridge_consolidation`，一致=True） |
| 源字段 / 长度 | `markdown` ｜ 10749 chars ｜ 15670 bytes(UTF-8) ｜ 239 行 |
| 源 sha256 | `sha256:b2c5eb5dad0b6db25574338b37aa5be1c8fdf20bbc7d3c75640e1c1727756841` |
| 契约基线 | 同值（hash 一致=True，chars 一致=True） |
| 交付包 / 类型 / lane | `wp-9632-01` / `01-asset-registration` / B |
| 生产脚本 | `scripts/kf_wp9632_01_register.py`（producer `kf-worker/m18694`） |

**登记口径（三态，互斥完备）**：`resolved_in_source`＝工作树内真实可达（仅源 markdown 自身）；`metadata_only`＝平台只暴露元数据、无字节载荷（6 张声明图）；`referenced_unresolved`＝正文声明的跨仓/跨目录路径，只登记不断言。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

| 步 | 动作 | 关键设置 |
|---|---|---|
| 1 | 取原文（唯一来源） | `POST /auth/login` → `GET /api/v1/tensor/reports/9632` 的 `markdown` 字段 |
| 2 | 源身份实测 | `hashlib.sha256(md.encode('utf-8'))`，与契约值比对后再落盘（不一致直接 SystemExit，不静默换源） |
| 3 | 字节副本 | `runs/wp-9632-01/source_9632.md`（本轮**入库**，见 ⑤-7） |
| 4 | 引用扫描 | 三路正则：markdown 图片 `![]()`、反引号路径令牌（含命令行拆词）、反引号点号标识符；另有 `Tensor[A-Z]…` 仓库令牌 |
| 5 | 定位器 | 每条资产 `{quote, start_char, end_char, start_line, end_line, unique}`，`md[start:end]==quote` 可行机器复核 |
| 6 | 解析判定 | 工作树内 `Path.exists()` + 同名文件命中证据；跨仓一律 unresolved |
| 7 | 平台图探针 | 逐图 `GET /tensor/reports/9632/figures/<name>`，仅记 HTTP 状态 |
| 8 | 结构门 | 清单 vs `schemas/asset_manifest.v1.wp9632-01.schema.json`（Draft-07，由 448 版派生仅改身份 const）；信封 vs `schemas/kf_result.v1.schema.json` |
| 9 | 六项验收 | `register.py verify` → `verification.json`（44 条定位器全部重解析） |
| 10 | 出图 + 交付 | `register.py figure`（定位图）；`publish.py` 渲染 KF-RESULT，`--check <id>` 事后核对线上副本 |

工具链：Python 3.11 标准库（urllib / hashlib / json / re）+ `jsonschema` + `matplotlib(Agg)`；全部产物落工作树（`docs/kf/wp-9632-01/`、`runs/wp-9632-01/`、`scripts/`、`tests/`），PR 共 **10 个文件 / +5146 行**，不触碰任何共享核心模块（无 `ke/`、无 `schemas/kf_result.v1` 改动）。

repro_cmd: `python3 scripts/kf_wp9632_01_register.py all && python3 scripts/kf_wp9632_01_publish.py --dry-run && pytest tests/test_kf_wp9632_01.py -q`

## ③ 计算结果（程序化注入的数字表）

**表 1 · 资产登记（25 条，按 kind/status）**

| kind / status | 条数 |
|---|---|
| `report_markdown/resolved` | 1 |
| `embedded_figure/metadata_only` | 6 |
| `source_code_reference/unresolved` | 3 |
| `data_artifact_reference/unresolved` | 2 |
| `figure_reference/unresolved` | 12 |
| `directory_reference/unresolved` | 1 |
| **合计** | **25**（resolved 1 / metadata_only 6 / unresolved 18） |

正文引用去重后 distinct 路径 **17**；记录声明图 **6** 张，其中带字节载荷 **0** 张；清册覆盖 **12 张表 / 60 行**、**7 个章节**。claims=0（契约要求空）。

**表 2 · 完整性会计**

| 指标 | 值 | 判定 |
|---|---|---|
| 定位器回环 | 44 / 44 | PASS |
| asset_id / label 重复 | 0 | PASS |
| 引文唯一命中 | 单次 9 / 复现 16（复现者靠 start_line+char_span 唯一） | 已标注 `unique=false` |
| 清单规范化字节 | `sha256:1a8beb586a7f75fc5a7bb955`（信封内嵌同值） | PASS |…
| 源 hash 线上重算 vs 契约 | 相同（差异 0 字节） | PASS |

**表 3 · 验收与交付一致性**

| 门 | 结果 | 证据 |
|---|---|---|
| `valid_json` | PASS | 自检 ok |
| `schema_pass` | PASS | 自检 ok |
| `source_hash_matches` | PASS | 自检 ok |
| `source_locator_resolvable` | PASS | 自检 ok |
| `no_duplicate_ids` | PASS | 自检 ok |
| `all_claims_have_evidence_or_explicit_unresolved_status` | PASS | 自检 ok |
| 平台线上副本 vs 交付包（事后核对 17389） | 9/9 PASS | 报告 17389 的 markdown 与 `KF-RESULT-wp-9632-01.md` 逐字节相同（sha256 `27e0f4ed2f8fca27…`），内嵌 json 块 == `kf_result.json` |
| 复算可重现性（`register all` 重跑） | PASS | 4 个文件、10 行差异，全部为 `generated_at/checked_at` 与随其变化的清单 sha；语义层洗掉时间戳后**完全相等**；PNG 0 字节差 = 0 |
| 离线不变量测试 | 19 passed / 0 failed | `pytest tests/test_kf_wp9632_01.py -q`（含定位器与发布核对的篡改负控） |

## ④ 结果图

本工单是**登记/文档型**交付，不产生物理场量，因此没有场量云图；替代证据是两张**可复算**图：fig1 由 `register.py figure` 从 manifest 的字符坐标直接绘出，fig2 由 `verification.json` 逐项绘制。

- **fig1 `wp9632_01_source_locator_map.png`（已入库）**：上panel＝源 markdown 239 行的行长曲线 + 25 条资产的定位行（颜色=kind，黄带=12 张表），下panel＝kind/resolution 直方图。
- **fig2 `wp9632_01_verification_dashboard.png`（本报告附图）**：(a) 25 条资产的 kind/resolution 条形；(b) 验收 6/6 + 发布核对 9/9 全绿矩阵；(c) 实测 vs 基线的严格相等对照表（误差 0%）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **平台不提供图的字节载荷**：`record.figures` 仅 name/caption，逐图端点探测无 200 载荷；已发布记录 17389 的 `figures[0].png_base64` 长度为 0（图片未被平台留存）。故 6 条图资产止于 `metadata_only`，**不声明任何图 hash**。
2. **跨仓只登记不断言**：正文声明的 TensorPhysics 侧路径（`src/tensorphysics/...`、`benchmarks/ice_ridge_consolidation/*`、`tests/test_ice_ridge_consolidation.py` 等 18 条）在交付仓内探测不到，一律 `referenced_unresolved` 并记录同名命中情况；未跨仓读取。
3. **引文唯一性不是全量成立**：单次命中 9/25，其余在正文中重复出现，已显式标 `unique=false`，靠 `start_line`+`char_span` 保证机器可复核的唯一定位。
4. **同一张图两行资产是有意的**：一行钉平台元数据（name/caption+端点探针），一行钉正文里的磁盘相对路径，可用性证据不同，合并会丢掉一类证据。
5. **复算会改时间戳**：`generated_at` / `checked_at` 每次重跑都变，随之清单 sha 变；因此 repo 里保留的是**与线上发布逐字节一致**的那一版，重跑只用于验证语义等价（本次已实测：仅 10 行差异、PNG 0 字节差）。
6. **不做二次手抄**：报告正文的物理判据数值（例如判据 (a)(b) 的相对误差）属下游抽取/复核工序，本包不复制、不复述。
7. **本轮实际修复的缺陷**：源字节副本原先位于未跟踪的 `runs/` 区，导致定位器/hash 无源可依时**静默 skip**（干净 checkout 下 14 passed + 1 failed/skip，定位器测试被跳过 = 「全部定位器可解析」不可伪），本轮把副本入库并加 `test_source_copy_is_committed` 守门，现 **19/19 全跑**；同时补上 `publish.py --check <report_id>` 事后门（原实现缺此回路），把「repo 交付物 == 线上副本」变成可复跑检查（线上实测 9/9）。
8. **未做的**：未重发 KF-RESULT 报告（同一交付物已有 17389，重复发布只会造噪；一致性改由 `--check` 证明）；`ke/__pycache__/*.pyc` 的 0 行二进制噪声（原 PR 的脏文件）不纳入本次提交。
9. **权威边界**：`verification.json` 是**生产脚本自检**（与 build 共享代码），不是验证结论；证据等级与验收以平台门重算为准。

## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）

本工序**没有物理基准量**（不对标实验/文献值），因此不存在物理百分比误差可比——这是事实性缺省，不是省略。唯一可比的量是**身份一致性**，且它按**严格相等**判定（比 3% 更严）：

| 关键量 | 实测 | 基准 | 相对误差 | < 3% ? |
|---|---|---|---|---|
| 源 sha256（线上重算） | `b2c5eb5dad0b…` | `b2c5eb5dad0b…` | 0 字节差异 | **是（0%）** |
| 源字符数 | 10749 | 契约 10749 | 0 | **是（0%）** |
| 源字节数(UTF-8) / 行数 | 15670 / 239 | 副本同值 | 0 / 0 | **是（0%）** |
| 定位器回环失败数 | 0 / 44 | 0 | 0% | **是（0%）** |
| 结构门错误数（清单 + 信封） | 0 | 0 | 0 | **是（0%）** |
| 线上副本 vs 交付包 | 9/9 | 全等 | 0 字节 | **是（0%）** |
| 离线不变量测试 | 19/19 | 全通过 | 0 | **是（0%）** |

**结论**：所有可比项误差均为 **0**，逐项 **< 3% 成立**（实为严格相等）。报告正文的物理判据（Stefan 固结层厚度、Maxwell-Eucken 有效导热率、阻力增量等）其精度判定属下游抽取/复核工单，本包**不冒充**该结论；如需对该报告的物理误差表态，应另开复核工单并给出实验/文献基准。

## ⑦ 交付物与审计链

| file | bytes |
|---|---|
| `docs/kf/wp-9632-01/KF-RESULT-wp-9632-01.md` | 49895 |
| `docs/kf/wp-9632-01/asset_manifest.json` | 35190 |
| `docs/kf/wp-9632-01/figures/wp9632_01_source_locator_map.png` | 213048 |
| `docs/kf/wp-9632-01/kf_result.json` | 39737 |
| `docs/kf/wp-9632-01/verification.json` | 2294 |
| `runs/wp-9632-01/source_9632.md` | 15670 |
| `schemas/asset_manifest.v1.wp9632-01.schema.json` | 6049 |
| `scripts/kf_wp9632_01_register.py` | 38648 |
| `scripts/kf_wp9632_01_publish.py` | 19520 |
| `tests/test_kf_wp9632_01.py` | 14122 |

审计链：源字节 → 定位器 → 清单 → 信封（内嵌清单 + 源 hash）→ KF-RESULT markdown（唯一 json 块）→ 线上报告 17389（`--check` 逐字节核对）。任一环改动都会让 `verify` 或 `--check` 直接 FAIL。
